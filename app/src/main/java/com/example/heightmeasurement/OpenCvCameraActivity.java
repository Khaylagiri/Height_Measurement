package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseLandmark;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.DetectorParameters;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class OpenCvCameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "HEIGHT_MEASURE_CAMERA";

    private JavaCameraView javaCameraView;
    private FrameLayout btnCapture;

    private final Object frameLock = new Object();
    private Mat latestFrame;
    private boolean isSaving = false;

    private PoseDetector poseDetector;

    private Bitmap warpedBaseBitmap;
    private Bitmap currentBitmap;

    private final List<MarkerData> detectedMarkers = new ArrayList<>();
    private final List<ReferencePoint> visibleReferencePoints = new ArrayList<>();

    private static final int MARKER_DICT = Objdetect.DICT_6X6_1000;
    private static final int MAX_WARP_W = 2200;
    private static final int MAX_WARP_H = 3200;

    private static final double MIN_VALID_HEIGHT_CM = 100.0;
    private static final double MAX_VALID_HEIGHT_CM = 220.0;
    private static final double MAX_ALLOWED_REF_RMSE_CM = 20.0;

    private static final Map<Integer, Double> REFERENCE_HEIGHTS_CM = new HashMap<>();
    static {
        REFERENCE_HEIGHTS_CM.put(143, 0.0);
        REFERENCE_HEIGHTS_CM.put(15, 85.0);
        REFERENCE_HEIGHTS_CM.put(134, 155.0);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);

        javaCameraView = findViewById(R.id.javaCameraView);
        btnCapture = findViewById(R.id.btnCapture);

        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        javaCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        javaCameraView.setCameraPermissionGranted();

        PoseDetectorOptions options = new PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
                .build();
        poseDetector = PoseDetection.getClient(options);

        btnCapture.setOnClickListener(v -> capturePhotoAndMeasure());
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (OpenCVLoader.initLocal()) {
            Log.d("OpenCV", "OpenCV loaded successfully");
            javaCameraView.enableView();
        } else {
            Log.e("OpenCV", "OpenCV initialization failed");
            Toast.makeText(this, "OpenCV gagal diinisialisasi", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (javaCameraView != null) {
            javaCameraView.disableView();
        }

        if (poseDetector != null) {
            poseDetector.close();
        }

        synchronized (frameLock) {
            if (latestFrame != null) {
                latestFrame.release();
                latestFrame = null;
            }
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d("OpenCV", "Camera started: " + width + " x " + height);
    }

    @Override
    public void onCameraViewStopped() {
        Log.d("OpenCV", "Camera stopped");
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgbaFrame = inputFrame.rgba();

        String line1 = new SimpleDateFormat(
                "EEEE, dd MMMM yyyy",
                new Locale("id", "ID")
        ).format(new Date());

        String line2 = new SimpleDateFormat(
                "HH:mm:ss",
                new Locale("id", "ID")
        ).format(new Date());

        drawTimestampLikeCamera(rgbaFrame, line1, line2);

        synchronized (frameLock) {
            if (latestFrame != null) {
                latestFrame.release();
            }
            latestFrame = rgbaFrame.clone();
        }

        return rgbaFrame;
    }

    private void capturePhotoAndMeasure() {
        if (isSaving) {
            Toast.makeText(this, "Sedang memproses...", Toast.LENGTH_SHORT).show();
            return;
        }

        Mat capturedMat;

        synchronized (frameLock) {
            if (latestFrame == null || latestFrame.empty()) {
                Toast.makeText(this, "Frame kamera belum tersedia", Toast.LENGTH_SHORT).show();
                return;
            }
            capturedMat = latestFrame.clone();
        }

        isSaving = true;
        Toast.makeText(this, "Mengambil dan memproses foto...", Toast.LENGTH_SHORT).show();

        new Thread(() -> {
            try {
                Bitmap capturedBitmap = matToBitmap(capturedMat);
                runFullMeasurementPipeline(capturedBitmap);
            } catch (Exception e) {
                Log.e(TAG, "capturePhotoAndMeasure error", e);
                runOnUiThread(() ->
                        Toast.makeText(
                                OpenCvCameraActivity.this,
                                "Proses gagal: " + e.getMessage(),
                                Toast.LENGTH_LONG
                        ).show()
                );
            } finally {
                capturedMat.release();
                isSaving = false;
            }
        }).start();
    }

    private void runFullMeasurementPipeline(Bitmap originalBitmap) {
        try {
            Bitmap warped = autoPerspectiveFromTwoBoards(originalBitmap);
            if (warped == null) {
                runOnUiThread(() ->
                        Toast.makeText(this, "Gagal auto perspective", Toast.LENGTH_SHORT).show()
                );
                return;
            }

            ArucoResult arucoResult = detectArucoAndDrawIds(warped);
            if (arucoResult == null || arucoResult.markers == null || arucoResult.markers.isEmpty()) {
                runOnUiThread(() ->
                        Toast.makeText(this, "Deteksi ArUco gagal setelah perspective", Toast.LENGTH_SHORT).show()
                );
                return;
            }

            detectedMarkers.clear();
            detectedMarkers.addAll(arucoResult.markers);

            visibleReferencePoints.clear();
            visibleReferencePoints.addAll(extractVisibleReferencePoints(detectedMarkers));

            logAllDetectedIds(detectedMarkers);
            logReferenceIds(visibleReferencePoints);

            warpedBaseBitmap = arucoResult.bitmapWithIds;
            warpedBaseBitmap = addTimestampToBitmap(warpedBaseBitmap);
            warpedBaseBitmap = drawReferenceScaleInfo(warpedBaseBitmap, visibleReferencePoints);

            CalibrationModel calibration = buildCalibrationModel(visibleReferencePoints);
            if (calibration == null) {
                runOnUiThread(() ->
                        Toast.makeText(
                                this,
                                "Kalibrasi marker referensi belum valid. Cek ID marker referensi.",
                                Toast.LENGTH_LONG
                        ).show()
                );
                return;
            }

            runPoseMeasurement(warpedBaseBitmap, calibration);

        } catch (Exception e) {
            Log.e(TAG, "runFullMeasurementPipeline error", e);
            runOnUiThread(() ->
                    Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show()
            );
        }
    }

    private void runPoseMeasurement(Bitmap bitmap, CalibrationModel calibration) {
        try {
            InputImage inputImage = InputImage.fromBitmap(bitmap, 0);

            poseDetector.process(inputImage)
                    .addOnSuccessListener(new OnSuccessListener<Pose>() {
                        @Override
                        public void onSuccess(Pose pose) {
                            PersonPoints points = extractHeadAndFootFromPose(pose);
                            if (points == null) {
                                Toast.makeText(OpenCvCameraActivity.this,
                                        "Pose tubuh tidak terdeteksi dengan baik.",
                                        Toast.LENGTH_LONG).show();
                                return;
                            }

                            boolean success = measureHeightFromPoints(points.head, points.foot, calibration);
                            if (!success) {
                                Toast.makeText(OpenCvCameraActivity.this,
                                        "Pengukuran tidak valid. Cek ID marker referensi.",
                                        Toast.LENGTH_LONG).show();
                            }
                        }
                    })
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            Log.e(TAG, "Pose detection failed", e);
                            Toast.makeText(OpenCvCameraActivity.this,
                                    "Pose detection gagal.",
                                    Toast.LENGTH_LONG).show();
                        }
                    });

        } catch (Exception e) {
            Log.e(TAG, "runPoseMeasurement error", e);
            runOnUiThread(() ->
                    Toast.makeText(this, "Pose error: " + e.getMessage(), Toast.LENGTH_LONG).show()
            );
        }
    }

    private boolean measureHeightFromPoints(Point headPoint, Point footPoint, CalibrationModel calibration) {
        Double headCm = yPixelToCm(headPoint.y, calibration);
        Double footCm = yPixelToCm(footPoint.y, calibration);

        Log.d(TAG, "headY=" + headPoint.y + ", footY=" + footPoint.y);
        Log.d(TAG, "headCm=" + headCm + ", footCm=" + footCm);

        if (headCm == null || footCm == null) return false;

        double heightCm = Math.abs(footCm - headCm);
        Log.d(TAG, "heightCm=" + heightCm);

        if (heightCm < MIN_VALID_HEIGHT_CM || heightCm > MAX_VALID_HEIGHT_CM) {
            Log.e(TAG, "Height out of range = " + heightCm + " cm");
            return false;
        }

        currentBitmap = drawMeasurementOnBitmap(warpedBaseBitmap, headPoint, footPoint, heightCm);
        saveBitmapToAppFiles(currentBitmap);

        runOnUiThread(() ->
                Toast.makeText(this, "Tinggi = " + format1(heightCm) + " cm", Toast.LENGTH_LONG).show()
        );

        return true;
    }

    private Bitmap matToBitmap(Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        return bitmap;
    }

    private CalibrationModel buildCalibrationModel(List<ReferencePoint> refs) {
        if (refs == null || refs.size() < 2) return null;

        double sumY = 0.0;
        double sumCm = 0.0;
        double sumYY = 0.0;
        double sumYCm = 0.0;
        int n = refs.size();

        for (ReferencePoint r : refs) {
            sumY += r.y;
            sumCm += r.cm;
            sumYY += r.y * r.y;
            sumYCm += r.y * r.cm;
        }

        double denom = n * sumYY - sumY * sumY;
        if (Math.abs(denom) < 1e-6) return null;

        double a = (n * sumYCm - sumY * sumCm) / denom;
        double b = (sumCm - a * sumY) / n;

        double rmse = 0.0;
        for (ReferencePoint r : refs) {
            double pred = a * r.y + b;
            double err = pred - r.cm;
            rmse += err * err;
        }
        rmse = Math.sqrt(rmse / n);

        double minCm = Double.MAX_VALUE;
        double maxCm = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;

        for (ReferencePoint r : refs) {
            minCm = Math.min(minCm, r.cm);
            maxCm = Math.max(maxCm, r.cm);
            minY = Math.min(minY, r.y);
            maxY = Math.max(maxY, r.y);
        }

        Log.d(TAG, "Calibration a=" + a + " b=" + b + " rmse=" + rmse);

        if (rmse > MAX_ALLOWED_REF_RMSE_CM) return null;
        if (Math.abs(maxY - minY) < 80.0) return null;
        if (Math.abs(maxCm - minCm) < 30.0) return null;

        return new CalibrationModel(a, b, minY, maxY, minCm, maxCm);
    }

    private Double yPixelToCm(double y, CalibrationModel model) {
        if (model == null) return null;

        double cm = model.a * y + model.b;

        if (cm < model.minCm - 40.0 || cm > model.maxCm + 40.0) {
            return null;
        }

        return cm;
    }

    private PersonPoints extractHeadAndFootFromPose(Pose pose) {
        PoseLandmark nose = pose.getPoseLandmark(PoseLandmark.NOSE);
        PoseLandmark leftEye = pose.getPoseLandmark(PoseLandmark.LEFT_EYE);
        PoseLandmark rightEye = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE);
        PoseLandmark leftEar = pose.getPoseLandmark(PoseLandmark.LEFT_EAR);
        PoseLandmark rightEar = pose.getPoseLandmark(PoseLandmark.RIGHT_EAR);

        PoseLandmark leftHeel = pose.getPoseLandmark(PoseLandmark.LEFT_HEEL);
        PoseLandmark rightHeel = pose.getPoseLandmark(PoseLandmark.RIGHT_HEEL);
        PoseLandmark leftFoot = pose.getPoseLandmark(PoseLandmark.LEFT_FOOT_INDEX);
        PoseLandmark rightFoot = pose.getPoseLandmark(PoseLandmark.RIGHT_FOOT_INDEX);
        PoseLandmark leftAnkle = pose.getPoseLandmark(PoseLandmark.LEFT_ANKLE);
        PoseLandmark rightAnkle = pose.getPoseLandmark(PoseLandmark.RIGHT_ANKLE);

        Double headY = minValidY(
                getLandmarkY(nose),
                getLandmarkY(leftEye),
                getLandmarkY(rightEye),
                getLandmarkY(leftEar),
                getLandmarkY(rightEar)
        );

        Double headX = avgValidX(
                getLandmarkX(nose),
                getLandmarkX(leftEye),
                getLandmarkX(rightEye),
                getLandmarkX(leftEar),
                getLandmarkX(rightEar)
        );

        Double footY = maxValidY(
                getLandmarkY(leftHeel),
                getLandmarkY(rightHeel),
                getLandmarkY(leftFoot),
                getLandmarkY(rightFoot),
                getLandmarkY(leftAnkle),
                getLandmarkY(rightAnkle)
        );

        Double footX = avgValidX(
                getLandmarkX(leftHeel),
                getLandmarkX(rightHeel),
                getLandmarkX(leftFoot),
                getLandmarkX(rightFoot),
                getLandmarkX(leftAnkle),
                getLandmarkX(rightAnkle)
        );

        if (headY == null || headX == null || footY == null || footX == null) {
            return null;
        }

        headY = Math.max(0.0, headY - 4.0);

        if (footY <= headY) return null;

        return new PersonPoints(new Point(headX, headY), new Point(footX, footY));
    }

    private ArucoResult detectArucoAndDrawIds(Bitmap bitmap) {
        Mat mat = new Mat();
        Mat gray = new Mat();
        Mat ids = new Mat();
        List<Mat> corners = new ArrayList<>();
        List<Mat> rejected = new ArrayList<>();

        try {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(mutableBitmap, mat);

            if (mat.channels() == 4) {
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (mat.channels() == 3) {
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = mat.clone();
            }

            Imgproc.equalizeHist(gray, gray);

            Dictionary dictionary = Objdetect.getPredefinedDictionary(MARKER_DICT);
            DetectorParameters parameters = new DetectorParameters();
            parameters.set_adaptiveThreshWinSizeMin(3);
            parameters.set_adaptiveThreshWinSizeMax(53);
            parameters.set_adaptiveThreshWinSizeStep(4);
            parameters.set_minMarkerPerimeterRate(0.005);
            parameters.set_maxMarkerPerimeterRate(4.0);
            parameters.set_polygonalApproxAccuracyRate(0.08);
            parameters.set_minCornerDistanceRate(0.005);
            parameters.set_minDistanceToBorder(1);
            parameters.set_cornerRefinementMethod(1);
            parameters.set_cornerRefinementWinSize(7);
            parameters.set_cornerRefinementMaxIterations(80);
            parameters.set_cornerRefinementMinAccuracy(0.01);

            ArucoDetector detector = new ArucoDetector(dictionary, parameters);
            detector.detectMarkers(gray, corners, ids, rejected);

            List<MarkerData> markers = new ArrayList<>();

            if (!ids.empty()) {
                for (int i = 0; i < ids.rows(); i++) {
                    int id = (int) ids.get(i, 0)[0];
                    Mat cornerMat = corners.get(i);

                    double[] xy0 = cornerMat.get(0, 0);
                    double[] xy1 = cornerMat.get(0, 1);
                    double[] xy2 = cornerMat.get(0, 2);
                    double[] xy3 = cornerMat.get(0, 3);

                    Point p0 = new Point(xy0[0], xy0[1]);
                    Point p1 = new Point(xy1[0], xy1[1]);
                    Point p2 = new Point(xy2[0], xy2[1]);
                    Point p3 = new Point(xy3[0], xy3[1]);

                    double s1 = distance(p0, p1);
                    double s2 = distance(p1, p2);
                    double s3 = distance(p2, p3);
                    double s4 = distance(p3, p0);
                    double avgSide = (s1 + s2 + s3 + s4) / 4.0;

                    MarkerData marker = new MarkerData(id, p0, p1, p2, p3, avgSide);
                    markers.add(marker);
                    drawMarker(mat, marker);
                }
            }

            Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, resultBitmap);

            return new ArucoResult(resultBitmap, markers);

        } catch (Exception e) {
            Log.e(TAG, "detectArucoAndDrawIds error", e);
            return null;
        } finally {
            for (Mat c : corners) c.release();
            for (Mat r : rejected) r.release();
            ids.release();
            gray.release();
            mat.release();
        }
    }

    private Bitmap autoPerspectiveFromTwoBoards(Bitmap bitmap) {
        Mat srcMat = new Mat();
        Mat gray = new Mat();
        Mat warped = new Mat();

        try {
            Utils.bitmapToMat(bitmap, srcMat);

            if (srcMat.channels() == 4) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (srcMat.channels() == 3) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = srcMat.clone();
            }

            Point[] boardCorners = detectTwoBoardCorners(gray);
            if (boardCorners == null) return null;

            Point tl = boardCorners[0];
            Point tr = boardCorners[1];
            Point br = boardCorners[2];
            Point bl = boardCorners[3];

            double widthTop = distance(tl, tr);
            double widthBottom = distance(bl, br);
            double maxWidth = Math.max(widthTop, widthBottom);

            double heightLeft = distance(tl, bl);
            double heightRight = distance(tr, br);
            double maxHeight = Math.max(heightLeft, heightRight);

            double scale = Math.min(
                    (double) MAX_WARP_W / Math.max(1.0, maxWidth),
                    (double) MAX_WARP_H / Math.max(1.0, maxHeight)
            );

            if (scale < 1.0) {
                maxWidth = Math.max(800, Math.round(maxWidth * scale));
                maxHeight = Math.max(1600, Math.round(maxHeight * scale));
            }

            MatOfPoint2f srcPoints = new MatOfPoint2f(tl, tr, br, bl);
            MatOfPoint2f dstPoints = new MatOfPoint2f(
                    new Point(0, 0),
                    new Point(maxWidth - 1, 0),
                    new Point(maxWidth - 1, maxHeight - 1),
                    new Point(0, maxHeight - 1)
            );

            Mat H = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Imgproc.warpPerspective(srcMat, warped, H, new Size(maxWidth, maxHeight));

            Bitmap result = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(warped, result);

            srcPoints.release();
            dstPoints.release();
            H.release();

            return result;

        } catch (Exception e) {
            Log.e(TAG, "autoPerspectiveFromTwoBoards error", e);
            return null;
        } finally {
            gray.release();
            srcMat.release();
            warped.release();
        }
    }

    private Point[] detectTwoBoardCorners(Mat gray) {
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        List<Mat> rejected = new ArrayList<>();

        try {
            Dictionary dictionary = Objdetect.getPredefinedDictionary(MARKER_DICT);
            DetectorParameters parameters = new DetectorParameters();
            parameters.set_cornerRefinementMethod(1);
            parameters.set_cornerRefinementWinSize(7);
            parameters.set_cornerRefinementMaxIterations(80);
            parameters.set_cornerRefinementMinAccuracy(0.01);

            ArucoDetector detector = new ArucoDetector(dictionary, parameters);
            detector.detectMarkers(gray, corners, ids, rejected);

            if (ids.empty() || corners.size() < 8) return null;

            List<Point> centers = new ArrayList<>();
            double avgMarkerSide = 0.0;

            for (Mat cornerMat : corners) {
                double[] xy0 = cornerMat.get(0, 0);
                double[] xy1 = cornerMat.get(0, 1);
                double[] xy2 = cornerMat.get(0, 2);
                double[] xy3 = cornerMat.get(0, 3);

                double cx = (xy0[0] + xy1[0] + xy2[0] + xy3[0]) / 4.0;
                double cy = (xy0[1] + xy1[1] + xy2[1] + xy3[1]) / 4.0;
                centers.add(new Point(cx, cy));

                double s1 = distance(new Point(xy0[0], xy0[1]), new Point(xy1[0], xy1[1]));
                double s2 = distance(new Point(xy1[0], xy1[1]), new Point(xy2[0], xy2[1]));
                double s3 = distance(new Point(xy2[0], xy2[1]), new Point(xy3[0], xy3[1]));
                double s4 = distance(new Point(xy3[0], xy3[1]), new Point(xy0[0], xy0[1]));
                avgMarkerSide += (s1 + s2 + s3 + s4) / 4.0;
            }

            avgMarkerSide /= Math.max(1, corners.size());

            int sidePad = Math.max(80, (int) Math.round(avgMarkerSide * 1.2));
            int topPad = Math.max(60, (int) Math.round(avgMarkerSide * 0.8));
            int bottomPad = Math.max(380, (int) Math.round(avgMarkerSide * 5.0));

            double medianY = computeMedianY(centers);

            List<Point> topPoints = new ArrayList<>();
            List<Point> bottomPoints = new ArrayList<>();

            for (Mat cornerMat : corners) {
                double[] xy0 = cornerMat.get(0, 0);
                double[] xy1 = cornerMat.get(0, 1);
                double[] xy2 = cornerMat.get(0, 2);
                double[] xy3 = cornerMat.get(0, 3);

                double cy = (xy0[1] + xy1[1] + xy2[1] + xy3[1]) / 4.0;

                if (cy < medianY) {
                    topPoints.add(new Point(xy0[0], xy0[1]));
                    topPoints.add(new Point(xy1[0], xy1[1]));
                    topPoints.add(new Point(xy2[0], xy2[1]));
                    topPoints.add(new Point(xy3[0], xy3[1]));
                } else {
                    bottomPoints.add(new Point(xy0[0], xy0[1]));
                    bottomPoints.add(new Point(xy1[0], xy1[1]));
                    bottomPoints.add(new Point(xy2[0], xy2[1]));
                    bottomPoints.add(new Point(xy3[0], xy3[1]));
                }
            }

            if (topPoints.size() < 4 || bottomPoints.size() < 4) return null;

            Rect topRect = boundingRectCustom(topPoints, sidePad, topPad, sidePad, sidePad, gray.cols(), gray.rows());
            Rect bottomRect = boundingRectCustom(bottomPoints, sidePad, sidePad, sidePad, bottomPad, gray.cols(), gray.rows());

            if (topRect == null || bottomRect == null) return null;

            Point tl = new Point(topRect.x, topRect.y);
            Point tr = new Point(topRect.x + topRect.width, topRect.y);
            Point br = new Point(bottomRect.x + bottomRect.width, bottomRect.y + bottomRect.height);
            Point bl = new Point(bottomRect.x, bottomRect.y + bottomRect.height);

            return new Point[]{tl, tr, br, bl};

        } catch (Exception e) {
            Log.e(TAG, "detectTwoBoardCorners error", e);
            return null;
        } finally {
            for (Mat c : corners) c.release();
            for (Mat r : rejected) r.release();
            ids.release();
        }
    }

    private List<ReferencePoint> extractVisibleReferencePoints(List<MarkerData> markers) {
        List<ReferencePoint> refs = new ArrayList<>();

        for (MarkerData m : markers) {
            if (REFERENCE_HEIGHTS_CM.containsKey(m.id)) {
                refs.add(new ReferencePoint(
                        m.id,
                        m.getCenter().y,
                        REFERENCE_HEIGHTS_CM.get(m.id),
                        m.getCenter()
                ));
            }
        }

        refs.sort(Comparator.comparingDouble(o -> o.y));
        return refs;
    }

    private Bitmap drawMeasurementOnBitmap(Bitmap bitmap, Point head, Point foot, double heightCm) {
        try {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Mat mat = new Mat();
            Utils.bitmapToMat(mutableBitmap, mat);

            Point verticalFoot = new Point(head.x, foot.y);

            Imgproc.line(mat, head, verticalFoot, new Scalar(0, 255, 255, 255), 6);
            Imgproc.circle(mat, head, 12, new Scalar(0, 0, 255, 255), -1);
            Imgproc.circle(mat, foot, 12, new Scalar(255, 0, 0, 255), -1);

            putHeaderText(mat, "HASIL PENGUKURAN");
            putHeaderText2(mat, "TINGGI BADAN: " + format1(heightCm) + " cm");

            Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, resultBitmap);
            mat.release();
            return resultBitmap;

        } catch (Exception e) {
            Log.e(TAG, "drawMeasurementOnBitmap error", e);
            return bitmap;
        }
    }

    private Bitmap drawReferenceScaleInfo(Bitmap bitmap, List<ReferencePoint> refs) {
        try {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Mat mat = new Mat();
            Utils.bitmapToMat(mutableBitmap, mat);

            for (ReferencePoint ref : refs) {
                putOutlinedTextMat(
                        mat,
                        "ID: " + ref.id + " (H: " + format1(ref.cm) + "cm)",
                        new Point(ref.center.x + 30, ref.center.y),
                        0.7,
                        new Scalar(255, 255, 255, 255),
                        new Scalar(20, 40, 70, 255)
                );
            }

            Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, resultBitmap);
            mat.release();
            return resultBitmap;

        } catch (Exception e) {
            Log.e(TAG, "drawReferenceScaleInfo error", e);
            return bitmap;
        }
    }

    private Bitmap addTimestampToBitmap(Bitmap bitmap) {
        try {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Mat mat = new Mat();
            Utils.bitmapToMat(mutableBitmap, mat);

            String line1 = new SimpleDateFormat(
                    "EEEE, dd MMMM yyyy",
                    new Locale("id", "ID")
            ).format(new Date());

            String line2 = new SimpleDateFormat(
                    "HH:mm:ss",
                    new Locale("id", "ID")
            ).format(new Date());

            drawTimestampLikeCamera(mat, line1, line2);

            Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, resultBitmap);
            mat.release();

            return resultBitmap;
        } catch (Exception e) {
            Log.e(TAG, "addTimestampToBitmap error", e);
            return bitmap;
        }
    }

    private void drawTimestampLikeCamera(Mat mat, String line1, String line2) {
        int left = 30;
        int line1Y = mat.rows() - 120;
        int line2Y = mat.rows() - 55;

        double fontScale = 1.2;
        int shadowThickness = 6;
        int textThickness = 2;

        Imgproc.putText(mat, line1, new Point(left, line1Y),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                new Scalar(0, 0, 0, 255), shadowThickness);

        Imgproc.putText(mat, line2, new Point(left, line2Y),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                new Scalar(0, 0, 0, 255), shadowThickness);

        Imgproc.putText(mat, line1, new Point(left, line1Y),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                new Scalar(255, 255, 255, 255), textThickness);

        Imgproc.putText(mat, line2, new Point(left, line2Y),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                new Scalar(255, 255, 255, 255), textThickness);
    }

    private void putHeaderText(Mat mat, String text) {
        Point pos = new Point(mat.cols() * 0.12, 65);
        Imgproc.rectangle(
                mat,
                new Point(pos.x - 20, pos.y - 40),
                new Point(pos.x + 600, pos.y + 18),
                new Scalar(255, 255, 255, 220),
                -1
        );
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
                new Scalar(20, 20, 20, 255), 3);
    }

    private void putHeaderText2(Mat mat, String text) {
        Point pos = new Point(mat.cols() * 0.12, 112);
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
                new Scalar(20, 20, 20, 255), 3);
    }

    private void putOutlinedTextMat(Mat mat, String text, Point pos, double scale, Scalar fg, Scalar bg) {
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, scale, bg, 6);
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, scale, fg, 2);
    }

    private void drawMarker(Mat mat, MarkerData marker) {
        Imgproc.line(mat, marker.p0, marker.p1, new Scalar(0, 255, 0, 255), 3);
        Imgproc.line(mat, marker.p1, marker.p2, new Scalar(0, 255, 0, 255), 3);
        Imgproc.line(mat, marker.p2, marker.p3, new Scalar(0, 255, 0, 255), 3);
        Imgproc.line(mat, marker.p3, marker.p0, new Scalar(0, 255, 0, 255), 3);

        Point center = marker.getCenter();
        Imgproc.circle(mat, center, 6, new Scalar(0, 0, 255, 255), -1);

        String label = "ID:" + marker.id;
        if (REFERENCE_HEIGHTS_CM.containsKey(marker.id)) {
            label += " (H:" + format1(REFERENCE_HEIGHTS_CM.get(marker.id)) + "cm)";
        }

        putOutlinedTextMat(
                mat,
                label,
                new Point(marker.p0.x + 5, Math.max(30, marker.p0.y - 10)),
                0.7,
                new Scalar(255, 255, 255, 255),
                new Scalar(20, 40, 70, 255)
        );
    }

    private void saveBitmapToAppFiles(Bitmap bitmap) {
        try {
            File picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
            if (picturesDir != null && !picturesDir.exists()) {
                picturesDir.mkdirs();
            }

            String fileName = new SimpleDateFormat(
                    "yyyyMMdd_HHmmss",
                    Locale.getDefault()
            ).format(new Date());

            File imageFile = new File(picturesDir, "CAM_" + fileName + ".jpg");

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos);
            fos.flush();
            fos.close();

            Log.d(TAG, "Saved: " + imageFile.getAbsolutePath());

            runOnUiThread(() ->
                    Toast.makeText(this, "Gambar berhasil disimpan", Toast.LENGTH_SHORT).show()
            );

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToAppFiles error", e);
            runOnUiThread(() ->
                    Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show()
            );
        }
    }

    private Double getLandmarkX(PoseLandmark l) {
        return l == null ? null : (double) l.getPosition().x;
    }

    private Double getLandmarkY(PoseLandmark l) {
        return l == null ? null : (double) l.getPosition().y;
    }

    private Double minValidY(Double... vals) {
        Double best = null;
        for (Double v : vals) {
            if (v == null) continue;
            if (best == null || v < best) best = v;
        }
        return best;
    }

    private Double maxValidY(Double... vals) {
        Double best = null;
        for (Double v : vals) {
            if (v == null) continue;
            if (best == null || v > best) best = v;
        }
        return best;
    }

    private Double avgValidX(Double... vals) {
        double sum = 0.0;
        int count = 0;
        for (Double v : vals) {
            if (v == null) continue;
            sum += v;
            count++;
        }
        return count == 0 ? null : sum / count;
    }

    private double computeMedianY(List<Point> points) {
        List<Double> ys = new ArrayList<>();
        for (Point p : points) ys.add(p.y);
        Collections.sort(ys);
        int n = ys.size();
        if (n == 0) return 0;
        if (n % 2 == 1) return ys.get(n / 2);
        return (ys.get(n / 2 - 1) + ys.get(n / 2)) / 2.0;
    }

    private Rect boundingRectCustom(List<Point> pts, int leftPad, int topPad, int rightPad, int bottomPad, int imageW, int imageH) {
        if (pts == null || pts.isEmpty()) return null;

        double minX = Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;

        for (Point p : pts) {
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
        }

        int x1 = Math.max(0, (int) Math.floor(minX) - leftPad);
        int y1 = Math.max(0, (int) Math.floor(minY) - topPad);
        int x2 = Math.min(imageW - 1, (int) Math.ceil(maxX) + rightPad);
        int y2 = Math.min(imageH - 1, (int) Math.ceil(maxY) + bottomPad);

        int w = x2 - x1;
        int h = y2 - y1;

        if (w <= 0 || h <= 0) return null;
        return new Rect(x1, y1, w, h);
    }

    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    private String format1(double value) {
        return String.format(Locale.US, "%.1f", value);
    }

    private void logAllDetectedIds(List<MarkerData> markers) {
        StringBuilder sb = new StringBuilder("ALL IDs: ");
        for (MarkerData m : markers) {
            sb.append(m.id).append(" ");
        }
        Log.d(TAG, sb.toString());
    }

    private void logReferenceIds(List<ReferencePoint> refs) {
        StringBuilder sb = new StringBuilder("REF IDs: ");
        for (ReferencePoint r : refs) {
            sb.append(r.id).append("(").append(r.cm).append("cm) ");
        }
        Log.d(TAG, sb.toString());
    }

    private static class ArucoResult {
        Bitmap bitmapWithIds;
        List<MarkerData> markers;

        ArucoResult(Bitmap bitmapWithIds, List<MarkerData> markers) {
            this.bitmapWithIds = bitmapWithIds;
            this.markers = markers;
        }
    }

    private static class MarkerData {
        int id;
        Point p0, p1, p2, p3;
        double avgSidePx;

        MarkerData(int id, Point p0, Point p1, Point p2, Point p3, double avgSidePx) {
            this.id = id;
            this.p0 = p0;
            this.p1 = p1;
            this.p2 = p2;
            this.p3 = p3;
            this.avgSidePx = avgSidePx;
        }

        Point getCenter() {
            return new Point(
                    (p0.x + p1.x + p2.x + p3.x) / 4.0,
                    (p0.y + p1.y + p2.y + p3.y) / 4.0
            );
        }
    }

    private static class ReferencePoint {
        int id;
        double y;
        double cm;
        Point center;

        ReferencePoint(int id, double y, double cm, Point center) {
            this.id = id;
            this.y = y;
            this.cm = cm;
            this.center = center;
        }
    }

    private static class CalibrationModel {
        double a;
        double b;
        double minY;
        double maxY;
        double minCm;
        double maxCm;

        CalibrationModel(double a, double b, double minY, double maxY, double minCm, double maxCm) {
            this.a = a;
            this.b = b;
            this.minY = minY;
            this.maxY = maxY;
            this.minCm = minCm;
            this.maxCm = maxCm;
        }
    }

    private static class PersonPoints {
        Point head;
        Point foot;

        PersonPoints(Point head, Point foot) {
            this.head = head;
            this.foot = foot;
        }
    }
}