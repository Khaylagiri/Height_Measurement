package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

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
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class GalleryOpenCvActivity extends AppCompatActivity {

    private static final String TAG = "HEIGHT_MEASURE";

    // Ganti kalau dictionary marker kamu beda
    private static final int MARKER_DICT = Objdetect.DICT_6X6_1000;

    // Ukuran hasil warp aman untuk Android
    private static final int MAX_WARP_W = 2200;
    private static final int MAX_WARP_H = 3200;

    // Referensi tinggi
    // Sesuaikan dengan marker kamu
    private static final int REF_BOTTOM_ID = 30;
    private static final double REF_BOTTOM_CM = 10.0;

    private static final int REF_MID_ID = 15;
    private static final double REF_MID_CM = 85.0;

    private static final int REF_TOP_ID = 42;
    private static final double REF_TOP_CM = 155.0;

    private ImageView imageViewResult;
    private CropOverlayView cropOverlayView;
    private Button btnPerspective, btnSaveGalleryImage;

    private Bitmap originalBitmap;
    private Bitmap currentBitmap;

    private boolean isMeasureMode = false;
    private Point headPoint = null;
    private Point footPoint = null;

    private List<MarkerData> detectedMarkers = new ArrayList<>();
    private Map<Integer, Double> markerHeightMap = new HashMap<>();

    private MarkerData refBottomMarker = null;
    private MarkerData refMidMarker = null;
    private MarkerData refTopMarker = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery_opencv);

        imageViewResult = findViewById(R.id.imageViewResult);
        cropOverlayView = findViewById(R.id.cropOverlayView);
        btnPerspective = findViewById(R.id.btnPerspective);
        btnSaveGalleryImage = findViewById(R.id.btnSaveGalleryImage);

        markerHeightMap = buildMarkerHeightMap();

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV gagal diinisialisasi", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        String uriString = getIntent().getStringExtra("image_uri");
        if (uriString == null) {
            Toast.makeText(this, "Gambar tidak ditemukan", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        Uri imageUri = Uri.parse(uriString);
        originalBitmap = loadAndRotateBitmap(imageUri);

        if (originalBitmap == null) {
            Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        currentBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        imageViewResult.setImageBitmap(currentBitmap);

        if (cropOverlayView != null) {
            cropOverlayView.setVisibility(View.GONE);
        }

        btnPerspective.setOnClickListener(v -> {
            try {
                btnPerspective.setEnabled(false);

                Bitmap warped = autoPerspectiveFromTwoBoards(originalBitmap);

                if (warped == null) {
                    Toast.makeText(this, "Gagal auto perspective", Toast.LENGTH_SHORT).show();
                    return;
                }

                ArucoResult arucoResult = detectArucoAndDrawIds(warped);

                if (arucoResult == null || arucoResult.markers == null || arucoResult.markers.isEmpty()) {
                    Toast.makeText(this, "Deteksi ArUco gagal setelah perspective", Toast.LENGTH_SHORT).show();
                    return;
                }

                detectedMarkers = arucoResult.markers;

                refBottomMarker = findMarkerById(detectedMarkers, REF_BOTTOM_ID);
                refMidMarker = findMarkerById(detectedMarkers, REF_MID_ID);
                refTopMarker = findMarkerById(detectedMarkers, REF_TOP_ID);

                currentBitmap = addTimestampToBitmap(arucoResult.bitmapWithIds);
                currentBitmap = drawReferenceScaleInfo(currentBitmap);
                imageViewResult.setImageBitmap(currentBitmap);

                isMeasureMode = true;
                headPoint = null;
                footPoint = null;

                Toast.makeText(
                        this,
                        "Perspective otomatis berhasil. Tap 1x kepala lalu 1x kaki.",
                        Toast.LENGTH_LONG
                ).show();

            } catch (OutOfMemoryError oom) {
                Log.e(TAG, "OOM perspective", oom);
                Toast.makeText(this, "Gambar terlalu besar untuk diproses", Toast.LENGTH_LONG).show();
            } catch (Exception e) {
                Log.e(TAG, "Perspective Auto error", e);
                Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            } finally {
                btnPerspective.setEnabled(true);
            }
        });

        btnSaveGalleryImage.setOnClickListener(v -> {
            if (currentBitmap != null) {
                saveBitmapToAppFiles(currentBitmap);
            }
        });

        imageViewResult.setOnTouchListener((v, event) -> {
            if (!isMeasureMode) return false;
            if (event.getAction() != MotionEvent.ACTION_DOWN) return false;

            Point bitmapPoint = viewPointToBitmapPoint(
                    event.getX(),
                    event.getY(),
                    imageViewResult,
                    currentBitmap
            );

            if (bitmapPoint == null) return false;

            if (headPoint == null) {
                headPoint = bitmapPoint;
                Toast.makeText(this, "Titik kepala dipilih", Toast.LENGTH_SHORT).show();
            } else if (footPoint == null) {
                footPoint = bitmapPoint;
                measureHeightAndDraw();
            } else {
                headPoint = bitmapPoint;
                footPoint = null;
                Toast.makeText(this, "Mulai ulang. Titik kepala dipilih", Toast.LENGTH_SHORT).show();
            }

            return true;
        });
    }

    private Map<Integer, Double> buildMarkerHeightMap() {
        Map<Integer, Double> map = new HashMap<>();
        map.put(REF_BOTTOM_ID, REF_BOTTOM_CM);
        map.put(REF_MID_ID, REF_MID_CM);
        map.put(REF_TOP_ID, REF_TOP_CM);
        return map;
    }

    // =========================================================
    // AUTO PERSPECTIVE
    // =========================================================
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
            if (boardCorners == null) {
                Log.e(TAG, "detectTwoBoardCorners gagal");
                return null;
            }

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

            if (ids.empty() || corners.size() < 8) {
                return null;
            }

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

            if (topPoints.size() < 4 || bottomPoints.size() < 4) {
                return null;
            }

            Rect topRect = boundingRectCustom(
                    topPoints, sidePad, topPad, sidePad, sidePad, gray.cols(), gray.rows()
            );

            Rect bottomRect = boundingRectCustom(
                    bottomPoints, sidePad, sidePad, sidePad, bottomPad, gray.cols(), gray.rows()
            );

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

    private double computeMedianY(List<Point> points) {
        List<Double> ys = new ArrayList<>();
        for (Point p : points) ys.add(p.y);
        ys.sort(Double::compareTo);
        int n = ys.size();
        if (n == 0) return 0;
        if (n % 2 == 1) return ys.get(n / 2);
        return (ys.get(n / 2 - 1) + ys.get(n / 2)) / 2.0;
    }

    private Rect boundingRectCustom(
            List<Point> pts,
            int leftPad,
            int topPad,
            int rightPad,
            int bottomPad,
            int imageW,
            int imageH
    ) {
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

    // =========================================================
    // DETEKSI MARKER
    // =========================================================
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
            double totalMarkerSidePx = 0.0;
            int markerCount = 0;
            Set<Integer> uniqueIds = new HashSet<>();

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

                    if (!uniqueIds.contains(id)) {
                        totalMarkerSidePx += avgSide;
                        markerCount++;
                        uniqueIds.add(id);
                    }
                }
            }

            double avgMarkerSidePixels = markerCount > 0 ? totalMarkerSidePx / markerCount : -1.0;

            Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, resultBitmap);

            return new ArucoResult(resultBitmap, markers, avgMarkerSidePixels);

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

    private void drawMarker(Mat mat, MarkerData marker) {
        Imgproc.line(mat, marker.p0, marker.p1, new Scalar(0, 255, 0, 255), 3);
        Imgproc.line(mat, marker.p1, marker.p2, new Scalar(0, 255, 0, 255), 3);
        Imgproc.line(mat, marker.p2, marker.p3, new Scalar(0, 255, 0, 255), 3);
        Imgproc.line(mat, marker.p3, marker.p0, new Scalar(0, 255, 0, 255), 3);

        Point center = marker.getCenter();
        Imgproc.circle(mat, center, 6, new Scalar(0, 0, 255, 255), -1);

        String label;
        if (markerHeightMap.containsKey(marker.id)) {
            label = "ID:" + marker.id + " (H:" + format1(markerHeightMap.get(marker.id)) + "cm)";
        } else {
            label = "ID:" + marker.id;
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

    private MarkerData findMarkerById(List<MarkerData> markers, int id) {
        for (MarkerData m : markers) {
            if (m.id == id) return m;
        }
        return null;
    }

    // =========================================================
    // HEIGHT
    // =========================================================
    private void measureHeightAndDraw() {
        if (headPoint == null || footPoint == null) return;

        if (refBottomMarker == null || refTopMarker == null) {
            Toast.makeText(this, "2 marker referensi belum tersedia", Toast.LENGTH_SHORT).show();
            return;
        }

        double headCm = yPixelToCmUsingTwoMarkers(headPoint.y);
        double footCm = yPixelToCmUsingTwoMarkers(footPoint.y);
        double heightCm = Math.abs(footCm - headCm);

        Bitmap drawn = drawMeasurementOnBitmap(currentBitmap, headPoint, footPoint, headCm, footCm, heightCm);
        currentBitmap = drawn;
        imageViewResult.setImageBitmap(currentBitmap);

        Toast.makeText(this, "Tinggi = " + format1(heightCm) + " cm", Toast.LENGTH_LONG).show();
    }

    private double yPixelToCmUsingTwoMarkers(double yPixel) {
        double yTop = refTopMarker.getCenter().y;
        double yBottom = refBottomMarker.getCenter().y;

        double cmTop = markerHeightMap.get(refTopMarker.id);
        double cmBottom = markerHeightMap.get(refBottomMarker.id);

        double dy = yBottom - yTop;
        if (Math.abs(dy) < 1e-6) return cmBottom;

        return cmTop + ((yPixel - yTop) / (yBottom - yTop)) * (cmBottom - cmTop);
    }

    private Bitmap drawMeasurementOnBitmap(Bitmap bitmap, Point head, Point foot,
                                           double headCm, double footCm, double heightCm) {
        try {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Mat mat = new Mat();
            Utils.bitmapToMat(mutableBitmap, mat);

            Point verticalFoot = new Point(head.x, foot.y);

            Imgproc.line(mat, head, verticalFoot, new Scalar(0, 255, 255, 255), 5);
            Imgproc.circle(mat, head, 11, new Scalar(0, 0, 255, 255), -1);
            Imgproc.circle(mat, foot, 11, new Scalar(255, 0, 0, 255), -1);

            double scaleX = Math.min(mat.cols() - 120, Math.max(head.x + 250, mat.cols() * 0.80));
            Point rulerTop = new Point(scaleX, refTopMarker.getCenter().y);
            Point rulerBottom = new Point(scaleX, refBottomMarker.getCenter().y);

            Imgproc.line(mat, rulerTop, rulerBottom, new Scalar(0, 220, 255, 255), 12);

            for (int i = 0; i <= 40; i++) {
                double t = i / 40.0;
                double y = rulerTop.y + t * (rulerBottom.y - rulerTop.y);
                double x1 = scaleX - 18;
                double x2 = (i % 5 == 0) ? scaleX + 18 : scaleX + 8;
                Imgproc.line(mat, new Point(x1, y), new Point(x2, y), new Scalar(30, 30, 30, 255), 2);
            }

            Imgproc.line(mat, head, new Point(scaleX, head.y), new Scalar(255, 255, 255, 255), 2);

            putOutlinedTextMat(
                    mat,
                    format1(heightCm) + " cm",
                    new Point(scaleX + 35, head.y + 5),
                    1.0,
                    new Scalar(255, 255, 255, 255),
                    new Scalar(20, 40, 70, 255)
            );

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

    private Bitmap drawReferenceScaleInfo(Bitmap bitmap) {
        try {
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Mat mat = new Mat();
            Utils.bitmapToMat(mutableBitmap, mat);

            if (refTopMarker != null) {
                putOutlinedTextMat(
                        mat,
                        "ID: " + REF_TOP_ID + " (H: " + format1(REF_TOP_CM) + "cm)",
                        new Point(refTopMarker.getCenter().x + 30, refTopMarker.getCenter().y),
                        0.7,
                        new Scalar(255, 255, 255, 255),
                        new Scalar(20, 40, 70, 255)
                );
            }

            if (refMidMarker != null) {
                putOutlinedTextMat(
                        mat,
                        "ID: " + REF_MID_ID + " (H: " + format1(REF_MID_CM) + "cm)",
                        new Point(refMidMarker.getCenter().x + 30, refMidMarker.getCenter().y),
                        0.7,
                        new Scalar(255, 255, 255, 255),
                        new Scalar(20, 40, 70, 255)
                );
            }

            if (refBottomMarker != null) {
                putOutlinedTextMat(
                        mat,
                        "ID: " + REF_BOTTOM_ID + " (H: " + format1(REF_BOTTOM_CM) + "cm)",
                        new Point(refBottomMarker.getCenter().x + 30, refBottomMarker.getCenter().y),
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

    private void putHeaderText(Mat mat, String text) {
        Point pos = new Point(mat.cols() * 0.18, 65);
        Imgproc.rectangle(
                mat,
                new Point(pos.x - 25, pos.y - 40),
                new Point(pos.x + 520, pos.y + 18),
                new Scalar(255, 255, 255, 220),
                -1
        );
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
                new Scalar(20, 20, 20, 255), 3);
    }

    private void putHeaderText2(Mat mat, String text) {
        Point pos = new Point(mat.cols() * 0.18, 112);
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
                new Scalar(20, 20, 20, 255), 3);
    }

    private void putOutlinedTextMat(Mat mat, String text, Point pos, double scale, Scalar fg, Scalar bg) {
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, scale, bg, 6);
        Imgproc.putText(mat, text, pos, Imgproc.FONT_HERSHEY_SIMPLEX, scale, fg, 2);
    }

    private String format1(double value) {
        return String.format(Locale.US, "%.1f", value);
    }

    private Point viewPointToBitmapPoint(float touchX, float touchY, ImageView imageView, Bitmap bitmap) {
        if (bitmap == null) return null;

        int imageW = bitmap.getWidth();
        int imageH = bitmap.getHeight();
        int viewW = imageView.getWidth();
        int viewH = imageView.getHeight();

        float scale = Math.min((float) viewW / imageW, (float) viewH / imageH);
        float displayedW = imageW * scale;
        float displayedH = imageH * scale;

        float offsetX = (viewW - displayedW) / 2f;
        float offsetY = (viewH - displayedH) / 2f;

        if (touchX < offsetX || touchX > offsetX + displayedW ||
                touchY < offsetY || touchY > offsetY + displayedH) {
            return null;
        }

        double bx = (touchX - offsetX) / scale;
        double by = (touchY - offsetY) / scale;

        bx = clamp(bx, 0, imageW - 1);
        by = clamp(by, 0, imageH - 1);

        return new Point(bx, by);
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

    private Bitmap loadAndRotateBitmap(Uri imageUri) {
        try {
            Bitmap original = loadBitmapFromUri(imageUri);
            if (original == null) return null;
            return rotateBitmapIfRequired(original, imageUri);
        } catch (Exception e) {
            Log.e(TAG, "loadAndRotateBitmap error", e);
            return null;
        }
    }

    private Bitmap loadBitmapFromUri(Uri uri) {
        try (InputStream inputStream = getContentResolver().openInputStream(uri)) {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeStream(inputStream, null, options);
        } catch (Exception e) {
            Log.e(TAG, "loadBitmapFromUri error", e);
            return null;
        }
    }

    private Bitmap rotateBitmapIfRequired(Bitmap bitmap, Uri imageUri) {
        try (InputStream inputStream = getContentResolver().openInputStream(imageUri)) {
            if (inputStream == null) return bitmap;

            ExifInterface exif = new ExifInterface(inputStream);
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            );

            Matrix matrix = new Matrix();

            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    matrix.postRotate(90);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    matrix.postRotate(180);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    matrix.postRotate(270);
                    break;
                default:
                    return bitmap;
            }

            return Bitmap.createBitmap(
                    bitmap,
                    0,
                    0,
                    bitmap.getWidth(),
                    bitmap.getHeight(),
                    matrix,
                    true
            );

        } catch (Exception e) {
            Log.e(TAG, "rotateBitmapIfRequired error", e);
            return bitmap;
        }
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

            File imageFile = new File(picturesDir, "GAL_" + fileName + ".jpg");

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos);
            fos.flush();
            fos.close();

            Toast.makeText(this, "Gambar berhasil disimpan", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToAppFiles error", e);
            Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show();
        }
    }

    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    private double clamp(double val, double min, double max) {
        return Math.max(min, Math.min(max, val));
    }

    private static class ArucoResult {
        Bitmap bitmapWithIds;
        List<MarkerData> markers;
        double avgMarkerSidePixels;

        ArucoResult(Bitmap bitmapWithIds, List<MarkerData> markers, double avgMarkerSidePixels) {
            this.bitmapWithIds = bitmapWithIds;
            this.markers = markers;
            this.avgMarkerSidePixels = avgMarkerSidePixels;
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
}