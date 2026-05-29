package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseLandmark;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class DeltaMeasurementActivity extends AppCompatActivity {

    private static final String TAG = "DELTA_MEASURE";

    private ImageView imageViewDeltaResult;
    private Button btnSelectTemplate;
    private Button btnSelectPerson;
    private Button btnProcessHeight;

    private Bitmap templateBitmap;
    private Bitmap personBitmap;

    private PoseDetector poseDetector;

    /*
     * Tinggi asli marker / papan referensi.
     * Sesuai arahan: marker dianggap 200 cm.
     */
    private static final double REAL_REFERENCE_HEIGHT_CM = 200.0;

    /*
     * Ukuran hasil perspective correction.
     * Semua foto akan di-warp ke ukuran ini agar pixel Y lebih stabil.
     */
    private static final int WARP_WIDTH = 900;
    private static final int WARP_HEIGHT = 1800;

    /*
     * Pixel Y titik atas marker/referensi setelah perspective correction.
     *
     * Cara kalibrasi:
     * 1. Jalankan aplikasi.
     * 2. Ambil screenshot hasil perspective.
     * 3. Buka di Paint.
     * 4. Arahkan cursor ke titik atas marker/referensi.
     * 5. Catat nilai Y.
     *
     * Kalau hasil tinggi belum sesuai, nilai ini boleh disesuaikan.
     */
    private static final double REF_TOP_Y_PX = 160.0;

    /*
     * Fallback kalau deteksi karpet merah gagal.
     * Isi dengan nilai Y batas atas karpet merah hasil cek manual di Paint.
     */
    private static final double FALLBACK_FLOOR_Y_PX = 1562.0;

    /*
     * ML Kit biasanya mendeteksi hidung/mata/dahi,
     * bukan benar-benar puncak kepala atau ujung hoodie.
     *
     * Kalau titik merah kepala masih terlalu bawah, naikkan nilai ini.
     * Kalau titik merah terlalu atas, turunkan nilai ini.
     */
    private static final double HEAD_OFFSET_UP_PX = 100.0;

    /*
     * Parameter deteksi karpet merah.
     *
     * Karena jaket/hoodie juga merah, deteksi karpet merah hanya dicari
     * di area bawah gambar.
     */
    private static final double CARPET_ROI_START_RATIO = 0.65;
    private static final double MIN_CARPET_WIDTH_RATIO = 0.25;
    private static final double MIN_CARPET_Y_RATIO = 0.60;
    private static final double MIN_CARPET_AREA = 5000.0;

    /*
     * Delta subtraction tetap ada untuk debug/validasi,
     * tetapi tidak dipakai untuk kaki.
     */
    private static final double PERSON_ROI_LEFT_RATIO = 0.10;
    private static final double PERSON_ROI_RIGHT_RATIO = 0.90;
    private static final int DELTA_THRESHOLD = 18;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_delta_measurement);

        imageViewDeltaResult = findViewById(R.id.imageViewDeltaResult);
        btnSelectTemplate = findViewById(R.id.btnSelectTemplate);
        btnSelectPerson = findViewById(R.id.btnSelectPerson);
        btnProcessHeight = findViewById(R.id.btnProcessHeight);

        imageViewDeltaResult.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageViewDeltaResult.setAdjustViewBounds(true);

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV gagal diinisialisasi", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        PoseDetectorOptions options = new PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
                .build();

        poseDetector = PoseDetection.getClient(options);

        btnSelectTemplate.setOnClickListener(v -> pickTemplateImage.launch("image/*"));
        btnSelectPerson.setOnClickListener(v -> pickPersonImage.launch("image/*"));
        btnProcessHeight.setOnClickListener(v -> processHeight());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (poseDetector != null) {
            poseDetector.close();
        }
    }

    private final ActivityResultLauncher<String> pickTemplateImage =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri == null) return;

                templateBitmap = loadAndRotateBitmap(uri);

                if (templateBitmap == null) {
                    Toast.makeText(
                            this,
                            "Gagal membuka foto template marker",
                            Toast.LENGTH_SHORT
                    ).show();
                    return;
                }

                showBitmap(templateBitmap);
                Toast.makeText(this, "Foto template marker dipilih", Toast.LENGTH_SHORT).show();
            });

    private final ActivityResultLauncher<String> pickPersonImage =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri == null) return;

                personBitmap = loadAndRotateBitmap(uri);

                if (personBitmap == null) {
                    Toast.makeText(
                            this,
                            "Gagal membuka foto marker + orang",
                            Toast.LENGTH_SHORT
                    ).show();
                    return;
                }

                showBitmap(personBitmap);
                Toast.makeText(this, "Foto marker + orang dipilih", Toast.LENGTH_SHORT).show();
            });

    private void processHeight() {
        if (templateBitmap == null) {
            Toast.makeText(this, "Pilih foto template marker dulu", Toast.LENGTH_SHORT).show();
            return;
        }

        if (personBitmap == null) {
            Toast.makeText(this, "Pilih foto marker + orang dulu", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            btnProcessHeight.setEnabled(false);

            /*
             * 1. Perspective correction foto marker kosong.
             */
            Bitmap warpedTemplate = perspectiveCorrectionNoId(templateBitmap);

            if (warpedTemplate == null) {
                btnProcessHeight.setEnabled(true);
                Toast.makeText(
                        this,
                        "Perspective correction template gagal",
                        Toast.LENGTH_LONG
                ).show();
                return;
            }

            /*
             * 2. Perspective correction foto marker + orang.
             */
            Bitmap warpedPerson = perspectiveCorrectionNoId(personBitmap);

            if (warpedPerson == null) {
                btnProcessHeight.setEnabled(true);
                Toast.makeText(
                        this,
                        "Perspective correction foto orang gagal",
                        Toast.LENGTH_LONG
                ).show();
                return;
            }

            /*
             * 3. Delta subtraction tetap dijalankan untuk debug/validasi.
             *    Hasilnya tidak dipakai untuk titik kaki.
             */
            makeDeltaMask(warpedTemplate, warpedPerson);

            /*
             * 4. Deteksi batas atas karpet merah.
             *
             * Prioritas:
             * - Pakai template kosong dulu, karena tidak ada jaket merah/orang.
             * - Kalau template gagal, baru pakai foto orang.
             * - Kalau semua gagal, pakai fallback manual.
             */
            Double floorYFromTemplate = detectRedCarpetTopBoundaryY(warpedTemplate);
            Double floorYFromPerson = detectRedCarpetTopBoundaryY(warpedPerson);

            double finalFloorY;

            if (floorYFromTemplate != null) {
                finalFloorY = floorYFromTemplate;
                Log.d(TAG, "Floor Y from template: " + finalFloorY);

            } else if (floorYFromPerson != null) {
                finalFloorY = floorYFromPerson;
                Log.d(TAG, "Floor Y from person: " + finalFloorY);

            } else {
                finalFloorY = FALLBACK_FLOOR_Y_PX;
                Log.d(TAG, "Floor Y fallback: " + finalFloorY);
            }

            /*
             * 5. Deteksi kepala dan hitung tinggi dari kepala ke floor.
             */
            detectHeadAndMeasureFromFloor(warpedPerson, finalFloorY);

        } catch (Exception e) {
            btnProcessHeight.setEnabled(true);
            Log.e(TAG, "processHeight error", e);
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    /*
     * Perspective correction tanpa membaca ID marker.
     *
     * Cara:
     * - grayscale
     * - threshold area hitam marker
     * - morphology untuk menyatukan marker
     * - kontur terbesar dianggap area papan
     * - warp ke WARP_WIDTH x WARP_HEIGHT
     */
    private Bitmap perspectiveCorrectionNoId(Bitmap inputBitmap) {
        Mat src = new Mat();
        Mat gray = new Mat();
        Mat binary = new Mat();
        Mat morph = new Mat();

        try {
            Utils.bitmapToMat(inputBitmap, src);

            if (src.empty()) {
                Log.e(TAG, "src empty");
                return null;
            }

            if (src.channels() == 4) {
                Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (src.channels() == 3) {
                Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = src.clone();
            }

            Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

            /*
             * Marker hitam dibuat putih pada binary mask.
             */
            Imgproc.threshold(
                    gray,
                    binary,
                    0,
                    255,
                    Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU
            );

            Mat kernelClose = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    new Size(35, 35)
            );

            Imgproc.morphologyEx(binary, morph, Imgproc.MORPH_CLOSE, kernelClose);
            Imgproc.dilate(morph, morph, kernelClose);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();

            Imgproc.findContours(
                    morph,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            if (contours.isEmpty()) {
                Log.e(TAG, "Tidak ada kontur papan marker");
                kernelClose.release();
                hierarchy.release();
                return null;
            }

            MatOfPoint biggestContour = null;
            double biggestArea = 0.0;

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);

                if (area > biggestArea) {
                    biggestArea = area;
                    biggestContour = contour;
                }
            }

            if (biggestContour == null || biggestArea < 20000) {
                Log.e(TAG, "Kontur papan terlalu kecil: " + biggestArea);
                kernelClose.release();
                hierarchy.release();
                return null;
            }

            MatOfPoint2f contour2f = new MatOfPoint2f(biggestContour.toArray());
            RotatedRect rotatedRect = Imgproc.minAreaRect(contour2f);

            Point[] rectPoints = new Point[4];
            rotatedRect.points(rectPoints);

            Point[] ordered = orderPoints(rectPoints);

            MatOfPoint2f srcPoints = new MatOfPoint2f(
                    ordered[0],
                    ordered[1],
                    ordered[2],
                    ordered[3]
            );

            MatOfPoint2f dstPoints = new MatOfPoint2f(
                    new Point(0, 0),
                    new Point(WARP_WIDTH - 1, 0),
                    new Point(WARP_WIDTH - 1, WARP_HEIGHT - 1),
                    new Point(0, WARP_HEIGHT - 1)
            );

            Mat transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Mat warped = new Mat();

            Imgproc.warpPerspective(
                    src,
                    warped,
                    transform,
                    new Size(WARP_WIDTH, WARP_HEIGHT)
            );

            Bitmap result = Bitmap.createBitmap(
                    warped.cols(),
                    warped.rows(),
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(warped, result);

            kernelClose.release();
            hierarchy.release();
            contour2f.release();
            srcPoints.release();
            dstPoints.release();
            transform.release();
            warped.release();

            return result;

        } catch (Exception e) {
            Log.e(TAG, "perspectiveCorrectionNoId error", e);
            return null;
        } finally {
            src.release();
            gray.release();
            binary.release();
            morph.release();
        }
    }

    /*
     * Deteksi batas atas karpet merah.
     *
     * Masalah sebelumnya:
     * Jika baju/jaket merah, sistem membaca warna merah dari jaket
     * sehingga FLOOR_Y salah berada di tengah badan.
     *
     * Perbaikan:
     * - Segmentasi warna merah.
     * - Fokus hanya area bawah gambar.
     * - Cari kontur merah terbesar di bawah.
     * - Ambil rect.y sebagai batas atas karpet.
     */
    private Double detectRedCarpetTopBoundaryY(Bitmap warpedBitmap) {
        Mat src = new Mat();
        Mat hsv = new Mat();

        Mat lowerRedMask = new Mat();
        Mat upperRedMask = new Mat();
        Mat redMask = new Mat();
        Mat bottomRoiMask = new Mat();

        try {
            Utils.bitmapToMat(warpedBitmap, src);

            if (src.empty()) {
                Log.e(TAG, "detectRedCarpetTopBoundaryY: src empty");
                return null;
            }

            if (src.channels() == 4) {
                Imgproc.cvtColor(src, hsv, Imgproc.COLOR_RGBA2RGB);
                Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_RGB2HSV);
            } else {
                Imgproc.cvtColor(src, hsv, Imgproc.COLOR_BGR2HSV);
            }

            /*
             * Range merah HSV.
             * Merah berada di dua area hue:
             * - 0 sampai 12
             * - 155 sampai 180
             */
            Core.inRange(
                    hsv,
                    new Scalar(0, 45, 40),
                    new Scalar(12, 255, 255),
                    lowerRedMask
            );

            Core.inRange(
                    hsv,
                    new Scalar(155, 45, 40),
                    new Scalar(180, 255, 255),
                    upperRedMask
            );

            Core.bitwise_or(lowerRedMask, upperRedMask, redMask);

            /*
             * Fokus hanya area bawah gambar supaya jaket merah tidak ikut terbaca.
             */
            bottomRoiMask = Mat.zeros(redMask.size(), CvType.CV_8UC1);

            int roiStartY = (int) (WARP_HEIGHT * CARPET_ROI_START_RATIO);
            int roiHeight = WARP_HEIGHT - roiStartY;

            Rect bottomRoi = new Rect(
                    0,
                    roiStartY,
                    WARP_WIDTH,
                    roiHeight
            );

            Imgproc.rectangle(
                    bottomRoiMask,
                    bottomRoi,
                    new Scalar(255),
                    -1
            );

            Core.bitwise_and(redMask, bottomRoiMask, redMask);

            Mat kernelOpen = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    new Size(15, 15)
            );

            Mat kernelClose = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    new Size(45, 45)
            );

            Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_OPEN, kernelOpen);
            Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_CLOSE, kernelClose);

            /*
             * Cari kontur merah terbesar di area bawah.
             */
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();

            Imgproc.findContours(
                    redMask,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            if (contours.isEmpty()) {
                Log.e(TAG, "Karpet merah tidak terdeteksi");

                kernelOpen.release();
                kernelClose.release();
                hierarchy.release();

                return null;
            }

            MatOfPoint bestContour = null;
            double bestArea = 0.0;

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                Rect rect = Imgproc.boundingRect(contour);

                boolean isBottomObject = rect.y > WARP_HEIGHT * MIN_CARPET_Y_RATIO;
                boolean isWideEnough = rect.width > WARP_WIDTH * MIN_CARPET_WIDTH_RATIO;
                boolean isAreaEnough = area > MIN_CARPET_AREA;

                if (isBottomObject && isWideEnough && isAreaEnough && area > bestArea) {
                    bestArea = area;
                    bestContour = contour;
                }
            }

            if (bestContour == null) {
                Log.e(TAG, "Kontur karpet merah tidak valid");

                kernelOpen.release();
                kernelClose.release();
                hierarchy.release();

                return null;
            }

            Rect carpetRect = Imgproc.boundingRect(bestContour);

            /*
             * Boundary atas karpet merah.
             */
            double floorY = carpetRect.y;

            Log.d(TAG, "Detected carpet rect: x=" + carpetRect.x
                    + ", y=" + carpetRect.y
                    + ", w=" + carpetRect.width
                    + ", h=" + carpetRect.height
                    + ", area=" + bestArea);

            Log.d(TAG, "Final carpet top boundary Y: " + floorY);

            kernelOpen.release();
            kernelClose.release();
            hierarchy.release();

            return floorY;

        } catch (Exception e) {
            Log.e(TAG, "detectRedCarpetTopBoundaryY error", e);
            return null;
        } finally {
            src.release();
            hsv.release();

            lowerRedMask.release();
            upperRedMask.release();
            redMask.release();
            bottomRoiMask.release();
        }
    }

    /*
     * Delta subtraction:
     * warpedPerson - warpedTemplate
     *
     * Hasil delta hanya untuk debug/validasi,
     * bukan untuk mencari titik kaki.
     */
    private Bitmap makeDeltaMask(Bitmap warpedTemplate, Bitmap warpedPerson) {
        Mat templateMat = new Mat();
        Mat personMat = new Mat();

        Mat templateGray = new Mat();
        Mat personGray = new Mat();

        Mat templateBlur = new Mat();
        Mat personBlur = new Mat();

        Mat diffGray = new Mat();
        Mat mask = new Mat();

        try {
            Utils.bitmapToMat(warpedTemplate, templateMat);
            Utils.bitmapToMat(warpedPerson, personMat);

            Imgproc.resize(templateMat, templateMat, new Size(WARP_WIDTH, WARP_HEIGHT));
            Imgproc.resize(personMat, personMat, new Size(WARP_WIDTH, WARP_HEIGHT));

            if (templateMat.channels() == 4) {
                Imgproc.cvtColor(templateMat, templateGray, Imgproc.COLOR_RGBA2GRAY);
            } else if (templateMat.channels() == 3) {
                Imgproc.cvtColor(templateMat, templateGray, Imgproc.COLOR_BGR2GRAY);
            } else {
                templateGray = templateMat.clone();
            }

            if (personMat.channels() == 4) {
                Imgproc.cvtColor(personMat, personGray, Imgproc.COLOR_RGBA2GRAY);
            } else if (personMat.channels() == 3) {
                Imgproc.cvtColor(personMat, personGray, Imgproc.COLOR_BGR2GRAY);
            } else {
                personGray = personMat.clone();
            }

            Imgproc.GaussianBlur(templateGray, templateBlur, new Size(7, 7), 0);
            Imgproc.GaussianBlur(personGray, personBlur, new Size(7, 7), 0);

            Core.absdiff(personBlur, templateBlur, diffGray);

            Imgproc.threshold(
                    diffGray,
                    mask,
                    DELTA_THRESHOLD,
                    255,
                    Imgproc.THRESH_BINARY
            );

            Mat roiMask = Mat.zeros(mask.size(), CvType.CV_8UC1);

            int roiLeft = (int) (WARP_WIDTH * PERSON_ROI_LEFT_RATIO);
            int roiRight = (int) (WARP_WIDTH * PERSON_ROI_RIGHT_RATIO);
            int roiWidth = roiRight - roiLeft;

            Rect centerRoi = new Rect(
                    roiLeft,
                    0,
                    roiWidth,
                    WARP_HEIGHT
            );

            Imgproc.rectangle(
                    roiMask,
                    centerRoi,
                    new Scalar(255),
                    -1
            );

            Core.bitwise_and(mask, roiMask, mask);

            Mat kernelOpen = Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    new Size(5, 5)
            );

            Mat kernelClose = Imgproc.getStructuringElement(
                    Imgproc.MORPH_RECT,
                    new Size(35, 55)
            );

            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernelOpen);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernelClose);

            Bitmap maskBitmap = matToBitmap(mask);

            roiMask.release();
            kernelOpen.release();
            kernelClose.release();

            return maskBitmap;

        } catch (Exception e) {
            Log.e(TAG, "makeDeltaMask error", e);
            return null;
        } finally {
            templateMat.release();
            personMat.release();

            templateGray.release();
            personGray.release();

            templateBlur.release();
            personBlur.release();

            diffGray.release();
            mask.release();
        }
    }

    /*
     * Deteksi kepala dengan ML Kit,
     * lalu tinggi dihitung dari kepala ke floorY hasil boundary karpet merah.
     */
    private void detectHeadAndMeasureFromFloor(Bitmap warpedPerson, double floorY) {
        try {
            InputImage inputImage = InputImage.fromBitmap(warpedPerson, 0);

            poseDetector.process(inputImage)
                    .addOnSuccessListener(new OnSuccessListener<Pose>() {
                        @Override
                        public void onSuccess(Pose pose) {
                            btnProcessHeight.setEnabled(true);

                            Point headPoint = extractHeadPointFromPose(pose);

                            if (headPoint == null) {
                                Toast.makeText(
                                        DeltaMeasurementActivity.this,
                                        "Titik kepala tidak terdeteksi",
                                        Toast.LENGTH_LONG
                                ).show();
                                return;
                            }

                            double heightCm = headToFloorHeightCm(headPoint, floorY);

                            Bitmap output = drawFinalResultFloorMode(
                                    warpedPerson,
                                    headPoint,
                                    floorY,
                                    heightCm
                            );

                            showBitmap(output);

                            Toast.makeText(
                                    DeltaMeasurementActivity.this,
                                    "Tinggi badan = " + format1(heightCm) + " cm",
                                    Toast.LENGTH_LONG
                            ).show();
                        }
                    })
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            btnProcessHeight.setEnabled(true);

                            Log.e(TAG, "Pose detection gagal", e);

                            Toast.makeText(
                                    DeltaMeasurementActivity.this,
                                    "Pose detection gagal",
                                    Toast.LENGTH_LONG
                            ).show();
                        }
                    });

        } catch (Exception e) {
            btnProcessHeight.setEnabled(true);

            Log.e(TAG, "detectHeadAndMeasureFromFloor error", e);

            Toast.makeText(
                    this,
                    "Pose error: " + e.getMessage(),
                    Toast.LENGTH_LONG
            ).show();
        }
    }

    /*
     * Ambil titik ujung kepala/top point.
     *
     * ML Kit memberi landmark wajah, bukan puncak kepala.
     * Karena itu headY dikurangi HEAD_OFFSET_UP_PX.
     */
    private Point extractHeadPointFromPose(Pose pose) {
        PoseLandmark nose = pose.getPoseLandmark(PoseLandmark.NOSE);
        PoseLandmark leftEye = pose.getPoseLandmark(PoseLandmark.LEFT_EYE);
        PoseLandmark rightEye = pose.getPoseLandmark(PoseLandmark.RIGHT_EYE);
        PoseLandmark leftEar = pose.getPoseLandmark(PoseLandmark.LEFT_EAR);
        PoseLandmark rightEar = pose.getPoseLandmark(PoseLandmark.RIGHT_EAR);

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

        if (headX == null || headY == null) {
            Log.e(TAG, "Head landmark null");
            return null;
        }

        headY = Math.max(0.0, headY - HEAD_OFFSET_UP_PX);

        return new Point(headX, headY);
    }

    /*
     * Rumus tinggi:
     *
     * tinggiCm =
     *      (floorY - headY) * tinggiReferensiCm
     *      / (floorY - refTopY)
     *
     * floorY berasal dari batas atas karpet merah.
     * refTopY berasal dari titik atas marker.
     * tinggi referensi = 200 cm.
     */
    private double headToFloorHeightCm(Point headPoint, double floorY) {
        double bodyPixelHeight = floorY - headPoint.y;
        double referencePixelHeight = floorY - REF_TOP_Y_PX;

        if (bodyPixelHeight < 0) {
            bodyPixelHeight = Math.abs(bodyPixelHeight);
        }

        if (referencePixelHeight <= 0) {
            return 0.0;
        }

        return bodyPixelHeight * REAL_REFERENCE_HEIGHT_CM / referencePixelHeight;
    }

    /*
     * Gambar hasil:
     * - titik kepala merah
     * - titik floor biru
     * - garis floor merah sebagai debug boundary karpet
     */
    private Bitmap drawFinalResultFloorMode(
            Bitmap personPerspectiveBitmap,
            Point headPoint,
            double floorY,
            double heightCm
    ) {
        Mat mat = new Mat();

        try {
            Bitmap mutable = personPerspectiveBitmap.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(mutable, mat);

            Point floorPoint = new Point(headPoint.x, floorY);

            /*
             * Titik kepala merah.
             */
            Imgproc.circle(
                    mat,
                    headPoint,
                    14,
                    new Scalar(255, 0, 0, 255),
                    -1
            );

            /*
             * Titik floor / titik bawah badan biru.
             */
            Imgproc.circle(
                    mat,
                    floorPoint,
                    14,
                    new Scalar(0, 0, 255, 255),
                    -1
            );

            /*
             * Garis floor merah sebagai boundary karpet merah.
             */
            Imgproc.line(
                    mat,
                    new Point(0, floorY),
                    new Point(WARP_WIDTH, floorY),
                    new Scalar(255, 0, 0, 255),
                    4
            );

            putOutlinedText(
                    mat,
                    "HASIL PENGUKURAN",
                    new Point(35, 70),
                    1.0,
                    new Scalar(255, 255, 255, 255),
                    new Scalar(0, 0, 0, 255)
            );

            putOutlinedText(
                    mat,
                    "TINGGI BADAN: " + format1(heightCm) + " cm",
                    new Point(35, 125),
                    0.9,
                    new Scalar(255, 255, 255, 255),
                    new Scalar(0, 0, 0, 255)
            );

            putOutlinedText(
                    mat,
                    "HEAD Y: " + format1(headPoint.y),
                    new Point(35, WARP_HEIGHT - 135),
                    0.6,
                    new Scalar(255, 255, 255, 255),
                    new Scalar(0, 0, 0, 255)
            );

            putOutlinedText(
                    mat,
                    "FLOOR Y: " + format1(floorY),
                    new Point(35, WARP_HEIGHT - 95),
                    0.6,
                    new Scalar(255, 255, 255, 255),
                    new Scalar(0, 0, 0, 255)
            );

            putOutlinedText(
                    mat,
                    "REF TOP Y: " + format1(REF_TOP_Y_PX),
                    new Point(35, WARP_HEIGHT - 55),
                    0.6,
                    new Scalar(255, 255, 255, 255),
                    new Scalar(0, 0, 0, 255)
            );

            Bitmap output = Bitmap.createBitmap(
                    mat.cols(),
                    mat.rows(),
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(mat, output);
            return output;

        } catch (Exception e) {
            Log.e(TAG, "drawFinalResultFloorMode error", e);
            return personPerspectiveBitmap;
        } finally {
            mat.release();
        }
    }

    private Point[] orderPoints(Point[] points) {
        Point topLeft = null;
        Point topRight = null;
        Point bottomRight = null;
        Point bottomLeft = null;

        double minSum = Double.MAX_VALUE;
        double maxSum = -Double.MAX_VALUE;
        double minDiff = Double.MAX_VALUE;
        double maxDiff = -Double.MAX_VALUE;

        for (Point p : points) {
            double sum = p.x + p.y;
            double diff = p.y - p.x;

            if (sum < minSum) {
                minSum = sum;
                topLeft = p;
            }

            if (sum > maxSum) {
                maxSum = sum;
                bottomRight = p;
            }

            if (diff < minDiff) {
                minDiff = diff;
                topRight = p;
            }

            if (diff > maxDiff) {
                maxDiff = diff;
                bottomLeft = p;
            }
        }

        return new Point[]{
                topLeft,
                topRight,
                bottomRight,
                bottomLeft
        };
    }

    private Bitmap loadAndRotateBitmap(Uri imageUri) {
        try {
            Bitmap original = loadBitmapFromUri(imageUri);

            if (original == null) {
                return null;
            }

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
            if (inputStream == null) {
                return bitmap;
            }

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

    private Bitmap matToBitmap(Mat mat) {
        Mat output = new Mat();

        try {
            if (mat == null || mat.empty()) {
                return null;
            }

            if (mat.channels() == 1) {
                Imgproc.cvtColor(mat, output, Imgproc.COLOR_GRAY2RGBA);
            } else {
                output = mat.clone();
            }

            Bitmap bitmap = Bitmap.createBitmap(
                    output.cols(),
                    output.rows(),
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(output, bitmap);
            return bitmap;

        } catch (Exception e) {
            Log.e(TAG, "matToBitmap error", e);
            return null;
        } finally {
            output.release();
        }
    }

    private void putOutlinedText(
            Mat mat,
            String text,
            Point pos,
            double scale,
            Scalar foreground,
            Scalar background
    ) {
        Imgproc.putText(
                mat,
                text,
                pos,
                Imgproc.FONT_HERSHEY_SIMPLEX,
                scale,
                background,
                6
        );

        Imgproc.putText(
                mat,
                text,
                pos,
                Imgproc.FONT_HERSHEY_SIMPLEX,
                scale,
                foreground,
                2
        );
    }

    private void showBitmap(Bitmap bitmap) {
        if (bitmap == null) return;

        imageViewDeltaResult.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageViewDeltaResult.setAdjustViewBounds(true);
        imageViewDeltaResult.setImageBitmap(bitmap);
    }

    private String format1(double value) {
        return String.format(Locale.US, "%.1f", value);
    }

    private Double getLandmarkX(PoseLandmark landmark) {
        if (landmark == null) return null;
        return (double) landmark.getPosition().x;
    }

    private Double getLandmarkY(PoseLandmark landmark) {
        if (landmark == null) return null;
        return (double) landmark.getPosition().y;
    }

    private Double minValidY(Double... values) {
        Double best = null;

        for (Double v : values) {
            if (v == null) continue;

            if (best == null || v < best) {
                best = v;
            }
        }

        return best;
    }

    private Double avgValidX(Double... values) {
        double sum = 0.0;
        int count = 0;

        for (Double v : values) {
            if (v == null) continue;

            sum += v;
            count++;
        }

        if (count == 0) return null;

        return sum / count;
    }
}