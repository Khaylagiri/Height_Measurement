package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MeasurementActivity extends AppCompatActivity {

    private static final String TAG = "MeasurementActivity";
    private static final String MODEL_ASSET_PATH = "pose_landmarker_lite.task";

    /*
     * PENTING:
     * Karena kamu bilang panjang marker/board vertikal = 200 cm,
     * maka skala tinggi dihitung dari tinggi gambar hasil perspective correction.
     *
     * Rumus:
     * cmPerPixel = BOARD_REAL_HEIGHT_CM / tinggiBoardDalamPixel
     */
    private static final double BOARD_REAL_HEIGHT_CM = 200.0;
    private static final int BOARD_TOP_MARGIN_PX = 0;
    private static final int BOARD_BOTTOM_MARGIN_PX = 0;

    /*
     * KOREKSI KEDALAMAN (DEPTH/PARALLAX CORRECTION)
     *
     * Perspective correction hanya meluruskan bidang board (2D).
     * Jika subjek berdiri di DEPAN board (lebih dekat ke kamera),
     * subjek akan tampak lebih besar dari skala board → tinggi terbaca lebih tinggi.
     *
     * Rumus fisika:
     * tinggi_sebenarnya = tinggi_terukur × (jarak_kamera - jarak_subjek) / jarak_kamera
     *
     * Contoh kasus dari kode lampiran:
     * - Kamera 350 cm dari board, subjek 50 cm di depan board
     * - Faktor = (350 - 50) / 350 = 0.857
     * - Tinggi terukur 186.1 cm × 0.857 = 159.5 cm.
     *
     * CARA KALIBRASI:
     * 1. Ukur jarak kamera ke board (dalam cm) → isi CAMERA_DISTANCE_CM
     * 2. Ukur jarak subjek ke board (dalam cm) → isi SUBJECT_DISTANCE_FROM_BOARD_CM
     * 3. Atau hitung langsung: DEPTH_CORRECTION_FACTOR = tinggi_asli / tinggi_aplikasi
     *    Contoh: 160.0 / 186.1 = 0.86
     */
    private static final double CAMERA_DISTANCE_CM = 350.0;
    private static final double SUBJECT_DISTANCE_FROM_BOARD_CM = 50.0;

    /*
     * Kalibrasi akhir dari kode lampiran.
     * Faktor harus divalidasi ulang jika jarak atau board berubah.
     */
    private static final double HEIGHT_CALIBRATION_FACTOR = 1.025;

    private ImageView imageViewMeasurement;
    private TextView tvMeasurementResult;
    private Button btnProcessMeasurement;
    private Button btnSaveMeasurement;

    private Bitmap originalBitmap;
    private Bitmap landmarkPreviewBitmap;
    private Bitmap currentBitmap;

    private PoseLandmarker poseLandmarker;
    private double cmPerPixel = -1.0;
    private double boardTopY = -1.0;
    private double boardBottomY = -1.0;
    private double boardPixelHeight = -1.0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_measurement);

        imageViewMeasurement = findViewById(R.id.imageViewMeasurement);
        tvMeasurementResult = findViewById(R.id.tvMeasurementResult);
        btnProcessMeasurement = findViewById(R.id.btnProcessMeasurement);
        btnSaveMeasurement = findViewById(R.id.btnSaveMeasurement);

        imageViewMeasurement.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageViewMeasurement.setAdjustViewBounds(true);

        btnProcessMeasurement.setText("Hitung Tinggi");
        btnSaveMeasurement.setText("Simpan");

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV gagal diinisialisasi", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        initPoseLandmarker();

        cmPerPixel = getIntent().getDoubleExtra("cm_per_pixel", -1.0);
        boardTopY = getIntent().getDoubleExtra("board_top_y", -1.0);
        boardBottomY = getIntent().getDoubleExtra("board_bottom_y", -1.0);
        boardPixelHeight = getIntent().getDoubleExtra("board_pixel_height", -1.0);

        if (cmPerPixel <= 0.0) {
            // Fallback dari kode lampiran: board 200 cm, 19,35 sel, 120 px/sel.
            boardTopY = 0.35 * 120.0;
            boardBottomY = (0.35 + 11.35 + 8.0) * 120.0;
            boardPixelHeight = boardBottomY - boardTopY;
            cmPerPixel = 200.0 / boardPixelHeight;
        }

        String uriString = getIntent().getStringExtra("image_uri");
        String landmarkUriString = getIntent().getStringExtra("landmark_image_uri");

        if (uriString == null || uriString.trim().isEmpty()) {
            Toast.makeText(this, "Gambar tidak ditemukan", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        /*
         * originalBitmap selalu gambar bersih. Bitmap ini dipakai ulang oleh
         * MediaPipe dan contour saat tombol Hitung Tinggi ditekan.
         */
        originalBitmap = loadAndRotateBitmap(Uri.parse(uriString));

        if (originalBitmap == null) {
            Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        /*
         * landmarkPreviewBitmap hanya untuk tampilan awal agar titik landmark
         * dari halaman MediaPipe tidak hilang setelah tombol Next ditekan.
         */
        if (landmarkUriString != null && !landmarkUriString.trim().isEmpty()) {
            landmarkPreviewBitmap = loadAndRotateBitmap(Uri.parse(landmarkUriString));
        }

        if (landmarkPreviewBitmap != null) {
            currentBitmap = landmarkPreviewBitmap;
            tvMeasurementResult.setText(
                    "Landmark MediaPipe tetap ditampilkan. Tekan Hitung Tinggi untuk memproses."
            );
        } else {
            currentBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
            tvMeasurementResult.setText("Siap menghitung tinggi badan");
        }

        imageViewMeasurement.setImageBitmap(currentBitmap);

        btnProcessMeasurement.setOnClickListener(v -> runMeasurementPipeline());

        btnSaveMeasurement.setOnClickListener(v -> {
            if (currentBitmap == null) {
                Toast.makeText(this, "Belum ada hasil untuk disimpan", Toast.LENGTH_SHORT).show();
                return;
            }

            saveBitmapToAppFiles(currentBitmap);
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (poseLandmarker != null) {
            poseLandmarker.close();
            poseLandmarker = null;
        }
    }

    private void initPoseLandmarker() {
        try {
            BaseOptions baseOptions = BaseOptions.builder()
                    .setModelAssetPath(MODEL_ASSET_PATH)
                    .build();

            PoseLandmarker.PoseLandmarkerOptions options =
                    PoseLandmarker.PoseLandmarkerOptions.builder()
                            .setBaseOptions(baseOptions)
                            .setRunningMode(RunningMode.IMAGE)
                            .setNumPoses(1)
                            .setMinPoseDetectionConfidence(0.5f)
                            .setMinPosePresenceConfidence(0.5f)
                            .setMinTrackingConfidence(0.5f)
                            .build();

            poseLandmarker = PoseLandmarker.createFromOptions(this, options);

        } catch (Exception e) {
            Log.e(TAG, "initPoseLandmarker error", e);
            Toast.makeText(
                    this,
                    "MediaPipe gagal init. Cek file pose_landmarker_lite.task di assets.",
                    Toast.LENGTH_LONG
            ).show();
        }
    }

    private void runMeasurementPipeline() {
        ContourResult contourResult = null;

        try {
            if (originalBitmap == null) {
                Toast.makeText(this, "Gambar belum tersedia", Toast.LENGTH_SHORT).show();
                return;
            }

            if (poseLandmarker == null) {
                Toast.makeText(this, "MediaPipe belum siap", Toast.LENGTH_LONG).show();
                return;
            }

            Bitmap workingBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);

            MPImage mpImage = new BitmapImageBuilder(workingBitmap).build();
            PoseLandmarkerResult poseResult = poseLandmarker.detect(mpImage);

            if (poseResult == null || poseResult.landmarks().isEmpty()) {
                currentBitmap = drawError(workingBitmap, "Tubuh tidak terdeteksi");
                imageViewMeasurement.setImageBitmap(currentBitmap);
                tvMeasurementResult.setText("Tubuh tidak terdeteksi oleh MediaPipe");
                return;
            }

            List<NormalizedLandmark> landmarks = poseResult.landmarks().get(0);

            if (landmarks.size() < 33) {
                currentBitmap = drawError(workingBitmap, "Landmark tubuh tidak lengkap");
                imageViewMeasurement.setImageBitmap(currentBitmap);
                tvMeasurementResult.setText("Landmark tubuh tidak lengkap");
                return;
            }

            contourResult = runContourProcess(workingBitmap, landmarks);

            BodyMeasurement measurement = estimateBodyMeasurement(
                    landmarks,
                    workingBitmap.getWidth(),
                    workingBitmap.getHeight(),
                    contourResult
            );

            PoseValidation validation = validatePose(
                    landmarks,
                    measurement,
                    workingBitmap.getWidth(),
                    workingBitmap.getHeight()
            );

            if (!validation.valid) {
                currentBitmap = drawError(workingBitmap, "POSE TIDAK VALID");
                imageViewMeasurement.setImageBitmap(currentBitmap);
                tvMeasurementResult.setText(validation.message);
                Toast.makeText(this, "Pose tidak valid untuk pengukuran", Toast.LENGTH_LONG).show();
                return;
            }

            int imageHeightPx = workingBitmap.getHeight();

            double verticalCm = pixelToCm(measurement.verticalHeightPx);
            double skeletonCm = pixelToCm(measurement.totalSkeletonPx);
            currentBitmap = drawFinalMeasurement(
                    workingBitmap,
                    landmarks,
                    contourResult,
                    measurement,
                    validation,
                    verticalCm,
                    skeletonCm,
                    imageHeightPx
            );

            imageViewMeasurement.setImageBitmap(currentBitmap);

            double depthFactor = (CAMERA_DISTANCE_CM - SUBJECT_DISTANCE_FROM_BOARD_CM) / CAMERA_DISTANCE_CM;
            String debugText = String.format(Locale.US,
                    "Tinggi badan: %.1f cm\n" +
                            "Status: %s\n\n" +
                            "--- Detail Debug ---\n" +
                            "verticalHeightPx: %.1f px\n" +
                            "headTop.y: %.1f px\n" +
                            "footBottom.y: %.1f px\n" +
                            "boardTopY: %.1f px\n" +
                            "boardBottomY: %.1f px\n" +
                            "boardPixelHeight: %.1f px\n" +
                            "cmPerPixel: %.6f cm/px\n" +
                            "Depth Factor: %.3f\n" +
                            "Calibration Factor: %.3f\n" +
                            "Camera/Subj: %.0f/%.0f cm\n" +
                            "heightCm: %.1f cm",
                    verticalCm,
                    validation.valid ? "Pose Valid" : "Pose Tidak Valid",
                    measurement.verticalHeightPx,
                    measurement.headTop.y,
                    measurement.footBottom.y,
                    boardTopY,
                    boardBottomY,
                    boardPixelHeight,
                    cmPerPixel,
                    depthFactor,
                    HEIGHT_CALIBRATION_FACTOR,
                    CAMERA_DISTANCE_CM,
                    SUBJECT_DISTANCE_FROM_BOARD_CM,
                    verticalCm
            );

            tvMeasurementResult.setText(debugText);

            Log.i(TAG, "MEASUREMENT_DEBUG: " +
                    "verticalHeightPx=" + measurement.verticalHeightPx +
                    ", headTop.y=" + measurement.headTop.y +
                    ", footBottom.y=" + measurement.footBottom.y +
                    ", boardTopY=" + boardTopY +
                    ", boardBottomY=" + boardBottomY +
                    ", boardPixelHeight=" + boardPixelHeight +
                    ", cmPerPixel=" + cmPerPixel +
                    ", heightCm=" + verticalCm +
                    ", poseValidationStatus=" + (validation.valid ? "Valid" : "Invalid")
            );

            Toast.makeText(this, "Tinggi badan berhasil dihitung", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "runMeasurementPipeline error", e);
            Toast.makeText(this, "Error measurement: " + e.getMessage(), Toast.LENGTH_LONG).show();

            if (originalBitmap != null) {
                currentBitmap = drawError(originalBitmap, "Error: " + e.getMessage());
                imageViewMeasurement.setImageBitmap(currentBitmap);
            }

            tvMeasurementResult.setText("Gagal menghitung tinggi badan");

        } finally {
            if (contourResult != null && contourResult.contour != null) {
                contourResult.contour.release();
            }
        }
    }

    private int getBoardPixelHeight(int imageHeightPx) {
        int boardPixelHeight = imageHeightPx - BOARD_TOP_MARGIN_PX - BOARD_BOTTOM_MARGIN_PX;
        return Math.max(1, boardPixelHeight);
    }

    private double pixelToCm(double heightPx) {
        /*
         * Menggunakan cm_per_pixel yang dikirim via Intent dari Perspective Correction.
         * Jika tidak ada, gunakan fallback grid 19,35 sel dari kode lampiran.
         */
        double currentScale = this.cmPerPixel;
        if (currentScale <= 0.0) {
            currentScale = 200.0 / 2322.0;
        }

        // Koreksi kedalaman: subjek berada di depan papan.
        double depthCorrectionFactor =
                (CAMERA_DISTANCE_CM - SUBJECT_DISTANCE_FROM_BOARD_CM) / CAMERA_DISTANCE_CM;

        // Kalibrasi akhir bersifat perkalian, bukan penambahan cm tetap.
        return heightPx
                * currentScale
                * depthCorrectionFactor
                * HEIGHT_CALIBRATION_FACTOR;
    }

    private BodyMeasurement estimateBodyMeasurement(
            List<NormalizedLandmark> landmarks,
            int imageWidth,
            int imageHeight,
            ContourResult contourResult
    ) {
        Point nose = landmarkToPoint(landmarks.get(0), imageWidth, imageHeight);

        Point leftEyeInner = landmarkToPoint(landmarks.get(1), imageWidth, imageHeight);
        Point rightEyeInner = landmarkToPoint(landmarks.get(4), imageWidth, imageHeight);

        Point leftEye = landmarkToPoint(landmarks.get(2), imageWidth, imageHeight);
        Point rightEye = landmarkToPoint(landmarks.get(5), imageWidth, imageHeight);

        Point leftEar = landmarkToPoint(landmarks.get(7), imageWidth, imageHeight);
        Point rightEar = landmarkToPoint(landmarks.get(8), imageWidth, imageHeight);

        Point leftShoulder = landmarkToPoint(landmarks.get(11), imageWidth, imageHeight);
        Point rightShoulder = landmarkToPoint(landmarks.get(12), imageWidth, imageHeight);

        Point leftHip = landmarkToPoint(landmarks.get(23), imageWidth, imageHeight);
        Point rightHip = landmarkToPoint(landmarks.get(24), imageWidth, imageHeight);

        Point leftKnee = landmarkToPoint(landmarks.get(25), imageWidth, imageHeight);
        Point rightKnee = landmarkToPoint(landmarks.get(26), imageWidth, imageHeight);

        Point leftAnkle = landmarkToPoint(landmarks.get(27), imageWidth, imageHeight);
        Point rightAnkle = landmarkToPoint(landmarks.get(28), imageWidth, imageHeight);

        Point leftHeel = landmarkToPoint(landmarks.get(29), imageWidth, imageHeight);
        Point rightHeel = landmarkToPoint(landmarks.get(30), imageWidth, imageHeight);

        Point leftFootIndex = landmarkToPoint(landmarks.get(31), imageWidth, imageHeight);
        Point rightFootIndex = landmarkToPoint(landmarks.get(32), imageWidth, imageHeight);

        Point eyeMid1 = midpoint(leftEye, rightEye);
        Point eyeMid2 = midpoint(leftEyeInner, rightEyeInner);
        Point eyeMid = midpoint(eyeMid1, eyeMid2);

        Point earMid = midpoint(leftEar, rightEar);
        Point shoulderMid = midpoint(leftShoulder, rightShoulder);
        Point hipMid = midpoint(leftHip, rightHip);

        /*
         * Estimasi puncak kepala anatomi.
         * Tidak mengambil batas rambut/hijab/topi/aksesoris.
         * Diambil murni dari landmark wajah dan bahu untuk menghindari bias.
         */
        Point headTop = estimateAnatomicalHeadTop(
                nose,
                eyeMid,
                earMid,
                shoulderMid
        );

        Point footBottom = getFootBottomPoint(
                leftAnkle,
                rightAnkle,
                leftHeel,
                rightHeel,
                leftFootIndex,
                rightFootIndex
        );

        /*
         * Hasil utama:
         * tinggi vertikal seperti stadiometer.
         */
        double verticalHeightPx = Math.abs(footBottom.y - headTop.y);

        /*
         * Skeleton hanya validasi/pembanding.
         */
        double headNeckPx = distance(headTop, shoulderMid);
        double torsoPx = distance(shoulderMid, hipMid);

        double leftUpperLegPx = distance(leftHip, leftKnee);
        double leftLowerLegPx = distance(leftKnee, leftAnkle);
        double leftFootPx = estimateFootExtension(leftAnkle, leftHeel, leftFootIndex);

        double rightUpperLegPx = distance(rightHip, rightKnee);
        double rightLowerLegPx = distance(rightKnee, rightAnkle);
        double rightFootPx = estimateFootExtension(rightAnkle, rightHeel, rightFootIndex);

        double leftLegPx = leftUpperLegPx + leftLowerLegPx + leftFootPx;
        double rightLegPx = rightUpperLegPx + rightLowerLegPx + rightFootPx;
        double legPx = robustAverageLeg(leftLegPx, rightLegPx);
        double totalSkeletonPx = headNeckPx + torsoPx + legPx;

        BodyMeasurement measurement = new BodyMeasurement();

        measurement.headTop = headTop;
        measurement.eyeMid = eyeMid;
        measurement.shoulderMid = shoulderMid;
        measurement.hipMid = hipMid;
        measurement.footBottom = footBottom;

        measurement.leftShoulder = leftShoulder;
        measurement.rightShoulder = rightShoulder;

        measurement.leftHip = leftHip;
        measurement.leftKnee = leftKnee;
        measurement.leftAnkle = leftAnkle;
        measurement.leftHeel = leftHeel;
        measurement.leftFootIndex = leftFootIndex;

        measurement.rightHip = rightHip;
        measurement.rightKnee = rightKnee;
        measurement.rightAnkle = rightAnkle;
        measurement.rightHeel = rightHeel;
        measurement.rightFootIndex = rightFootIndex;

        measurement.headNeckPx = headNeckPx;
        measurement.torsoPx = torsoPx;
        measurement.leftLegPx = leftLegPx;
        measurement.rightLegPx = rightLegPx;
        measurement.legPx = legPx;
        measurement.verticalHeightPx = verticalHeightPx;
        measurement.totalSkeletonPx = totalSkeletonPx;

        return measurement;
    }

    private Point estimateAnatomicalHeadTop(
            Point nose,
            Point eyeMid,
            Point earMid,
            Point shoulderMid
    ) {
        Point faceCenter = midpoint(nose, eyeMid);

        double dirX = faceCenter.x - shoulderMid.x;
        double dirY = faceCenter.y - shoulderMid.y;
        double len = Math.sqrt(dirX * dirX + dirY * dirY);

        if (len <= 0.0001) {
            double fallback = distance(nose, shoulderMid) * 0.35;
            return new Point(nose.x, nose.y - fallback);
        }

        dirX /= len;
        dirY /= len;

        /*
         * Koreksi puncak kepala anatomi.
         * Kalau hasil tinggi terlalu pendek secara konsisten, angka ini boleh dinaikkan sedikit.
         * Kalau terlalu tinggi, angka ini boleh diturunkan sedikit.
         */
        double eyeToShoulder = distance(eyeMid, shoulderMid);
        double earToEye = distance(earMid, eyeMid);
        // Kombinasi terpilih sementara dari MAE pengujian keempat: 0,35 dan 0,10.
        double correction = eyeToShoulder * 0.35 + earToEye * 0.10;

        return new Point(
                faceCenter.x + dirX * correction,
                faceCenter.y + dirY * correction
        );
    }

    private Point getFootBottomPoint(Point... points) {
        Point bottom = points[0];

        for (Point p : points) {
            if (p.y > bottom.y) {
                bottom = p;
            }
        }

        return bottom;
    }

    private double estimateFootExtension(Point ankle, Point heel, Point footIndex) {
        double ankleToHeel = distance(ankle, heel);
        double ankleToFoot = distance(ankle, footIndex);
        return Math.max(ankleToHeel, ankleToFoot);
    }

    private double robustAverageLeg(double leftLegPx, double rightLegPx) {
        double min = Math.min(leftLegPx, rightLegPx);
        double max = Math.max(leftLegPx, rightLegPx);

        if (min <= 0.0001) {
            return max;
        }

        double ratio = max / min;

        if (ratio <= 1.25) {
            return (leftLegPx + rightLegPx) / 2.0;
        } else {
            return max;
        }
    }

    private double calculateAngle(Point p1, Point center, Point p2) {
        double v1x = p1.x - center.x;
        double v1y = p1.y - center.y;
        double v2x = p2.x - center.x;
        double v2y = p2.y - center.y;

        double dotProduct = v1x * v2x + v1y * v2y;
        double magnitude1 = Math.sqrt(v1x * v1x + v1y * v1y);
        double magnitude2 = Math.sqrt(v2x * v2x + v2y * v2y);

        if (magnitude1 * magnitude2 == 0) return 0.0;

        double cosAngle = dotProduct / (magnitude1 * magnitude2);
        cosAngle = Math.max(-1.0, Math.min(1.0, cosAngle));

        return Math.toDegrees(Math.acos(cosAngle));
    }

    private PoseValidation validatePose(
            List<NormalizedLandmark> landmarks,
            BodyMeasurement measurement,
            int imageWidth,
            int imageHeight
    ) {
        PoseValidation validation = new PoseValidation();
        validation.valid = true;
        validation.message = "OK";

        List<String> warnings = new ArrayList<>();

        // --- HARD FAIL: landmark pokok null ---
        if (measurement.headTop == null || measurement.footBottom == null ||
                measurement.leftShoulder == null || measurement.rightShoulder == null ||
                measurement.leftHip == null || measurement.rightHip == null ||
                measurement.leftKnee == null || measurement.rightKnee == null ||
                measurement.leftAnkle == null || measurement.rightAnkle == null ||
                measurement.leftHeel == null || measurement.rightHeel == null ||
                measurement.leftFootIndex == null || measurement.rightFootIndex == null) {
            validation.valid = false;
            validation.message = "Landmark tubuh tidak lengkap. Pastikan seluruh tubuh terlihat.";
            Log.w(TAG, "validatePose HARD FAIL: satu atau lebih landmark null");
            return validation;
        }

        // 1. Cek visibilitas landmark utama (threshold SANGAT rendah: >= 0.1)
        //    Landmark dari perspective-corrected image biasanya punya visibility rendah.
        int[] keyLandmarks = {0, 11, 12, 23, 24, 27, 28};  // hanya landmark paling penting
        int lowVisCount = 0;
        for (int idx : keyLandmarks) {
            if (idx >= landmarks.size()) continue;
            NormalizedLandmark lm = landmarks.get(idx);
            float visibility = lm.visibility().isPresent() ? lm.visibility().get() : 0.0f;
            Log.d(TAG, "validatePose: landmark[" + idx + "] visibility=" + visibility);
            if (visibility < 0.1f) {
                lowVisCount++;
            }
        }
        if (lowVisCount > 3) {
            // Lebih dari setengah landmark utama tidak terlihat sama sekali
            warnings.add("Visibilitas rendah pada " + lowVisCount + "/" + keyLandmarks.length + " landmark utama");
            Log.w(TAG, "validatePose WARNING: " + lowVisCount + " landmark utama visibility < 0.1");
        }

        // 2. Cek apakah headTop dan footBottom masuk frame (padding sangat kecil: 2px)
        if (!isInside(measurement.headTop, imageWidth, imageHeight) ||
                !isInside(measurement.footBottom, imageWidth, imageHeight)) {
            warnings.add("Kepala atau kaki di luar batas gambar");
            Log.w(TAG, "validatePose WARNING: head/foot outside image bounds");
        }

        // 3. Cek kemiringan bahu (maks 35% — cukup toleran untuk pose alami)
        double shoulderTilt = Math.abs(measurement.leftShoulder.y - measurement.rightShoulder.y);
        double shoulderWidth = Math.max(1.0, distance(measurement.leftShoulder, measurement.rightShoulder));
        double shoulderTiltRatio = shoulderTilt / shoulderWidth;
        Log.d(TAG, "validatePose: shoulderTilt=" + String.format(Locale.US, "%.3f", shoulderTiltRatio));
        if (shoulderTiltRatio > 0.35) {
            warnings.add("Bahu terlalu miring (" + String.format(Locale.US, "%.0f%%", shoulderTiltRatio * 100) + ")");
        }

        // 4. Cek kemiringan pinggul (maks 35%)
        double hipTilt = Math.abs(measurement.leftHip.y - measurement.rightHip.y);
        double hipWidth = Math.max(1.0, distance(measurement.leftHip, measurement.rightHip));
        double hipTiltRatio = hipTilt / hipWidth;
        Log.d(TAG, "validatePose: hipTilt=" + String.format(Locale.US, "%.3f", hipTiltRatio));
        if (hipTiltRatio > 0.35) {
            warnings.add("Pinggul terlalu miring (" + String.format(Locale.US, "%.0f%%", hipTiltRatio * 100) + ")");
        }

        // 5. Cek kaki tidak terlalu menekuk (knee angle >= 130 derajat)
        double leftKneeAngle = calculateAngle(measurement.leftHip, measurement.leftKnee, measurement.leftAnkle);
        double rightKneeAngle = calculateAngle(measurement.rightHip, measurement.rightKnee, measurement.rightAnkle);
        Log.d(TAG, "validatePose: leftKneeAngle=" + String.format(Locale.US, "%.1f", leftKneeAngle)
                + " rightKneeAngle=" + String.format(Locale.US, "%.1f", rightKneeAngle));
        if (leftKneeAngle < 130.0 || rightKneeAngle < 130.0) {
            warnings.add("Lutut terlalu menekuk (L=" + String.format(Locale.US, "%.0f°", leftKneeAngle)
                    + " R=" + String.format(Locale.US, "%.0f°", rightKneeAngle) + ")");
        }

        // 6. Cek kemiringan torso (maks 30%)
        double torsoHeight = Math.abs(measurement.hipMid.y - measurement.shoulderMid.y);
        double torsoWidthOffset = Math.abs(measurement.hipMid.x - measurement.shoulderMid.x);
        double torsoTiltRatio = torsoHeight > 0 ? (torsoWidthOffset / torsoHeight) : 0;
        Log.d(TAG, "validatePose: torsoTilt=" + String.format(Locale.US, "%.3f", torsoTiltRatio));
        if (torsoHeight > 0 && torsoTiltRatio > 0.30) {
            warnings.add("Badan terlalu miring (" + String.format(Locale.US, "%.0f%%", torsoTiltRatio * 100) + ")");
        }

        // 7. Cek posisi horizontal (5% - 95% dari lebar gambar — hampir selalu lolos)
        double minXLimit = imageWidth * 0.05;
        double maxXLimit = imageWidth * 0.95;
        if (measurement.headTop.x < minXLimit || measurement.headTop.x > maxXLimit ||
                measurement.footBottom.x < minXLimit || measurement.footBottom.x > maxXLimit) {
            warnings.add("Subjek terlalu di tepi gambar");
            Log.w(TAG, "validatePose WARNING: subject too close to horizontal edge");
        }

        // --- Tentukan hasil ---
        // Pose tetap VALID meskipun ada warning.
        // Hanya gagal total kalau ada >= 4 warning sekaligus (pose benar-benar buruk).
        if (warnings.size() >= 4) {
            validation.valid = false;
            validation.message = "Pose kurang ideal:\n• " + String.join("\n• ", warnings);
            Log.w(TAG, "validatePose FAIL: " + warnings.size() + " warnings — " + String.join("; ", warnings));
        } else if (!warnings.isEmpty()) {
            validation.valid = true;  // TETAP VALID, tapi beri peringatan
            validation.message = "Pose valid dengan catatan:\n• " + String.join("\n• ", warnings);
            Log.i(TAG, "validatePose PASS with warnings: " + String.join("; ", warnings));
        } else {
            validation.valid = true;
            validation.message = "Pose valid — posisi ideal";
            Log.i(TAG, "validatePose PASS — all checks OK");
        }

        return validation;
    }

    private ContourResult runContourProcess(Bitmap bitmap, List<NormalizedLandmark> landmarks) {
        Mat src = new Mat();
        Mat gray = new Mat();
        Mat blur = new Mat();
        Mat edges = new Mat();
        Mat dilated = new Mat();
        Mat hierarchy = new Mat();

        List<MatOfPoint> contours = new ArrayList<>();

        try {
            Utils.bitmapToMat(bitmap, src);

            if (src.channels() == 4) {
                Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (src.channels() == 3) {
                Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = src.clone();
            }

            // Konfigurasi visual final: Gaussian 5 x 5, Canny 60/180,
            // dan dilasi 7 x 7.
            // Kontur hanya menjadi visualisasi pendukung dan tidak digunakan
            // dalam perhitungan tinggi badan.
            Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0);
            Imgproc.Canny(blur, edges, 60, 180);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7));
            Imgproc.dilate(edges, dilated, kernel);
            kernel.release();

            Imgproc.findContours(
                    dilated,
                    contours,
                    hierarchy,
                    Imgproc.RETR_EXTERNAL,
                    Imgproc.CHAIN_APPROX_SIMPLE
            );

            Rect poseRect = getPoseBoundingRect(
                    landmarks,
                    bitmap.getWidth(),
                    bitmap.getHeight(),
                    80
            );

            double bestScore = -1.0;
            MatOfPoint bestContour = null;
            Rect bestRect = null;

            for (MatOfPoint contour : contours) {
                Rect rect = Imgproc.boundingRect(contour);
                double area = Imgproc.contourArea(contour);

                if (area < 300) {
                    continue;
                }

                double overlap = overlapArea(rect, poseRect);

                if (overlap <= 0) {
                    continue;
                }

                double score = overlap + area * 0.03;

                if (score > bestScore) {
                    bestScore = score;
                    bestContour = contour;
                    bestRect = rect;
                }
            }

            ContourResult result = new ContourResult();

            if (bestContour != null && bestRect != null) {
                result.valid = true;
                result.contour = new MatOfPoint(bestContour.toArray());
                result.bodyRect = bestRect;
            } else {
                result.valid = false;
                result.contour = null;
                result.bodyRect = poseRect;
            }

            return result;

        } catch (Exception e) {
            Log.e(TAG, "runContourProcess error", e);

            ContourResult result = new ContourResult();
            result.valid = false;
            result.contour = null;
            result.bodyRect = getPoseBoundingRect(
                    landmarks,
                    bitmap.getWidth(),
                    bitmap.getHeight(),
                    80
            );
            return result;

        } finally {
            for (MatOfPoint c : contours) {
                c.release();
            }

            src.release();
            gray.release();
            blur.release();
            edges.release();
            dilated.release();
            hierarchy.release();
        }
    }

    private Rect getPoseBoundingRect(List<NormalizedLandmark> landmarks, int imageWidth, int imageHeight, int padding) {
        double minX = Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;

        for (NormalizedLandmark lm : landmarks) {
            double x = lm.x() * imageWidth;
            double y = lm.y() * imageHeight;

            if (x < -imageWidth || x > imageWidth * 2.0 ||
                    y < -imageHeight || y > imageHeight * 2.0) {
                continue;
            }

            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }

        if (minX == Double.MAX_VALUE || minY == Double.MAX_VALUE) {
            return new Rect(0, 0, imageWidth, imageHeight);
        }

        int left = clampInt((int) Math.round(minX) - padding, 0, imageWidth - 1);
        int top = clampInt((int) Math.round(minY) - padding, 0, imageHeight - 1);
        int right = clampInt((int) Math.round(maxX) + padding, 0, imageWidth - 1);
        int bottom = clampInt((int) Math.round(maxY) + padding, 0, imageHeight - 1);

        return new Rect(
                left,
                top,
                Math.max(1, right - left),
                Math.max(1, bottom - top)
        );
    }

    private double overlapArea(Rect a, Rect b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);

        int width = Math.max(0, x2 - x1);
        int height = Math.max(0, y2 - y1);

        return width * height;
    }

    private Bitmap drawFinalMeasurement(
            Bitmap source,
            List<NormalizedLandmark> landmarks,
            ContourResult contourResult,
            BodyMeasurement measurement,
            PoseValidation validation,
            double heightCm,
            double skeletonCm,
            int imageHeightPx
    ) {
        Mat mat = new Mat();

        try {
            Bitmap mutable = source.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(mutable, mat);

            int w = mat.cols();
            int h = mat.rows();

            if (contourResult != null && contourResult.valid && contourResult.contour != null) {
                List<MatOfPoint> drawContours = new ArrayList<>();
                drawContours.add(contourResult.contour);

                Imgproc.drawContours(
                        mat,
                        drawContours,
                        -1,
                        new Scalar(255, 128, 0, 255),
                        3
                );

                if (contourResult.bodyRect != null) {
                    Imgproc.rectangle(
                            mat,
                            new Point(contourResult.bodyRect.x, contourResult.bodyRect.y),
                            new Point(
                                    contourResult.bodyRect.x + contourResult.bodyRect.width,
                                    contourResult.bodyRect.y + contourResult.bodyRect.height
                            ),
                            new Scalar(255, 128, 0, 255),
                            3
                    );
                }
            }

            drawPoseConnections(mat, landmarks, w, h);
            drawMeasurementSegments(mat, measurement);

            for (NormalizedLandmark lm : landmarks) {
                Point p = landmarkToPoint(lm, w, h);

                if (isInside(p, w, h)) {
                    Imgproc.circle(mat, p, 5, new Scalar(0, 255, 0, 255), -1);
                }
            }

            String subtitle = "Vertical: " + format1(measurement.verticalHeightPx) +
                    " px | cm/px: " + String.format(Locale.US, "%.5f", this.cmPerPixel);

            String status = validation.valid ? "Pose Valid" : "Pose Tidak Valid";

            drawResultHeader(
                    mat,
                    "TINGGI BADAN: " + format1(heightCm) + " cm",
                    subtitle,
                    status
            );

            Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, resultBitmap);
            return resultBitmap;

        } catch (Exception e) {
            Log.e(TAG, "drawFinalMeasurement error", e);
            return source;

        } finally {
            mat.release();
        }
    }

    private void drawResultHeader(Mat mat, String title, String subtitle, String status) {
        Imgproc.rectangle(
                mat,
                new Point(0, 0),
                new Point(mat.cols(), 165),
                new Scalar(255, 255, 255, 235),
                -1
        );

        Imgproc.putText(
                mat,
                title,
                new Point(24, 48),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.05,
                new Scalar(0, 0, 255, 255),
                3
        );

        Imgproc.putText(
                mat,
                subtitle,
                new Point(24, 92),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.65,
                new Scalar(0, 0, 0, 255),
                2
        );

        Imgproc.putText(
                mat,
                status,
                new Point(24, 132),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.65,
                new Scalar(0, 90, 255, 255),
                2
        );
    }

    private Bitmap drawError(Bitmap source, String message) {
        Mat mat = new Mat();

        try {
            Bitmap mutable = source.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(mutable, mat);

            Imgproc.rectangle(
                    mat,
                    new Point(0, 0),
                    new Point(mat.cols(), 130),
                    new Scalar(255, 255, 255, 235),
                    -1
            );

            Imgproc.putText(
                    mat,
                    message,
                    new Point(24, 70),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    new Scalar(0, 0, 255, 255),
                    3
            );

            Bitmap result = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, result);
            return result;

        } catch (Exception e) {
            return source;

        } finally {
            mat.release();
        }
    }

    private void drawMeasurementSegments(Mat mat, BodyMeasurement measurement) {
        try {
            Scalar skeletonColor = new Scalar(255, 0, 255, 255);
            Scalar heightColor = new Scalar(0, 0, 255, 255);

            Imgproc.line(mat, measurement.headTop, measurement.shoulderMid, skeletonColor, 4);
            Imgproc.line(mat, measurement.shoulderMid, measurement.hipMid, skeletonColor, 4);

            Imgproc.line(mat, measurement.leftHip, measurement.leftKnee, skeletonColor, 4);
            Imgproc.line(mat, measurement.leftKnee, measurement.leftAnkle, skeletonColor, 4);
            Imgproc.line(mat, measurement.leftAnkle, measurement.leftHeel, skeletonColor, 4);
            Imgproc.line(mat, measurement.leftAnkle, measurement.leftFootIndex, skeletonColor, 4);

            Imgproc.line(mat, measurement.rightHip, measurement.rightKnee, skeletonColor, 4);
            Imgproc.line(mat, measurement.rightKnee, measurement.rightAnkle, skeletonColor, 4);
            Imgproc.line(mat, measurement.rightAnkle, measurement.rightHeel, skeletonColor, 4);
            Imgproc.line(mat, measurement.rightAnkle, measurement.rightFootIndex, skeletonColor, 4);

            /*
             * Garis merah adalah tinggi utama:
             * headTop ke footBottom.
             */
            Imgproc.line(mat, measurement.headTop, measurement.footBottom, heightColor, 7);
            Imgproc.circle(mat, measurement.headTop, 10, heightColor, -1);
            Imgproc.circle(mat, measurement.footBottom, 10, heightColor, -1);
            Imgproc.circle(mat, measurement.shoulderMid, 8, skeletonColor, -1);
            Imgproc.circle(mat, measurement.hipMid, 8, skeletonColor, -1);

        } catch (Exception e) {
            Log.e(TAG, "drawMeasurementSegments error", e);
        }
    }

    private void drawPoseConnections(Mat mat, List<NormalizedLandmark> landmarks, int w, int h) {
        int[][] connections = new int[][]{
                {0, 1}, {1, 2}, {2, 3}, {3, 7},
                {0, 4}, {4, 5}, {5, 6}, {6, 8},
                {9, 10},
                {11, 12},
                {11, 13}, {13, 15},
                {12, 14}, {14, 16},
                {11, 23}, {12, 24}, {23, 24},
                {23, 25}, {25, 27}, {27, 29}, {29, 31},
                {24, 26}, {26, 28}, {28, 30}, {30, 32}
        };

        for (int[] pair : connections) {
            int start = pair[0];
            int end = pair[1];

            if (start >= landmarks.size() || end >= landmarks.size()) {
                continue;
            }

            Point p1 = landmarkToPoint(landmarks.get(start), w, h);
            Point p2 = landmarkToPoint(landmarks.get(end), w, h);

            if (!isInside(p1, w, h) || !isInside(p2, w, h)) {
                continue;
            }

            Imgproc.line(mat, p1, p2, new Scalar(0, 255, 255, 255), 3);
        }
    }

    private Point landmarkToPoint(NormalizedLandmark landmark, int w, int h) {
        return new Point(landmark.x() * w, landmark.y() * h);
    }

    private Point midpoint(Point a, Point b) {
        return new Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
    }

    private double distance(Point a, Point b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    private boolean isInside(Point p, int w, int h) {
        return p.x >= 0 && p.x < w && p.y >= 0 && p.y < h;
    }

    private int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
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

            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

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

            String fileName = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
            File imageFile = new File(picturesDir, "MEASUREMENT_" + fileName + ".jpg");

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos);
            fos.flush();
            fos.close();

            Toast.makeText(this, "Gambar hasil measurement berhasil disimpan", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToAppFiles error", e);
            Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show();
        }
    }

    private String format1(double value) {
        return String.format(Locale.US, "%.1f", value);
    }

    private static class ContourResult {
        boolean valid = false;
        MatOfPoint contour = null;
        Rect bodyRect = null;
    }

    private static class PoseValidation {
        boolean valid = true;
        String message = "OK";
    }

    private static class BodyMeasurement {
        Point headTop;
        Point eyeMid;
        Point shoulderMid;
        Point hipMid;
        Point footBottom;

        Point leftShoulder;
        Point rightShoulder;

        Point leftHip;
        Point leftKnee;
        Point leftAnkle;
        Point leftHeel;
        Point leftFootIndex;

        Point rightHip;
        Point rightKnee;
        Point rightAnkle;
        Point rightHeel;
        Point rightFootIndex;

        double headNeckPx;
        double torsoPx;
        double leftLegPx;
        double rightLegPx;
        double legPx;

        double verticalHeightPx;
        double totalSkeletonPx;
    }
}
