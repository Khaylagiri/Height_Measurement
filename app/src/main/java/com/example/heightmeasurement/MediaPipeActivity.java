package com.example.heightmeasurement;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

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
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MediaPipeActivity extends AppCompatActivity {

    private static final String TAG = "MEDIAPIPE_HEIGHT";
    private static final String MODEL_ASSET_PATH = "pose_landmarker_lite.task";
    private static final int MAX_GALLERY_IMAGE_SIDE_PX = 2200;

    private ImageView imageViewMediaPipe;
    private Button btnPickMediaPipeImage;
    private Button btnProcessMediaPipe;
    private Button btnNextMeasurement;

    private Bitmap selectedBitmap;
    private Bitmap currentResultBitmap;

    private PoseLandmarker poseLandmarker;

    /*
     * Data yang sudah dikirim oleh PerspectiveCorrectionActivity lama.
     */
    private double cmPerPixel = -1.0;
    private double boardTopY = -1.0;
    private double boardBottomY = -1.0;
    private double boardPixelHeight = -1.0;
    private double boardRealHeightCm = -1.0;
    private double cameraDistanceCm = -1.0;
    private double subjectDistanceFromBoardCm = -1.0;
    private int expectedOutputWidthPx = -1;
    private int expectedOutputHeightPx = -1;

    /*
     * Path file PNG hasil perspective.
     */
    private String calibratedImagePath = null;
    private boolean calibratedFromPerspective = false;

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri == null) {
                    Toast.makeText(this, "Tidak ada gambar dipilih", Toast.LENGTH_SHORT).show();
                    return;
                }

                Bitmap loadedBitmap = ImageUtils.decodeUri(
                        this,
                        uri,
                        MAX_GALLERY_IMAGE_SIDE_PX
                );

                if (loadedBitmap == null) {
                    Toast.makeText(
                            this,
                            "Gagal membuka gambar. Ukuran foto mungkin terlalu besar atau format tidak didukung.",
                            Toast.LENGTH_LONG
                    ).show();
                    return;
                }

                releaseDisplayedBitmaps();

                selectedBitmap = loadedBitmap;
                currentResultBitmap = null;

                /*
                 * Foto manual bukan hasil perspective, jadi tidak memiliki kalibrasi cm.
                 */
                resetCalibration();

                imageViewMediaPipe.setImageBitmap(selectedBitmap);

                btnPickMediaPipeImage.setVisibility(View.VISIBLE);
                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);

                Toast.makeText(
                        this,
                        "Foto dipilih. Silakan klik Proses MediaPipe",
                        Toast.LENGTH_SHORT
                ).show();
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mediapipe);

        imageViewMediaPipe = findViewById(R.id.imageViewMediaPipe);
        btnPickMediaPipeImage = findViewById(R.id.btnPickMediaPipeImage);
        btnProcessMediaPipe = findViewById(R.id.btnProcessMediaPipe);
        btnNextMeasurement = findViewById(R.id.btnSaveMediaPipe);

        imageViewMediaPipe.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageViewMediaPipe.setAdjustViewBounds(true);

        /*
         * Sembunyikan dulu. loadImageFromIntentIfAvailable() yang menentukan
         * apakah tombol pilih foto perlu ditampilkan atau tidak.
         */
        btnPickMediaPipeImage.setVisibility(View.GONE);
        btnProcessMediaPipe.setVisibility(View.GONE);

        btnNextMeasurement.setText("Next");
        btnNextMeasurement.setVisibility(View.GONE);

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV gagal diinisialisasi", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        initPoseLandmarker();

        btnPickMediaPipeImage.setOnClickListener(v -> {
            Toast.makeText(this, "Pilih foto dari galeri", Toast.LENGTH_SHORT).show();
            pickImageLauncher.launch("image/*");
        });

        btnProcessMediaPipe.setOnClickListener(v -> {
            if (selectedBitmap == null) {
                Toast.makeText(
                        this,
                        "Gambar belum masuk. Pilih foto atau masuk dari Perspective dulu.",
                        Toast.LENGTH_LONG
                ).show();
                return;
            }

            btnNextMeasurement.setVisibility(View.GONE);

            Toast.makeText(this, "Memproses MediaPipe", Toast.LENGTH_SHORT).show();
            processMediaPipePose();
        });

        btnNextMeasurement.setOnClickListener(v -> openMeasurementActivity());

        /*
         * Membaca gambar hasil perspective dari Intent.
         */
        loadImageFromIntentIfAvailable();
    }

    /**
     * Ketika MediaPipeActivity dibuka dari PerspectiveCorrectionActivity,
     * gambar hasil perspective langsung dibaca dari image_path.
     * Pengguna tidak perlu memilih foto lagi.
     */
    private void loadImageFromIntentIfAvailable() {
        try {
            Intent sourceIntent = getIntent();

            if (sourceIntent == null) {
                showManualPicker();
                return;
            }

            String imagePath = sourceIntent.getStringExtra("image_path");
            String uriString = sourceIntent.getStringExtra("image_uri");
            boolean autoProcess = sourceIntent.getBooleanExtra("auto_process", false);

            cmPerPixel = sourceIntent.getDoubleExtra("cm_per_pixel", -1.0);
            boardTopY = sourceIntent.getDoubleExtra("board_top_y", -1.0);
            boardBottomY = sourceIntent.getDoubleExtra("board_bottom_y", -1.0);
            boardPixelHeight = sourceIntent.getDoubleExtra("board_pixel_height", -1.0);
            boardRealHeightCm = sourceIntent.getDoubleExtra("board_real_height_cm", -1.0);
            expectedOutputWidthPx = sourceIntent.getIntExtra("output_width_px", -1);
            expectedOutputHeightPx = sourceIntent.getIntExtra("output_height_px", -1);
            cameraDistanceCm = sourceIntent.getDoubleExtra("camera_distance_cm", -1.0);
            subjectDistanceFromBoardCm = sourceIntent.getDoubleExtra(
                    "subject_distance_from_board_cm",
                    -1.0
            );

            boolean fromPerspective = imagePath != null && !imagePath.trim().isEmpty();

            if (fromPerspective) {
                File imageFile = new File(imagePath);

                if (!imageFile.exists() || !imageFile.isFile()) {
                    Toast.makeText(
                            this,
                            "File hasil perspective tidak ditemukan",
                            Toast.LENGTH_LONG
                    ).show();

                    Log.e(TAG, "File perspective tidak ditemukan: " + imagePath);
                    showManualPicker();
                    return;
                }

                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                options.inSampleSize = 1;

                Bitmap loadedBitmap = BitmapFactory.decodeFile(imagePath, options);

                if (loadedBitmap == null) {
                    Toast.makeText(
                            this,
                            "Gagal membuka gambar hasil perspective",
                            Toast.LENGTH_LONG
                    ).show();

                    showManualPicker();
                    return;
                }

                releaseDisplayedBitmaps();

                selectedBitmap = loadedBitmap;
                currentResultBitmap = null;
                calibratedFromPerspective = true;
                calibratedImagePath = imagePath;

                if (!hasValidCalibration()) {
                    Toast.makeText(
                            this,
                            "Data kalibrasi perspective tidak lengkap",
                            Toast.LENGTH_LONG
                    ).show();
                    Log.e(TAG, buildCalibrationDebug());
                    releaseDisplayedBitmaps();
                    return;
                }

                if (!hasExpectedImageSize(selectedBitmap)) {
                    Toast.makeText(
                            this,
                            "Ukuran hasil perspective berubah: "
                                    + selectedBitmap.getWidth() + "x" + selectedBitmap.getHeight()
                                    + ", seharusnya "
                                    + expectedOutputWidthPx + "x" + expectedOutputHeightPx,
                            Toast.LENGTH_LONG
                    ).show();
                    releaseDisplayedBitmaps();
                    return;
                }

                imageViewMediaPipe.setImageBitmap(selectedBitmap);

                /*
                 * Karena gambar sudah berasal dari perspective,
                 * tombol Pilih Foto disembunyikan.
                 */
                btnPickMediaPipeImage.setVisibility(View.GONE);
                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);

                Log.i(
                        TAG,
                        String.format(
                                Locale.US,
                                "Perspective diterima: path=%s, bitmap=%dx%d, "
                                        + "cmPerPixel=%.6f, boardTop=%.1f, "
                                        + "boardBottom=%.1f, boardHeight=%.1f",
                                imagePath,
                                selectedBitmap.getWidth(),
                                selectedBitmap.getHeight(),
                                cmPerPixel,
                                boardTopY,
                                boardBottomY,
                                boardPixelHeight
                        )
                );

                Toast.makeText(
                        this,
                        "Gambar hasil perspective siap diproses",
                        Toast.LENGTH_SHORT
                ).show();

                if (autoProcess) {
                    processMediaPipePose();
                }

                return;
            }

            /*
             * Kompatibilitas jika Activity menerima image_uri secara langsung.
             */
            if (uriString != null && !uriString.trim().isEmpty()) {
                Bitmap loadedBitmap = ImageUtils.decodeUri(
                        this,
                        Uri.parse(uriString),
                        MAX_GALLERY_IMAGE_SIDE_PX
                );

                if (loadedBitmap == null) {
                    Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_LONG).show();
                    showManualPicker();
                    return;
                }

                releaseDisplayedBitmaps();

                selectedBitmap = loadedBitmap;
                currentResultBitmap = null;
                resetCalibration();

                imageViewMediaPipe.setImageBitmap(selectedBitmap);

                btnPickMediaPipeImage.setVisibility(View.VISIBLE);
                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);

                if (autoProcess) {
                    processMediaPipePose();
                }

                return;
            }

            /*
             * MediaPipeActivity dibuka langsung dari menu utama.
             */
            showManualPicker();

        } catch (OutOfMemoryError error) {
            Log.e(TAG, "Memori tidak cukup membuka hasil perspective", error);

            Toast.makeText(
                    this,
                    "Memori tidak cukup membuka gambar",
                    Toast.LENGTH_LONG
            ).show();

            showManualPicker();

        } catch (Exception e) {
            Log.e(TAG, "loadImageFromIntentIfAvailable error", e);

            Toast.makeText(
                    this,
                    "Error membuka gambar: " + e.getMessage(),
                    Toast.LENGTH_LONG
            ).show();

            showManualPicker();
        }
    }

    private void showManualPicker() {
        calibratedFromPerspective = false;
        calibratedImagePath = null;

        imageViewMediaPipe.setImageBitmap(null);

        btnPickMediaPipeImage.setVisibility(View.VISIBLE);
        btnProcessMediaPipe.setVisibility(View.GONE);
        btnNextMeasurement.setVisibility(View.GONE);
    }

    /**
     * Mengirim foto bersih hasil perspective dan foto overlay landmark
     * ke MeasurementActivity.
     */
    private void openMeasurementActivity() {
        if (currentResultBitmap == null) {
            Toast.makeText(
                    this,
                    "Proses MediaPipe dulu sebelum lanjut ke Measurement",
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        if (!calibratedFromPerspective || !hasValidCalibration()) {
            Toast.makeText(
                    this,
                    "Data kalibrasi perspective tidak valid",
                    Toast.LENGTH_LONG
            ).show();
            Log.e(TAG, buildCalibrationDebug());
            return;
        }

        if (calibratedImagePath == null || calibratedImagePath.trim().isEmpty()) {
            Toast.makeText(this, "Path hasil perspective tidak tersedia", Toast.LENGTH_LONG).show();
            return;
        }

        File cleanFile = new File(calibratedImagePath);
        if (!cleanFile.exists() || !cleanFile.isFile()) {
            Toast.makeText(this, "File hasil perspective tidak ditemukan", Toast.LENGTH_LONG).show();
            return;
        }

        String overlayPath = saveBitmapToCache(
                currentResultBitmap,
                "MEDIAPIPE_OVERLAY"
        );

        if (overlayPath == null) {
            Toast.makeText(this, "Gagal menyiapkan gambar landmark", Toast.LENGTH_LONG).show();
            return;
        }

        Intent intent = new Intent(MediaPipeActivity.this, MeasurementActivity.class);

        // Bersih: dipakai untuk menghitung ulang landmark/tinggi.
        intent.putExtra("image_path", calibratedImagePath);

        // Overlay: langsung ditampilkan di halaman Measurement.
        intent.putExtra("overlay_path", overlayPath);

        intent.putExtra("cm_per_pixel", cmPerPixel);
        intent.putExtra("board_top_y", boardTopY);
        intent.putExtra("board_bottom_y", boardBottomY);
        intent.putExtra("board_pixel_height", boardPixelHeight);
        intent.putExtra("board_real_height_cm", boardRealHeightCm);
        intent.putExtra("output_width_px", expectedOutputWidthPx);
        intent.putExtra("output_height_px", expectedOutputHeightPx);
        intent.putExtra("camera_distance_cm", cameraDistanceCm);
        intent.putExtra(
                "subject_distance_from_board_cm",
                subjectDistanceFromBoardCm
        );

        startActivity(intent);
    }

    private String saveBitmapToCache(Bitmap bitmap, String prefix) {
        FileOutputStream outputStream = null;

        try {
            File directory = new File(getCacheDir(), "measurement_flow");
            if (!directory.exists() && !directory.mkdirs()) {
                return null;
            }

            String timestamp = new SimpleDateFormat(
                    "yyyyMMdd_HHmmss_SSS",
                    Locale.US
            ).format(new Date());

            File outputFile = new File(
                    directory,
                    prefix + "_" + timestamp + ".png"
            );

            outputStream = new FileOutputStream(outputFile);
            boolean success = bitmap.compress(
                    Bitmap.CompressFormat.PNG,
                    100,
                    outputStream
            );
            outputStream.flush();

            return success ? outputFile.getAbsolutePath() : null;

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToCache error", e);
            return null;

        } finally {
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    private boolean hasValidCalibration() {
        if (cmPerPixel <= 0.0
                || boardTopY < 0.0
                || boardBottomY <= boardTopY
                || boardPixelHeight <= 0.0
                || boardRealHeightCm <= 0.0
                || expectedOutputWidthPx <= 0
                || expectedOutputHeightPx <= 0
                || cameraDistanceCm <= 0.0
                || subjectDistanceFromBoardCm < 0.0
                || subjectDistanceFromBoardCm >= cameraDistanceCm) {
            return false;
        }

        double boardHeightDifference = Math.abs(
                boardPixelHeight - (boardBottomY - boardTopY)
        );

        double reconstructedBoardCm = boardPixelHeight * cmPerPixel;

        return boardHeightDifference <= 2.0
                && Math.abs(reconstructedBoardCm - boardRealHeightCm) <= 0.5;
    }

    private boolean hasExpectedImageSize(Bitmap bitmap) {
        return bitmap != null
                && bitmap.getWidth() == expectedOutputWidthPx
                && bitmap.getHeight() == expectedOutputHeightPx;
    }

    private String buildCalibrationDebug() {
        return String.format(
                Locale.US,
                "CALIBRATION cmPerPixel=%.6f top=%.1f bottom=%.1f "
                        + "boardPx=%.1f boardCm=%.1f output=%dx%d "
                        + "camera=%.1f subject=%.1f",
                cmPerPixel,
                boardTopY,
                boardBottomY,
                boardPixelHeight,
                boardRealHeightCm,
                expectedOutputWidthPx,
                expectedOutputHeightPx,
                cameraDistanceCm,
                subjectDistanceFromBoardCm
        );
    }

    private void resetCalibration() {
        cmPerPixel = -1.0;
        boardTopY = -1.0;
        boardBottomY = -1.0;
        boardPixelHeight = -1.0;
        boardRealHeightCm = -1.0;
        cameraDistanceCm = -1.0;
        subjectDistanceFromBoardCm = -1.0;
        expectedOutputWidthPx = -1;
        expectedOutputHeightPx = -1;
        calibratedImagePath = null;
        calibratedFromPerspective = false;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (poseLandmarker != null) {
            poseLandmarker.close();
            poseLandmarker = null;
        }

        releaseDisplayedBitmaps();
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

            Toast.makeText(this, "MediaPipe siap", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "initPoseLandmarker error", e);

            Toast.makeText(
                    this,
                    "MediaPipe gagal init. Cek file pose_landmarker_lite.task di assets.",
                    Toast.LENGTH_LONG
            ).show();
        }
    }

    private void processMediaPipePose() {
        try {
            btnNextMeasurement.setVisibility(View.GONE);

            if (poseLandmarker == null) {
                Toast.makeText(this, "PoseLandmarker belum siap", Toast.LENGTH_LONG).show();

                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            if (selectedBitmap == null) {
                Toast.makeText(this, "Gambar belum tersedia", Toast.LENGTH_LONG).show();

                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            MPImage mpImage = new BitmapImageBuilder(selectedBitmap).build();
            PoseLandmarkerResult result = poseLandmarker.detect(mpImage);

            if (result == null || result.landmarks().isEmpty()) {
                Toast.makeText(
                        this,
                        "Tubuh tidak terdeteksi oleh MediaPipe",
                        Toast.LENGTH_LONG
                ).show();

                currentResultBitmap = drawError(
                        selectedBitmap,
                        "Tubuh tidak terdeteksi"
                );

                imageViewMediaPipe.setImageBitmap(currentResultBitmap);

                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            List<NormalizedLandmark> landmarks = result.landmarks().get(0);

            if (landmarks.size() < 33) {
                currentResultBitmap = drawError(
                        selectedBitmap,
                        "Landmark tubuh tidak lengkap"
                );

                imageViewMediaPipe.setImageBitmap(currentResultBitmap);

                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            currentResultBitmap = drawPoseAndHeight(selectedBitmap, landmarks);
            imageViewMediaPipe.setImageBitmap(currentResultBitmap);

            btnProcessMediaPipe.setVisibility(View.GONE);

            if (calibratedFromPerspective && hasValidCalibration()) {
                btnPickMediaPipeImage.setVisibility(View.GONE);
                btnNextMeasurement.setVisibility(View.VISIBLE);

                Toast.makeText(
                        this,
                        "MediaPipe selesai diproses",
                        Toast.LENGTH_SHORT
                ).show();

            } else {
                btnPickMediaPipeImage.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);

                Toast.makeText(
                        this,
                        "Pose terdeteksi. Untuk menghitung cm, mulai dari Perspective Correction.",
                        Toast.LENGTH_LONG
                ).show();
            }

        } catch (OutOfMemoryError oom) {
            Log.e(TAG, "processMediaPipePose out of memory", oom);

            Toast.makeText(
                    this,
                    "Memori tidak cukup untuk memproses gambar",
                    Toast.LENGTH_LONG
            ).show();

            btnProcessMediaPipe.setVisibility(View.VISIBLE);
            btnNextMeasurement.setVisibility(View.GONE);

        } catch (Exception e) {
            Log.e(TAG, "processMediaPipePose error", e);

            Toast.makeText(
                    this,
                    "Error MediaPipe: " + e.getMessage(),
                    Toast.LENGTH_LONG
            ).show();

            currentResultBitmap = drawError(
                    selectedBitmap,
                    "ERROR: " + e.getMessage()
            );

            imageViewMediaPipe.setImageBitmap(currentResultBitmap);

            btnProcessMediaPipe.setVisibility(View.VISIBLE);
            btnNextMeasurement.setVisibility(View.GONE);
        }
    }

    private Bitmap drawPoseAndHeight(
            Bitmap source,
            List<NormalizedLandmark> landmarks
    ) {
        Mat mat = new Mat();

        try {
            Bitmap mutable = source.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(mutable, mat);

            int w = mat.cols();
            int h = mat.rows();

            drawPoseConnections(mat, landmarks, w, h);

            double minX = Double.MAX_VALUE;
            double minY = Double.MAX_VALUE;
            double maxX = -Double.MAX_VALUE;
            double maxY = -Double.MAX_VALUE;

            for (NormalizedLandmark lm : landmarks) {
                double x = lm.x() * w;
                double y = lm.y() * h;

                if (x < 0 || x >= w || y < 0 || y >= h) {
                    continue;
                }

                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);

                Imgproc.circle(
                        mat,
                        new Point(x, y),
                        6,
                        new Scalar(0, 255, 0, 255),
                        -1
                );
            }

            if (minX == Double.MAX_VALUE || minY == Double.MAX_VALUE) {
                return drawError(source, "Landmark tidak valid");
            }

            double bodyHeightPxRaw = maxY - minY;
            double headCorrectionPx = bodyHeightPxRaw * 0.08;
            double footCorrectionPx = bodyHeightPxRaw * 0.03;

            minY = Math.max(0, minY - headCorrectionPx);
            maxY = Math.min(h - 1, maxY + footCorrectionPx);

            int left = (int) Math.max(0, minX - 40);
            int top = (int) Math.max(0, minY);
            int right = (int) Math.min(w - 1, maxX + 40);
            int bottom = (int) Math.min(h - 1, maxY);

            Imgproc.rectangle(
                    mat,
                    new Point(left, top),
                    new Point(right, bottom),
                    new Scalar(255, 0, 0, 255),
                    4
            );

            int centerX = (left + right) / 2;

            Imgproc.line(
                    mat,
                    new Point(centerX, top),
                    new Point(centerX, bottom),
                    new Scalar(0, 255, 255, 255),
                    4
            );

            Imgproc.circle(
                    mat,
                    new Point(centerX, top),
                    10,
                    new Scalar(0, 0, 255, 255),
                    -1
            );

            Imgproc.circle(
                    mat,
                    new Point(centerX, bottom),
                    10,
                    new Scalar(255, 255, 0, 255),
                    -1
            );

            drawMeasurementSegments(mat, landmarks, w, h);

            Bitmap resultBitmap = Bitmap.createBitmap(
                    mat.cols(),
                    mat.rows(),
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(mat, resultBitmap);
            return resultBitmap;

        } catch (Exception e) {
            Log.e(TAG, "drawPoseAndHeight error", e);
            return drawError(source, "Draw error: " + e.getMessage());

        } finally {
            mat.release();
        }
    }

    private Point estimateHeadTop(Point nose, Point shoulderMid) {
        double noseToShoulder = distance(nose, shoulderMid);
        double headTopCorrection = noseToShoulder * 0.45;

        double dirX = nose.x - shoulderMid.x;
        double dirY = nose.y - shoulderMid.y;
        double len = Math.sqrt(dirX * dirX + dirY * dirY);

        if (len > 0) {
            dirX /= len;
            dirY /= len;

            return new Point(
                    nose.x + dirX * headTopCorrection,
                    nose.y + dirY * headTopCorrection
            );
        }

        return new Point(
                nose.x,
                nose.y - headTopCorrection
        );
    }

    private void drawMeasurementSegments(
            Mat mat,
            List<NormalizedLandmark> landmarks,
            int w,
            int h
    ) {
        try {
            Point nose = landmarkToPoint(landmarks.get(0), w, h);

            Point leftShoulder = landmarkToPoint(landmarks.get(11), w, h);
            Point rightShoulder = landmarkToPoint(landmarks.get(12), w, h);

            Point leftHip = landmarkToPoint(landmarks.get(23), w, h);
            Point rightHip = landmarkToPoint(landmarks.get(24), w, h);

            Point leftKnee = landmarkToPoint(landmarks.get(25), w, h);
            Point rightKnee = landmarkToPoint(landmarks.get(26), w, h);

            Point leftAnkle = landmarkToPoint(landmarks.get(27), w, h);
            Point rightAnkle = landmarkToPoint(landmarks.get(28), w, h);

            Point leftHeel = landmarkToPoint(landmarks.get(29), w, h);
            Point rightHeel = landmarkToPoint(landmarks.get(30), w, h);

            Point leftFoot = landmarkToPoint(landmarks.get(31), w, h);
            Point rightFoot = landmarkToPoint(landmarks.get(32), w, h);

            Point shoulderMid = midpoint(leftShoulder, rightShoulder);
            Point hipMid = midpoint(leftHip, rightHip);
            Point headTop = estimateHeadTop(nose, shoulderMid);

            Scalar segmentColor = new Scalar(255, 0, 255, 255);

            Imgproc.line(mat, headTop, shoulderMid, segmentColor, 5);
            Imgproc.line(mat, shoulderMid, hipMid, segmentColor, 5);

            Imgproc.line(mat, leftHip, leftKnee, segmentColor, 5);
            Imgproc.line(mat, leftKnee, leftAnkle, segmentColor, 5);
            Imgproc.line(mat, leftAnkle, leftHeel, segmentColor, 5);
            Imgproc.line(mat, leftAnkle, leftFoot, segmentColor, 5);

            Imgproc.line(mat, rightHip, rightKnee, segmentColor, 5);
            Imgproc.line(mat, rightKnee, rightAnkle, segmentColor, 5);
            Imgproc.line(mat, rightAnkle, rightHeel, segmentColor, 5);
            Imgproc.line(mat, rightAnkle, rightFoot, segmentColor, 5);

            Imgproc.circle(mat, headTop, 9, new Scalar(255, 0, 255, 255), -1);
            Imgproc.circle(mat, shoulderMid, 9, new Scalar(255, 0, 255, 255), -1);
            Imgproc.circle(mat, hipMid, 9, new Scalar(255, 0, 255, 255), -1);

        } catch (Exception e) {
            Log.e(TAG, "drawMeasurementSegments error", e);
        }
    }

    private void drawPoseConnections(
            Mat mat,
            List<NormalizedLandmark> landmarks,
            int w,
            int h
    ) {
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

            NormalizedLandmark p1 = landmarks.get(start);
            NormalizedLandmark p2 = landmarks.get(end);

            double x1 = p1.x() * w;
            double y1 = p1.y() * h;
            double x2 = p2.x() * w;
            double y2 = p2.y() * h;

            if (!isPointInside(x1, y1, w, h)
                    || !isPointInside(x2, y2, w, h)) {
                continue;
            }

            Imgproc.line(
                    mat,
                    new Point(x1, y1),
                    new Point(x2, y2),
                    new Scalar(0, 255, 255, 255),
                    3
            );
        }
    }

    private Point landmarkToPoint(
            NormalizedLandmark landmark,
            int w,
            int h
    ) {
        return new Point(
                landmark.x() * w,
                landmark.y() * h
        );
    }

    private Point midpoint(Point a, Point b) {
        return new Point(
                (a.x + b.x) / 2.0,
                (a.y + b.y) / 2.0
        );
    }

    private double distance(Point a, Point b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;

        return Math.sqrt(dx * dx + dy * dy);
    }

    private boolean isPointInside(
            double x,
            double y,
            int w,
            int h
    ) {
        return x >= 0 && x < w && y >= 0 && y < h;
    }

    private Bitmap drawError(Bitmap source, String message) {
        Mat mat = new Mat();

        try {
            Bitmap mutable = source.copy(Bitmap.Config.ARGB_8888, true);
            Utils.bitmapToMat(mutable, mat);

            Imgproc.rectangle(
                    mat,
                    new Point(0, 0),
                    new Point(mat.cols(), 160),
                    new Scalar(255, 255, 255, 230),
                    -1
            );

            Imgproc.putText(
                    mat,
                    message,
                    new Point(20, 65),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    new Scalar(0, 0, 255, 255),
                    3
            );

            Bitmap result = Bitmap.createBitmap(
                    mat.cols(),
                    mat.rows(),
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(mat, result);
            return result;

        } catch (Exception e) {
            return source;

        } finally {
            mat.release();
        }
    }

    private void releaseDisplayedBitmaps() {
        imageViewMediaPipe.setImageBitmap(null);

        if (currentResultBitmap != null
                && currentResultBitmap != selectedBitmap
                && !currentResultBitmap.isRecycled()) {
            currentResultBitmap.recycle();
        }

        if (selectedBitmap != null && !selectedBitmap.isRecycled()) {
            selectedBitmap.recycle();
        }

        currentResultBitmap = null;
        selectedBitmap = null;
    }
}
