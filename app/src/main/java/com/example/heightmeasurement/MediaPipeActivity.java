package com.example.heightmeasurement;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
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
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MediaPipeActivity extends AppCompatActivity {

    private static final String TAG = "MEDIAPIPE_HEIGHT";
    private static final String MODEL_ASSET_PATH = "pose_landmarker_lite.task";
    private static final double DEFAULT_CAMERA_DISTANCE_CM = 150.0;

    private ImageView imageViewMediaPipe;
    private Button btnPickMediaPipeImage;
    private Button btnProcessMediaPipe;
    private Button btnNextMeasurement;

    private Bitmap selectedBitmap;
    private Bitmap currentResultBitmap;

    private PoseLandmarker poseLandmarker;
    private double cmPerPixel = -1.0;
    private double boardTopY = -1.0;
    private double boardBottomY = -1.0;
    private double boardPixelHeight = -1.0;
    private double cameraDistanceCm = DEFAULT_CAMERA_DISTANCE_CM;

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri == null) {
                    Toast.makeText(this, "Tidak ada gambar dipilih", Toast.LENGTH_SHORT).show();
                    return;
                }

                selectedBitmap = loadAndRotateBitmap(uri);

                if (selectedBitmap == null) {
                    Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_LONG).show();
                    return;
                }

                currentResultBitmap = null;
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

        btnPickMediaPipeImage.setVisibility(View.VISIBLE);
        btnProcessMediaPipe.setVisibility(View.VISIBLE);

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

        loadImageFromIntentIfAvailable();
    }

    private void loadImageFromIntentIfAvailable() {
        try {
            String imagePath = getIntent().getStringExtra("image_path");
            String uriString = getIntent().getStringExtra("image_uri");
            boolean autoProcess = getIntent().getBooleanExtra("auto_process", false);
            cmPerPixel = getIntent().getDoubleExtra("cm_per_pixel", -1.0);
            boardTopY = getIntent().getDoubleExtra("board_top_y", -1.0);
            boardBottomY = getIntent().getDoubleExtra("board_bottom_y", -1.0);
            boardPixelHeight = getIntent().getDoubleExtra("board_pixel_height", -1.0);
            cameraDistanceCm = getIntent().getDoubleExtra(
                    "camera_distance_cm",
                    DEFAULT_CAMERA_DISTANCE_CM
            );

            boolean fromPerspective = imagePath != null && !imagePath.trim().isEmpty();

            if (!fromPerspective && (uriString == null || uriString.trim().isEmpty())) {
                btnPickMediaPipeImage.setVisibility(View.VISIBLE);
                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            if (fromPerspective) {
                selectedBitmap = loadBitmapFromPath(imagePath);
            } else {
                selectedBitmap = loadAndRotateBitmap(Uri.parse(uriString));
            }

            if (selectedBitmap == null) {
                Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_LONG).show();

                if (fromPerspective) {
                    btnPickMediaPipeImage.setVisibility(View.GONE);
                } else {
                    btnPickMediaPipeImage.setVisibility(View.VISIBLE);
                }

                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            currentResultBitmap = null;
            imageViewMediaPipe.setImageBitmap(selectedBitmap);

            btnProcessMediaPipe.setVisibility(View.VISIBLE);
            btnNextMeasurement.setVisibility(View.GONE);

            if (fromPerspective) {
                btnPickMediaPipeImage.setVisibility(View.GONE);

                Toast.makeText(
                        this,
                        "Gambar hasil perspective siap diproses",
                        Toast.LENGTH_SHORT
                ).show();

            } else {
                btnPickMediaPipeImage.setVisibility(View.VISIBLE);

                Toast.makeText(
                        this,
                        "Foto siap diproses",
                        Toast.LENGTH_SHORT
                ).show();
            }

            if (autoProcess) {
                processMediaPipePose();
            }

        } catch (Exception e) {
            Log.e(TAG, "loadImageFromIntentIfAvailable error", e);
            Toast.makeText(
                    this,
                    "Error membuka gambar: " + e.getMessage(),
                    Toast.LENGTH_LONG
            ).show();

            btnProcessMediaPipe.setVisibility(View.VISIBLE);
            btnNextMeasurement.setVisibility(View.GONE);
        }
    }

    private Bitmap loadBitmapFromPath(String imagePath) {
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;

            return BitmapFactory.decodeFile(imagePath, options);

        } catch (Exception e) {
            Log.e(TAG, "loadBitmapFromPath error", e);
            return null;
        }
    }

    private void openMeasurementActivity() {
        if (selectedBitmap == null || currentResultBitmap == null) {
            Toast.makeText(
                    this,
                    "Proses MediaPipe dulu sebelum lanjut ke Measurement",
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        /*
         * Kirim DUA gambar:
         * 1. selectedBitmap = gambar bersih untuk proses hitung tinggi.
         * 2. currentResultBitmap = gambar yang sudah berisi titik/garis landmark
         *    untuk langsung ditampilkan ketika halaman Measurement dibuka.
         *
         * Dengan cara ini landmark tidak hilang, tetapi proses pengukuran tetap
         * memakai gambar bersih supaya garis hasil gambar tidak mengganggu contour.
         */
        String cleanImageUri = saveBitmapToCacheForMeasurement(
                selectedBitmap,
                "CLEAN"
        );

        String landmarkImageUri = saveBitmapToCacheForMeasurement(
                currentResultBitmap,
                "LANDMARK"
        );

        if (cleanImageUri == null || landmarkImageUri == null) {
            Toast.makeText(
                    this,
                    "Gagal menyiapkan gambar untuk Measurement",
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        Intent intent = new Intent(MediaPipeActivity.this, MeasurementActivity.class);
        intent.putExtra("image_uri", cleanImageUri);
        intent.putExtra("landmark_image_uri", landmarkImageUri);
        intent.putExtra("cm_per_pixel", cmPerPixel);
        intent.putExtra("board_top_y", boardTopY);
        intent.putExtra("board_bottom_y", boardBottomY);
        intent.putExtra("board_pixel_height", boardPixelHeight);
        intent.putExtra("camera_distance_cm", cameraDistanceCm);
        startActivity(intent);
    }

    private String saveBitmapToCacheForMeasurement(Bitmap bitmap, String prefix) {
        FileOutputStream fos = null;

        try {
            File cacheDir = new File(getCacheDir(), "mediapipe_result");

            if (!cacheDir.exists()) {
                cacheDir.mkdirs();
            }

            String fileName = new SimpleDateFormat(
                    "yyyyMMdd_HHmmss",
                    Locale.getDefault()
            ).format(new Date());

            File imageFile = new File(
                    cacheDir,
                    prefix + "_MEDIAPIPE_TO_MEASUREMENT_" + fileName + ".png"
            );

            fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();

            return Uri.fromFile(imageFile).toString();

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToCacheForMeasurement error", e);
            return null;

        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (Exception ignored) {
                }
            }
        }
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
                Toast.makeText(this, "Tubuh tidak terdeteksi oleh MediaPipe", Toast.LENGTH_LONG).show();

                currentResultBitmap = drawError(selectedBitmap, "Tubuh tidak terdeteksi");
                imageViewMediaPipe.setImageBitmap(currentResultBitmap);

                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            List<NormalizedLandmark> landmarks = result.landmarks().get(0);

            if (landmarks.size() < 33) {
                currentResultBitmap = drawError(selectedBitmap, "Landmark tubuh tidak lengkap");
                imageViewMediaPipe.setImageBitmap(currentResultBitmap);

                btnProcessMediaPipe.setVisibility(View.VISIBLE);
                btnNextMeasurement.setVisibility(View.GONE);
                return;
            }

            currentResultBitmap = drawPoseAndHeight(selectedBitmap, landmarks);
            imageViewMediaPipe.setImageBitmap(currentResultBitmap);

            btnPickMediaPipeImage.setVisibility(View.GONE);
            btnProcessMediaPipe.setVisibility(View.GONE);
            btnNextMeasurement.setVisibility(View.VISIBLE);

            Toast.makeText(this, "MediaPipe selesai diproses", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "processMediaPipePose error", e);
            Toast.makeText(this, "Error MediaPipe: " + e.getMessage(), Toast.LENGTH_LONG).show();

            currentResultBitmap = drawError(selectedBitmap, "ERROR: " + e.getMessage());
            imageViewMediaPipe.setImageBitmap(currentResultBitmap);

            btnProcessMediaPipe.setVisibility(View.VISIBLE);
            btnNextMeasurement.setVisibility(View.GONE);
        }
    }

    private Bitmap drawPoseAndHeight(Bitmap source, List<NormalizedLandmark> landmarks) {
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

            /*
             * Tidak menampilkan tulisan tinggi di halaman MediaPipe.
             * Halaman ini hanya untuk validasi pose/landmark.
             */
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
        } else {
            return new Point(
                    nose.x,
                    nose.y - headTopCorrection
            );
        }
    }

    private void drawMeasurementSegments(Mat mat, List<NormalizedLandmark> landmarks, int w, int h) {
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

            NormalizedLandmark p1 = landmarks.get(start);
            NormalizedLandmark p2 = landmarks.get(end);

            double x1 = p1.x() * w;
            double y1 = p1.y() * h;
            double x2 = p2.x() * w;
            double y2 = p2.y() * h;

            if (!isPointInside(x1, y1, w, h) || !isPointInside(x2, y2, w, h)) {
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

    private Point landmarkToPoint(NormalizedLandmark landmark, int w, int h) {
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

    private boolean isPointInside(double x, double y, int w, int h) {
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
}
