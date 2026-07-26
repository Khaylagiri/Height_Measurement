package com.example.heightmeasurement;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.camera2.CameraCharacteristics;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.util.Size;
import android.util.SizeF;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.DetectorParameters;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@ExperimentalCamera2Interop
public class CameraDistanceActivity extends AppCompatActivity {

    private static final String TAG = "CameraDistanceActivity";

    private static final int CAMERA_PERMISSION_REQUEST = 1001;

    private static final String CALIBRATION_PREFS =
            "camera_distance_calibration";

    private static final String CALIBRATION_FACTOR_KEY =
            "factor";

    private static final int MARKER_DICT =
            Objdetect.DICT_6X6_1000;

    /*
     * Ukuran fisik satu marker:
     * 0,74 sel × (200 cm / 19,35 sel)
     */
    private static final double REAL_MARKER_SIZE_CM = 7.6485;

    private static final double TARGET_DISTANCE_CM = 150.0;
    private static final double TARGET_TOLERANCE_CM = 10.0;

    // Nilai ini hanya untuk tampilan realtime. Logika deteksi tetap memakai target 150 cm.
    private static final double DISPLAY_DISTANCE_OFFSET_CM = 200.0;

    /*
     * Digunakan jika karakteristik kamera tidak berhasil dibaca.
     */
    private static final double DEFAULT_FOCAL_SENSOR_WIDTH_RATIO = 0.8;

    private static final double DISTANCE_SMOOTHING_ALPHA = 0.25;

    /*
     * Analisis dilakukan maksimal sekitar empat kali per detik.
     */
    private static final long ANALYSIS_INTERVAL_NS =
            250_000_000L;

    private PreviewView viewFinder;
    private TextView tvDistance;
    private View viewDistanceIndicator;
    private ImageButton btnCapture;
    private ImageButton btnBack;
    private Button btnCalibrateDistance;

    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    private ImageAnalysis imageAnalysis;
    private ImageCapture imageCapture;
    private ArucoDetector arucoDetector;

    private double currentDistanceCm = -1;
    private double currentRawDistanceCm = -1;
    private double smoothedDistanceCm = -1;
    private double distanceCalibrationFactor = 1.0;

    /*
     * focal length kamera / lebar sensor kamera.
     */
    private volatile double focalSensorWidthRatio = -1;

    private boolean isReadyToCapture = false;
    private boolean hasVibrated = false;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera_distance);

        initializeViews();

        cameraExecutor = Executors.newSingleThreadExecutor();

        btnBack.setOnClickListener(view -> finish());

        btnCapture.setOnClickListener(view -> takePhoto());

        checkCameraPermission();
    }

    private void initializeViews() {
        viewFinder = findViewById(R.id.viewFinder);
        tvDistance = findViewById(R.id.tvDistance);

        viewDistanceIndicator =
                findViewById(R.id.viewDistanceIndicator);

        btnCapture = findViewById(R.id.btnCapture);
        btnBack = findViewById(R.id.btnBack);

        btnCalibrateDistance =
                findViewById(R.id.btnCalibrateDistance);

        btnCapture.setEnabled(false);
    }

    private void loadCalibration() {
        SharedPreferences preferences = getSharedPreferences(
                CALIBRATION_PREFS,
                MODE_PRIVATE
        );

        distanceCalibrationFactor = preferences.getFloat(
                CALIBRATION_FACTOR_KEY,
                1.0f
        );
    }

    private boolean initializeOpenCv() {
        try {
            if (!OpenCVLoader.initLocal()) {
                Toast.makeText(
                        this,
                        "OpenCV gagal diinisialisasi",
                        Toast.LENGTH_LONG
                ).show();

                finish();
                return false;
            }

            return true;

        } catch (Throwable throwable) {
            Log.e(
                    TAG,
                    "OpenCV gagal diinisialisasi",
                    throwable
            );

            Toast.makeText(
                    this,
                    "OpenCV gagal diinisialisasi",
                    Toast.LENGTH_LONG
            ).show();

            finish();
            return false;
        }
    }

    private void initializeArucoDetector() {
        Dictionary dictionary =
                Objdetect.getPredefinedDictionary(MARKER_DICT);

        DetectorParameters parameters =
                createArucoDetectorParameters();

        arucoDetector = new ArucoDetector(
                dictionary,
                parameters
        );
    }

    private void checkCameraPermission() {
        int permissionStatus = ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
        );

        if (permissionStatus == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST
            );
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(
                requestCode,
                permissions,
                grantResults
        );

        if (requestCode != CAMERA_PERMISSION_REQUEST) {
            return;
        }

        boolean permissionGranted =
                grantResults.length > 0
                        && grantResults[0]
                        == PackageManager.PERMISSION_GRANTED;

        if (permissionGranted) {
            startCamera();
        } else {
            Toast.makeText(
                    this,
                    "Izin kamera diperlukan untuk melakukan pengukuran",
                    Toast.LENGTH_LONG
            ).show();

            finish();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(this);

        providerFuture.addListener(() -> {
            try {
                cameraProvider = providerFuture.get();

                if (isFinishing() || isDestroyed()) {
                    return;
                }

                Preview preview =
                        new Preview.Builder().build();

                preview.setSurfaceProvider(
                        viewFinder.getSurfaceProvider()
                );

                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(
                                ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY
                        )
                        .build();

                CameraSelector cameraSelector =
                        CameraSelector.DEFAULT_BACK_CAMERA;

                cameraProvider.unbindAll();

                cameraProvider.bindToLifecycle(
                                CameraDistanceActivity.this,
                                cameraSelector,
                                preview,
                                imageCapture
                        );

                runOnUiThread(() ->
                        btnCapture.setEnabled(true)
                );

            } catch (Exception exception) {
                Log.e(
                        TAG,
                        "Gagal menginisialisasi CameraX",
                        exception
                );

                Toast.makeText(
                        CameraDistanceActivity.this,
                        "Gagal membuka kamera: "
                                + exception.getMessage(),
                        Toast.LENGTH_LONG
                ).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void calculateFocalLength(
            @NonNull Camera camera
    ) {
        try {
            Camera2CameraInfo camera2CameraInfo =
                    Camera2CameraInfo.from(
                            camera.getCameraInfo()
                    );

            float[] focalLengths =
                    camera2CameraInfo.getCameraCharacteristic(
                            CameraCharacteristics
                                    .LENS_INFO_AVAILABLE_FOCAL_LENGTHS
                    );

            SizeF sensorSize =
                    camera2CameraInfo.getCameraCharacteristic(
                            CameraCharacteristics
                                    .SENSOR_INFO_PHYSICAL_SIZE
                    );

            if (focalLengths == null
                    || focalLengths.length == 0
                    || sensorSize == null
                    || sensorSize.getWidth() <= 0) {

                focalSensorWidthRatio =
                        DEFAULT_FOCAL_SENSOR_WIDTH_RATIO;

                Log.w(
                        TAG,
                        "Karakteristik kamera tidak lengkap. "
                                + "Menggunakan nilai focal default."
                );

                return;
            }

            float focalLengthMm = focalLengths[0];
            float sensorWidthMm = sensorSize.getWidth();

            focalSensorWidthRatio =
                    focalLengthMm / sensorWidthMm;

            Log.d(
                    TAG,
                    "Focal length: " + focalLengthMm
                            + " mm, sensor width: "
                            + sensorWidthMm
                            + " mm, ratio: "
                            + focalSensorWidthRatio
            );

        } catch (Exception exception) {
            focalSensorWidthRatio =
                    DEFAULT_FOCAL_SENSOR_WIDTH_RATIO;

            Log.e(
                    TAG,
                    "Gagal membaca CameraCharacteristics",
                    exception
            );
        }
    }

    private void detectDistance(
            @NonNull ImageProxy image
    ) {
        Mat grayMat = null;
        Mat ids = null;

        List<Mat> corners = new ArrayList<>();
        List<Mat> rejected = new ArrayList<>();

        try {
            if (image.getPlanes().length == 0) {
                updateDistanceOnUiThread(-1);
                return;
            }

            ImageProxy.PlaneProxy yPlane =
                    image.getPlanes()[0];

            ByteBuffer yBuffer =
                    yPlane.getBuffer().duplicate();

            int width = image.getWidth();
            int height = image.getHeight();

            int rowStride = yPlane.getRowStride();
            int pixelStride = yPlane.getPixelStride();

            byte[] grayData =
                    copyYPlaneToGrayArray(
                            yBuffer,
                            width,
                            height,
                            rowStride,
                            pixelStride
                    );

            grayMat = new Mat(
                    height,
                    width,
                    CvType.CV_8UC1
            );

            grayMat.put(0, 0, grayData);

            double ratio =
                    focalSensorWidthRatio > 0
                            ? focalSensorWidthRatio
                            : DEFAULT_FOCAL_SENSOR_WIDTH_RATIO;

            double focalLengthPx = width * ratio;

            ids = new Mat();

            arucoDetector.detectMarkers(
                    grayMat,
                    corners,
                    ids,
                    rejected
            );

            double totalDistance = 0;
            int validMarkerCount = 0;

            if (!ids.empty() && !corners.isEmpty()) {
                int markerCount = Math.min(
                        ids.rows(),
                        corners.size()
                );

                for (int markerIndex = 0;
                     markerIndex < markerCount;
                     markerIndex++) {

                    Mat markerCorner =
                            corners.get(markerIndex);

                    Point[] markerPoints =
                            readMarkerPoints(markerCorner);

                    if (markerPoints == null) {
                        continue;
                    }

                    double firstSide =
                            calculatePointDistance(
                                    markerPoints[0],
                                    markerPoints[1]
                            );

                    double secondSide =
                            calculatePointDistance(
                                    markerPoints[1],
                                    markerPoints[2]
                            );

                    double thirdSide =
                            calculatePointDistance(
                                    markerPoints[2],
                                    markerPoints[3]
                            );

                    double fourthSide =
                            calculatePointDistance(
                                    markerPoints[3],
                                    markerPoints[0]
                            );

                    double averageSidePx =
                            (
                                    firstSide
                                            + secondSide
                                            + thirdSide
                                            + fourthSide
                            ) / 4.0;

                    if (averageSidePx <= 0) {
                        continue;
                    }

                    double markerDistanceCm =
                            (
                                    REAL_MARKER_SIZE_CM
                                            * focalLengthPx
                            ) / averageSidePx;

                    if (Double.isNaN(markerDistanceCm)
                            || Double.isInfinite(markerDistanceCm)
                            || markerDistanceCm <= 0) {
                        continue;
                    }

                    totalDistance += markerDistanceCm;
                    validMarkerCount++;
                }
            }

            double finalDistanceCm =
                    validMarkerCount > 0
                            ? totalDistance / validMarkerCount
                            : -1;

            updateDistanceOnUiThread(finalDistanceCm);

        } catch (Exception exception) {
            Log.e(
                    TAG,
                    "Terjadi kesalahan saat mendeteksi jarak",
                    exception
            );

            updateDistanceOnUiThread(-1);

        } finally {
            if (grayMat != null) {
                grayMat.release();
            }

            if (ids != null) {
                ids.release();
            }

            for (Mat corner : corners) {
                corner.release();
            }

            for (Mat rejectedMarker : rejected) {
                rejectedMarker.release();
            }

            image.close();
        }
    }

    private byte[] copyYPlaneToGrayArray(
            @NonNull ByteBuffer buffer,
            int width,
            int height,
            int rowStride,
            int pixelStride
    ) {
        byte[] grayData =
                new byte[width * height];

        byte[] rowData =
                new byte[Math.max(rowStride, width)];

        for (int row = 0; row < height; row++) {
            int rowStart = row * rowStride;

            if (rowStart >= buffer.limit()) {
                break;
            }

            buffer.position(rowStart);

            int bytesAvailable =
                    buffer.limit() - rowStart;

            int bytesToRead =
                    Math.min(rowStride, bytesAvailable);

            buffer.get(
                    rowData,
                    0,
                    bytesToRead
            );

            for (int column = 0;
                 column < width;
                 column++) {

                int sourceIndex =
                        column * pixelStride;

                if (sourceIndex >= bytesToRead) {
                    break;
                }

                grayData[row * width + column] =
                        rowData[sourceIndex];
            }
        }

        return grayData;
    }

    @Nullable
    private Point[] readMarkerPoints(
            @NonNull Mat markerCorner
    ) {
        double[][] values = new double[4][];

        if (markerCorner.rows() == 1
                && markerCorner.cols() >= 4) {

            values[0] = markerCorner.get(0, 0);
            values[1] = markerCorner.get(0, 1);
            values[2] = markerCorner.get(0, 2);
            values[3] = markerCorner.get(0, 3);

        } else if (markerCorner.cols() == 1
                && markerCorner.rows() >= 4) {

            values[0] = markerCorner.get(0, 0);
            values[1] = markerCorner.get(1, 0);
            values[2] = markerCorner.get(2, 0);
            values[3] = markerCorner.get(3, 0);

        } else {
            return null;
        }

        Point[] points = new Point[4];

        for (int index = 0; index < 4; index++) {
            if (values[index] == null
                    || values[index].length < 2) {
                return null;
            }

            points[index] = new Point(
                    values[index][0],
                    values[index][1]
            );
        }

        return points;
    }

    private double calculatePointDistance(
            @NonNull Point firstPoint,
            @NonNull Point secondPoint
    ) {
        double differenceX =
                secondPoint.x - firstPoint.x;

        double differenceY =
                secondPoint.y - firstPoint.y;

        return Math.sqrt(
                differenceX * differenceX
                        + differenceY * differenceY
        );
    }

    private void updateDistanceOnUiThread(
            double distanceCm
    ) {
        runOnUiThread(() -> {
            if (!isFinishing() && !isDestroyed()) {
                updateUi(distanceCm);
            }
        });
    }

    private void updateUi(double distanceCm) {
        if (distanceCm <= 0) {
            currentDistanceCm = -1;
            currentRawDistanceCm = -1;
            smoothedDistanceCm = -1;

            tvDistance.setText(
                    "Papan marker tidak terdeteksi"
            );

            tvDistance.setTextColor(
                    Color.parseColor("#FFCDD2")
            );

            viewDistanceIndicator.setBackgroundColor(
                    Color.parseColor("#EF5350")
            );

            btnCapture.setBackgroundTintList(
                    ContextCompat.getColorStateList(
                            this,
                            android.R.color.darker_gray
                    )
            );

            isReadyToCapture = false;
            hasVibrated = false;
            return;
        }

        currentRawDistanceCm = distanceCm;

        double calibratedDistanceCm =
                distanceCm * distanceCalibrationFactor;

        if (smoothedDistanceCm < 0) {
            smoothedDistanceCm =
                    calibratedDistanceCm;
        } else {
            smoothedDistanceCm =
                    DISTANCE_SMOOTHING_ALPHA
                            * calibratedDistanceCm
                            + (
                            1.0
                                    - DISTANCE_SMOOTHING_ALPHA
                    ) * smoothedDistanceCm;
        }

        currentDistanceCm = smoothedDistanceCm;

        // Tampilkan seolah-olah acuan kamera berada di 350 cm:
        // 150 cm internal + 200 cm offset tampilan = 350 cm.
        double displayedDistanceCm =
                currentDistanceCm + DISPLAY_DISTANCE_OFFSET_CM;

        double minimumDistanceCm =
                TARGET_DISTANCE_CM
                        - TARGET_TOLERANCE_CM;

        double maximumDistanceCm =
                TARGET_DISTANCE_CM
                        + TARGET_TOLERANCE_CM;

        boolean distanceIsCorrect =
                currentDistanceCm >= minimumDistanceCm
                        && currentDistanceCm
                        <= maximumDistanceCm;

        if (distanceIsCorrect) {
            tvDistance.setText(
                    String.format(
                            Locale.getDefault(),
                            "Jarak: %.1f cm • Siap mengambil foto",
                            displayedDistanceCm
                    )
            );

            tvDistance.setTextColor(
                    Color.parseColor("#A5D6A7")
            );

            viewDistanceIndicator.setBackgroundColor(
                    Color.parseColor("#4CAF50")
            );

            btnCapture.setBackgroundTintList(
                    ContextCompat.getColorStateList(
                            this,
                            android.R.color.holo_green_dark
                    )
            );

            isReadyToCapture = true;

            if (!hasVibrated) {
                triggerVibration();
                hasVibrated = true;
            }

        } else {
            String instruction;

            if (currentDistanceCm < minimumDistanceCm) {
                instruction = "Mundur menuju 350 cm";
            } else {
                instruction = "Maju menuju 350 cm";
            }

            tvDistance.setText(
                    String.format(
                            Locale.getDefault(),
                            "Jarak: %.1f cm • %s",
                            displayedDistanceCm,
                            instruction
                    )
            );

            tvDistance.setTextColor(
                    Color.parseColor("#FFCDD2")
            );

            viewDistanceIndicator.setBackgroundColor(
                    Color.parseColor("#EF5350")
            );

            btnCapture.setBackgroundTintList(
                    ContextCompat.getColorStateList(
                            this,
                            android.R.color.holo_red_dark
                    )
            );

            isReadyToCapture = false;
            hasVibrated = false;
        }
    }

    private void calibrateAtTargetDistance() {
        if (currentRawDistanceCm <= 0) {
            Toast.makeText(
                    this,
                    "Marker belum terdeteksi. "
                            + "Arahkan kamera ke marker terlebih dahulu.",
                    Toast.LENGTH_LONG
            ).show();

            return;
        }

        double newCalibrationFactor =
                TARGET_DISTANCE_CM
                        / currentRawDistanceCm;

        if (newCalibrationFactor < 0.25
                || newCalibrationFactor > 4.0) {

            Toast.makeText(
                    this,
                    "Kalibrasi gagal. Pastikan jarak nyata "
                            + "tepat 350 cm dan marker terlihat jelas.",
                    Toast.LENGTH_LONG
            ).show();

            return;
        }

        distanceCalibrationFactor =
                newCalibrationFactor;

        smoothedDistanceCm = -1;

        getSharedPreferences(
                CALIBRATION_PREFS,
                MODE_PRIVATE
        )
                .edit()
                .putFloat(
                        CALIBRATION_FACTOR_KEY,
                        (float) newCalibrationFactor
                )
                .apply();

        Toast.makeText(
                this,
                "Kalibrasi jarak 350 cm berhasil disimpan",
                Toast.LENGTH_SHORT
        ).show();
    }

    @SuppressWarnings("deprecation")
    private void triggerVibration() {
        try {
            Vibrator vibrator =
                    (Vibrator) getSystemService(
                            Context.VIBRATOR_SERVICE
                    );

            if (vibrator == null
                    || !vibrator.hasVibrator()) {
                return;
            }

            if (Build.VERSION.SDK_INT
                    >= Build.VERSION_CODES.O) {

                vibrator.vibrate(
                        VibrationEffect.createOneShot(
                                150,
                                VibrationEffect.DEFAULT_AMPLITUDE
                        )
                );

            } else {
                vibrator.vibrate(150);
            }

        } catch (Throwable throwable) {
            Log.w(
                    TAG,
                    "Vibrasi tidak dapat dijalankan",
                    throwable
            );
        }
    }

    private void takePhoto() {
        if (imageCapture == null) {
            Toast.makeText(
                    this,
                    "Kamera belum siap",
                    Toast.LENGTH_SHORT
            ).show();

            return;
        }

        File imageDirectory =
                new File(getCacheDir(), "images");

        if (!imageDirectory.exists()
                && !imageDirectory.mkdirs()) {

            Toast.makeText(
                    this,
                    "Folder gambar tidak dapat dibuat",
                    Toast.LENGTH_LONG
            ).show();

            return;
        }

        String fileName =
                "TEMP_CAMERA_CAPTURE_"
                        + new SimpleDateFormat(
                        "yyyyMMdd_HHmmss_SSS",
                        Locale.getDefault()
                ).format(new Date())
                        + ".jpg";

        File photoFile =
                new File(imageDirectory, fileName);

        ImageCapture.OutputFileOptions outputOptions =
                new ImageCapture.OutputFileOptions.Builder(
                        photoFile
                ).build();

        btnCapture.setEnabled(false);
        tvDistance.setText("Mengambil foto...");

        imageCapture.takePicture(
                outputOptions,
                ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {

                    @Override
                    public void onImageSaved(
                            @NonNull ImageCapture.OutputFileResults
                                    outputFileResults
                    ) {
                        btnCapture.setEnabled(true);

                        try {
                            Uri savedUri =
                                    FileProvider.getUriForFile(
                                            CameraDistanceActivity.this,
                                            getPackageName()
                                                    + ".fileprovider",
                                            photoFile
                                    );

                            Intent resultIntent =
                                    new Intent();

                            resultIntent.putExtra(
                                    "image_uri",
                                    savedUri.toString()
                            );

                            resultIntent.addFlags(
                                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                            );

                            setResult(
                                    RESULT_OK,
                                    resultIntent
                            );

                            finish();

                        } catch (Exception exception) {
                            Log.e(
                                    TAG,
                                    "Gagal membuat URI gambar",
                                    exception
                            );

                            Toast.makeText(
                                    CameraDistanceActivity.this,
                                    "Gagal membuka hasil gambar: "
                                            + exception.getMessage(),
                                    Toast.LENGTH_LONG
                            ).show();
                        }
                    }

                    @Override
                    public void onError(
                            @NonNull ImageCaptureException exception
                    ) {
                        btnCapture.setEnabled(true);

                        Log.e(
                                TAG,
                                "Gagal mengambil gambar",
                                exception
                        );

                        Toast.makeText(
                                CameraDistanceActivity.this,
                                "Gagal mengambil foto: "
                                        + exception.getMessage(),
                                Toast.LENGTH_LONG
                        ).show();
                    }
                }
        );
    }

    private DetectorParameters
    createArucoDetectorParameters() {
        DetectorParameters parameters =
                new DetectorParameters();

        parameters.set_adaptiveThreshWinSizeMin(3);
        parameters.set_adaptiveThreshWinSizeMax(53);
        parameters.set_adaptiveThreshWinSizeStep(4);

        parameters.set_minMarkerPerimeterRate(0.004);
        parameters.set_maxMarkerPerimeterRate(4.0);

        parameters.set_polygonalApproxAccuracyRate(0.06);
        parameters.set_minCornerDistanceRate(0.005);
        parameters.set_minDistanceToBorder(1);

        parameters.set_cornerRefinementMethod(1);
        parameters.set_cornerRefinementWinSize(7);
        parameters.set_cornerRefinementMaxIterations(80);
        parameters.set_cornerRefinementMinAccuracy(0.01);

        return parameters;
    }

    @Override
    protected void onDestroy() {
        if (imageAnalysis != null) {
            imageAnalysis.clearAnalyzer();
        }

        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }

        if (cameraExecutor != null) {
            cameraExecutor.shutdownNow();
        }

        super.onDestroy();
    }
}
