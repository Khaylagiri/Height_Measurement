package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
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
import java.util.Locale;

public class OpenCvCameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private JavaCameraView javaCameraView;
    private FrameLayout btnCapture;

    private final Object frameLock = new Object();
    private Mat latestFrame;
    private boolean isSaving = false;

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

        btnCapture.setOnClickListener(v -> capturePhoto());
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

        int baseY = rgbaFrame.rows() - 70;

        // shadow / outline hitam
        Imgproc.putText(
                rgbaFrame,
                line1,
                new Point(30, baseY),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.8,
                new Scalar(0, 0, 0, 255),
                5
        );

        Imgproc.putText(
                rgbaFrame,
                line2,
                new Point(30, baseY + 40),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.8,
                new Scalar(0, 0, 0, 255),
                5
        );

        // teks putih utama
        Imgproc.putText(
                rgbaFrame,
                line1,
                new Point(30, baseY),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.8,
                new Scalar(255, 255, 255, 255),
                2
        );

        Imgproc.putText(
                rgbaFrame,
                line2,
                new Point(30, baseY + 40),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.8,
                new Scalar(255, 255, 255, 255),
                2
        );

        synchronized (frameLock) {
            if (latestFrame != null) {
                latestFrame.release();
            }
            latestFrame = rgbaFrame.clone();
        }

        return rgbaFrame;
    }

    private void capturePhoto() {
        if (isSaving) {
            Toast.makeText(this, "Sedang menyimpan foto...", Toast.LENGTH_SHORT).show();
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
        Toast.makeText(this, "Mengambil foto...", Toast.LENGTH_SHORT).show();

        new Thread(() -> {
            saveFrameToFile(capturedMat);
            capturedMat.release();
            isSaving = false;
        }).start();
    }

    private void saveFrameToFile(Mat mat) {
        try {
            Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, bitmap);

            File picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
            if (picturesDir != null && !picturesDir.exists()) {
                picturesDir.mkdirs();
            }

            String fileName = new SimpleDateFormat(
                    "yyyyMMdd_HHmmss",
                    Locale.getDefault()
            ).format(new Date());

            File imageFile = new File(picturesDir, "IMG_" + fileName + ".jpg");

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();

            Log.d("OpenCV", "Foto tersimpan: " + imageFile.getAbsolutePath());

            runOnUiThread(() ->
                    Toast.makeText(
                            OpenCvCameraActivity.this,
                            "Foto berhasil disimpan",
                            Toast.LENGTH_SHORT
                    ).show()
            );

        } catch (Exception e) {
            Log.e("OpenCV", "Gagal menyimpan foto: " + e.getMessage(), e);

            runOnUiThread(() ->
                    Toast.makeText(
                            OpenCvCameraActivity.this,
                            "Gagal menyimpan foto",
                            Toast.LENGTH_SHORT
                    ).show()
            );
        }
    }
}