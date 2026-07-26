package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;
import android.graphics.BitmapFactory;

public class GalleryOpenCvActivity extends AppCompatActivity {

    private static final String TAG = "GalleryOpenCvActivity";
    private ImageView imageView;
    private TextView tvResult;
    private Bitmap selectedBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_measurement);

        imageView = findViewById(R.id.imageViewMeasurement);
        tvResult = findViewById(R.id.tvMeasurementResult);

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV gagal dimuat.");
        }

        String uriString = getIntent().getStringExtra("image_uri");
        if (uriString != null) {
            loadBitmap(Uri.parse(uriString));
        }
    }

    private void loadBitmap(Uri uri) {
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            selectedBitmap = BitmapFactory.decodeStream(is);
            imageView.setImageBitmap(selectedBitmap);
        } catch (Exception e) {
            Log.e(TAG, "Error loading bitmap", e);
        }
    }

    private void processImage() {
        if (selectedBitmap == null) return;
        Mat src = new Mat();
        Utils.bitmapToMat(selectedBitmap, src);
        // Implementasi logika ArUco & Pose di sini
        src.release();
    }
}
