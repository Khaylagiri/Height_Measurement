package com.example.heightmeasurement;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {

    private LinearLayout btnPhoto, btnFile;

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    openOpenCvCamera();
                } else {
                    Toast.makeText(this, "Permission kamera ditolak", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        if (OpenCVLoader.initLocal()) {
            Log.i("opencv", "successfully integrated");
        } else {
            Log.e("opencv", "failed to integrate");
        }

        btnPhoto = findViewById(R.id.btnPhoto);
        btnFile = findViewById(R.id.btnFile);

        btnPhoto.setOnClickListener(v -> checkCameraPermissionAndOpen());

        btnFile.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, FileListActivity.class);
            startActivity(intent);
        });
    }

    private void checkCameraPermissionAndOpen() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            openOpenCvCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void openOpenCvCamera() {
        Intent intent = new Intent(MainActivity.this, OpenCvCameraActivity.class);
        startActivity(intent);
    }
}