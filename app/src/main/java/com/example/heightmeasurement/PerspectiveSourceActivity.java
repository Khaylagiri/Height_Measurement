package com.example.heightmeasurement;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class PerspectiveSourceActivity extends AppCompatActivity {

    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    openPerspectiveCorrection(uri, -1.0);
                }
            });

    private final ActivityResultLauncher<Intent> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    String imageUri = result.getData().getStringExtra("image_uri");
                    if (imageUri != null) {
                        double cameraDistanceCm = result.getData().getDoubleExtra(
                                "camera_distance_cm",
                                -1.0
                        );
                        openPerspectiveCorrection(Uri.parse(imageUri), cameraDistanceCm);
                    }
                }
            });

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    launchRealTimeCamera();
                } else {
                    Toast.makeText(
                            this,
                            "Izin kamera diperlukan untuk membuka kamera real time",
                            Toast.LENGTH_LONG
                    ).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_perspective_source);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.perspectiveSourceRoot), (view, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        TextView btnBack = findViewById(R.id.btnBack);
        LinearLayout btnRealTimeCamera = findViewById(R.id.btnRealTimeCamera);
        LinearLayout btnGallery = findViewById(R.id.btnGallery);

        btnBack.setOnClickListener(view -> finish());
        btnRealTimeCamera.setOnClickListener(view -> checkCameraPermissionAndLaunch());
        btnGallery.setOnClickListener(view -> galleryLauncher.launch("image/*"));
    }

    private void checkCameraPermissionAndLaunch() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            launchRealTimeCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void launchRealTimeCamera() {
        Intent intent = new Intent(this, CameraDistanceActivity.class);
        cameraLauncher.launch(intent);
    }

    private void openPerspectiveCorrection(Uri imageUri, double cameraDistanceCm) {
        Intent intent = new Intent(this, PerspectiveCorrectionActivity.class);
        intent.putExtra("image_uri", imageUri.toString());
        if (cameraDistanceCm > 0) {
            intent.putExtra("camera_distance_cm", cameraDistanceCm);
        }
        startActivity(intent);
        finish();
    }
}
