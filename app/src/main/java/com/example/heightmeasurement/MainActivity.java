package com.example.heightmeasurement;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    private LinearLayout btnPerspective;
    private LinearLayout btnMediaPipe;
    private LinearLayout btnMeasurement;
    private LinearLayout btnFile;

    private final ActivityResultLauncher<String> perspectiveLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    openPerspective(uri);
                } else {
                    Toast.makeText(this, "Tidak ada gambar dipilih", Toast.LENGTH_SHORT).show();
                }
            });

    private final ActivityResultLauncher<String> measurementLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    openMeasurement(uri);
                } else {
                    Toast.makeText(this, "Tidak ada gambar dipilih", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (view, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            view.setPadding(
                    systemBars.left,
                    systemBars.top,
                    systemBars.right,
                    systemBars.bottom
            );
            return insets;
        });

        btnPerspective = findViewById(R.id.btnPerspective);
        btnMediaPipe = findViewById(R.id.btnMediaPipe);
        btnMeasurement = findViewById(R.id.btnMeasurement);
        btnFile = findViewById(R.id.btnFile);

        btnPerspective.setOnClickListener(view -> openPerspectivePicker());

        btnMediaPipe.setOnClickListener(view -> openMediaPipe());

        btnMeasurement.setOnClickListener(view -> openMeasurementPicker());

        btnFile.setOnClickListener(view -> openFileList());
    }

    private void openPerspectivePicker() {
        perspectiveLauncher.launch("image/*");
    }

    private void openMeasurementPicker() {
        measurementLauncher.launch("image/*");
    }

    private void openPerspective(Uri uri) {
        Intent intent = new Intent(MainActivity.this, PerspectiveCorrectionActivity.class);
        intent.putExtra("image_uri", uri.toString());
        startActivity(intent);
    }

    private void openMediaPipe() {
        Intent intent = new Intent(MainActivity.this, MediaPipeActivity.class);
        startActivity(intent);
    }

    private void openMeasurement(Uri uri) {
        // Pengukuran cm wajib melewati Perspective Correction agar memiliki skala valid.
        Intent intent = new Intent(MainActivity.this, PerspectiveCorrectionActivity.class);
        intent.putExtra("image_uri", uri.toString());
        startActivity(intent);
    }

    private void openFileList() {
        Intent intent = new Intent(MainActivity.this, FileListActivity.class);
        startActivity(intent);
    }
}
