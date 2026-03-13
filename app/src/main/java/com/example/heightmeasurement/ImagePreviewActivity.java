package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

public class ImagePreviewActivity extends AppCompatActivity {

    private ImageView imageViewPreview;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_preview);

        imageViewPreview = findViewById(R.id.imageViewPreview);

        String imagePath = getIntent().getStringExtra("image_path");

        if (imagePath != null) {
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            if (bitmap != null) {
                imageViewPreview.setImageBitmap(bitmap);
            } else {
                Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show();
            }
        } else {
            Toast.makeText(this, "Path gambar tidak ditemukan", Toast.LENGTH_SHORT).show();
        }
    }
}