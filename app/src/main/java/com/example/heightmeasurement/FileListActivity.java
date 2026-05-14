package com.example.heightmeasurement;

import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileInputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;

public class FileListActivity extends AppCompatActivity {

    private ListView listView;
    private final ArrayList<File> imageFiles = new ArrayList<>();
    private final ArrayList<String> fileNames = new ArrayList<>();
    private ArrayAdapter<String> adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_file_list);

        listView = findViewById(R.id.listViewFiles);

        loadImageFiles();

        if (fileNames.isEmpty()) {
            Toast.makeText(this, "Belum ada foto hasil capture", Toast.LENGTH_SHORT).show();
        }

        adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_list_item_1,
                fileNames
        );

        listView.setAdapter(adapter);

        listView.setOnItemClickListener((parent, view, position, id) -> {
            Intent intent = new Intent(FileListActivity.this, ImagePreviewActivity.class);
            intent.putExtra("image_path", imageFiles.get(position).getAbsolutePath());
            startActivity(intent);
        });

        listView.setOnItemLongClickListener((parent, view, position, id) -> {
            showFileOptions(position);
            return true;
        });
    }

    private void showFileOptions(int position) {
        String[] options = {"Simpan ke Galeri", "Hapus", "Batal"};

        new AlertDialog.Builder(this)
                .setTitle(fileNames.get(position))
                .setItems(options, (dialog, which) -> {
                    if (which == 0) {
                        saveImageToGallery(imageFiles.get(position));
                    } else if (which == 1) {
                        deleteImage(position);
                    } else {
                        dialog.dismiss();
                    }
                })
                .show();
    }

    private void saveImageToGallery(File sourceFile) {
        try {
            String fileName = sourceFile.getName();
            String mimeType = fileName.toLowerCase().endsWith(".png")
                    ? "image/png"
                    : "image/jpeg";

            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
            values.put(MediaStore.Images.Media.MIME_TYPE, mimeType);
            values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/HeightMeasurement");

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.Images.Media.IS_PENDING, 1);
            }

            Uri uri = getContentResolver().insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    values
            );

            if (uri == null) {
                Toast.makeText(this, "Gagal membuat file di galeri", Toast.LENGTH_SHORT).show();
                return;
            }

            try (
                    FileInputStream inputStream = new FileInputStream(sourceFile);
                    OutputStream outputStream = getContentResolver().openOutputStream(uri)
            ) {
                if (outputStream == null) {
                    Toast.makeText(this, "Gagal membuka galeri", Toast.LENGTH_SHORT).show();
                    return;
                }

                byte[] buffer = new byte[8192];
                int length;

                while ((length = inputStream.read(buffer)) > 0) {
                    outputStream.write(buffer, 0, length);
                }

                outputStream.flush();
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContentValues updateValues = new ContentValues();
                updateValues.put(MediaStore.Images.Media.IS_PENDING, 0);
                getContentResolver().update(uri, updateValues, null, null);
            }

            Toast.makeText(this, "Foto berhasil disimpan ke galeri", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Gagal menyimpan ke galeri: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void deleteImage(int position) {
        File fileToDelete = imageFiles.get(position);

        boolean deleted = fileToDelete.delete();

        if (deleted) {
            Toast.makeText(this, "Foto berhasil dihapus", Toast.LENGTH_SHORT).show();
            imageFiles.remove(position);
            fileNames.remove(position);
            adapter.notifyDataSetChanged();
        } else {
            Toast.makeText(this, "Gagal menghapus foto", Toast.LENGTH_SHORT).show();
        }
    }

    private void loadImageFiles() {
        imageFiles.clear();
        fileNames.clear();

        File picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

        if (picturesDir != null && picturesDir.exists()) {
            File[] files = picturesDir.listFiles((dir, name) ->
                    name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png"));

            if (files != null) {
                Arrays.sort(files, (f1, f2) -> Long.compare(f2.lastModified(), f1.lastModified()));

                for (File file : files) {
                    imageFiles.add(file);
                    fileNames.add(file.getName());
                }
            }
        }
    }
}