package com.example.heightmeasurement;

import android.content.Intent;
import android.os.Bundle;
import android.os.Environment;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
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

            return true;
        });
    }

    private void loadImageFiles() {
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