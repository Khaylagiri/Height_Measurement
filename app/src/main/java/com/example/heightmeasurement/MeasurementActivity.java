package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MeasurementActivity extends AppCompatActivity {

    private ImageView imageViewMeasurement;
    private TextView tvMeasurementResult;
    private Button btnProcessMeasurement;
    private Button btnSaveMeasurement;

    private Bitmap originalBitmap;
    private Bitmap currentBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_measurement);

        imageViewMeasurement = findViewById(R.id.imageViewMeasurement);
        tvMeasurementResult = findViewById(R.id.tvMeasurementResult);
        btnProcessMeasurement = findViewById(R.id.btnProcessMeasurement);
        btnSaveMeasurement = findViewById(R.id.btnSaveMeasurement);

        imageViewMeasurement.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageViewMeasurement.setAdjustViewBounds(true);

        String uriString = getIntent().getStringExtra("image_uri");

        if (uriString == null) {
            Toast.makeText(this, "Gambar tidak ditemukan", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        originalBitmap = loadAndRotateBitmap(Uri.parse(uriString));

        if (originalBitmap == null) {
            Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        currentBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        imageViewMeasurement.setImageBitmap(currentBitmap);

        tvMeasurementResult.setText("Siap memproses tinggi badan");

        btnProcessMeasurement.setOnClickListener(v -> runMeasurementPipeline());

        btnSaveMeasurement.setOnClickListener(v -> {
            if (currentBitmap != null) {
                saveBitmapToAppFiles(currentBitmap);
            }
        });
    }

    private void runMeasurementPipeline() {
        /*
         * Pipeline final yang nanti kamu isi:
         *
         * 1. Perspective Correction
         * 2. MediaPipe Pose Detection
         * 3. Contour Refinement
         * 4. Hitung tinggi badan
         * 5. Tampilkan hasil
         *
         * Untuk sekarang activity ini sudah aman:
         * - tidak pakai kode OpenCvCamera
         * - tidak pakai Delta
         * - tidak ada error headCm / footCm
         * - tidak ada debug line
         */

        if (originalBitmap == null) {
            Toast.makeText(this, "Gambar belum tersedia", Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap workingBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);

        /*
         * TODO 1:
         * Ganti bagian ini dengan fungsi perspective correction kamu.
         *
         * Contoh nanti:
         * Bitmap perspectiveBitmap = runPerspectiveCorrection(workingBitmap);
         */
        Bitmap perspectiveBitmap = workingBitmap;

        /*
         * TODO 2:
         * Ganti bagian ini dengan deteksi MediaPipe.
         *
         * Contoh nanti:
         * PoseResult poseResult = runMediaPipeDetection(perspectiveBitmap);
         */
        boolean mediaPipeReady = true;

        /*
         * TODO 3:
         * Ganti bagian ini dengan proses contour.
         *
         * Contoh nanti:
         * ContourResult contourResult = runContourRefinement(perspectiveBitmap, poseResult);
         */
        boolean contourReady = true;

        /*
         * TODO 4:
         * Ganti nilai dummy ini dengan hasil hitung tinggi badan asli.
         */
        double heightCm = 0.0;

        currentBitmap = drawMeasurementStatus(
                perspectiveBitmap,
                "Measurement Pipeline Ready",
                "Perspective → MediaPipe → Contour → Height"
        );

        imageViewMeasurement.setImageBitmap(currentBitmap);

        if (mediaPipeReady && contourReady && heightCm > 0) {
            tvMeasurementResult.setText(
                    "Tinggi badan: " + String.format(Locale.US, "%.1f", heightCm) + " cm"
            );
        } else {
            tvMeasurementResult.setText(
                    "Pipeline siap. Masukkan kode Perspective, MediaPipe, Contour, dan perhitungan tinggi."
            );
        }

        Toast.makeText(this, "Measurement pipeline siap", Toast.LENGTH_SHORT).show();
    }

    private Bitmap drawMeasurementStatus(Bitmap bitmap, String title, String subtitle) {
        try {
            Bitmap result = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(result);

            Paint bgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            bgPaint.setColor(Color.argb(190, 17, 24, 39));

            Paint titlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            titlePaint.setColor(Color.WHITE);
            titlePaint.setTextSize(42f);
            titlePaint.setFakeBoldText(true);

            Paint subtitlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            subtitlePaint.setColor(Color.rgb(224, 242, 254));
            subtitlePaint.setTextSize(30f);

            float padding = 28f;
            float boxHeight = 150f;
            float left = 30f;
            float top = 30f;
            float right = result.getWidth() - 30f;
            float bottom = top + boxHeight;

            RectF rect = new RectF(left, top, right, bottom);
            canvas.drawRoundRect(rect, 24f, 24f, bgPaint);

            canvas.drawText(title, left + padding, top + 58f, titlePaint);
            canvas.drawText(subtitle, left + padding, top + 108f, subtitlePaint);

            return result;

        } catch (Exception e) {
            return bitmap;
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
            return null;
        }
    }

    private Bitmap loadBitmapFromUri(Uri uri) {
        try (InputStream inputStream = getContentResolver().openInputStream(uri)) {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            return BitmapFactory.decodeStream(inputStream, null, options);

        } catch (Exception e) {
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
            return bitmap;
        }
    }

    private void saveBitmapToAppFiles(Bitmap bitmap) {
        try {
            File picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

            if (picturesDir != null && !picturesDir.exists()) {
                picturesDir.mkdirs();
            }

            String fileName = new SimpleDateFormat(
                    "yyyyMMdd_HHmmss",
                    Locale.getDefault()
            ).format(new Date());

            File imageFile = new File(
                    picturesDir,
                    "MEASUREMENT_" + fileName + ".jpg"
            );

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos);
            fos.flush();
            fos.close();

            Toast.makeText(this, "Gambar berhasil disimpan", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show();
        }
    }
}