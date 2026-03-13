package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class GalleryOpenCvActivity extends AppCompatActivity {

    private ImageView imageViewResult;
    private Button btnSaveGalleryImage;
    private Bitmap processedBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery_opencv);

        imageViewResult = findViewById(R.id.imageViewResult);
        btnSaveGalleryImage = findViewById(R.id.btnSaveGalleryImage);

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV gagal diinisialisasi", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        String uriString = getIntent().getStringExtra("image_uri");
        if (uriString == null) {
            Toast.makeText(this, "Gambar tidak ditemukan", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        Uri imageUri = Uri.parse(uriString);
        processedBitmap = processImageWithTimestamp(imageUri);

        if (processedBitmap != null) {
            imageViewResult.setImageBitmap(processedBitmap);
        } else {
            Toast.makeText(this, "Gagal memproses gambar", Toast.LENGTH_SHORT).show();
        }

        btnSaveGalleryImage.setOnClickListener(v -> {
            if (processedBitmap != null) {
                saveBitmapToAppFiles(processedBitmap);
            }
        });
    }

    private Bitmap processImageWithTimestamp(Uri imageUri) {
        try {
            Bitmap originalBitmap = loadBitmapFromUri(imageUri);
            if (originalBitmap == null) return null;

            Bitmap rotatedBitmap = rotateBitmapIfRequired(originalBitmap, imageUri);
            Bitmap mutableBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true);

            Mat mat = new Mat();
            Utils.bitmapToMat(mutableBitmap, mat);

            String line1 = new SimpleDateFormat(
                    "EEEE, dd MMMM yyyy",
                    new Locale("id", "ID")
            ).format(new Date());

            String line2 = new SimpleDateFormat(
                    "HH:mm:ss",
                    new Locale("id", "ID")
            ).format(new Date());

            drawTimestamp(mat, line1, line2);

            Bitmap resultBitmap = Bitmap.createBitmap(
                    mat.cols(),
                    mat.rows(),
                    Bitmap.Config.ARGB_8888
            );
            Utils.matToBitmap(mat, resultBitmap);
            mat.release();

            return resultBitmap;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void drawTimestamp(Mat mat, String line1, String line2) {
        double fontScale = Math.max(mat.cols(), mat.rows()) / 900.0;

        // Biar tidak terlalu kecil
        if (fontScale < 0.8) {
            fontScale = 0.8;
        }

        int thicknessMain = Math.max(2, (int) (fontScale * 2));
        int thicknessShadow = Math.max(4, (int) (fontScale * 4));

        int[] baseLine1 = new int[1];
        int[] baseLine2 = new int[1];

        Size textSize1 = Imgproc.getTextSize(
                line1,
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                thicknessMain,
                baseLine1
        );

        Size textSize2 = Imgproc.getTextSize(
                line2,
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                thicknessMain,
                baseLine2
        );

        int leftPadding = 30;
        int rightPadding = 30;
        int topPadding = 20;
        int bottomPadding = 20;
        int lineSpacing = 20;
        int bottomMargin = 30;

        int line2Y = mat.rows() - bottomMargin;
        int line1Y = (int) (line2Y - textSize2.height - lineSpacing);

        int bgLeft = leftPadding - 15;
        int bgTop = (int) (line1Y - textSize1.height - topPadding);
        int bgRight = (int) (Math.max(
                leftPadding + textSize1.width,
                leftPadding + textSize2.width
        ) + rightPadding - 15);
        int bgBottom = line2Y + bottomPadding;

        // Jaga supaya kotak tidak keluar gambar
        bgLeft = Math.max(bgLeft, 0);
        bgTop = Math.max(bgTop, 0);
        bgRight = Math.min(bgRight, mat.cols());
        bgBottom = Math.min(bgBottom, mat.rows());

        // Background hitam transparan
        Imgproc.rectangle(
                mat,
                new Point(bgLeft, bgTop),
                new Point(bgRight, bgBottom),
                new Scalar(0, 0, 0, 120),
                -1
        );

        // Shadow hitam
        Imgproc.putText(
                mat,
                line1,
                new Point(leftPadding, line1Y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                new Scalar(0, 0, 0, 255),
                thicknessShadow
        );

        Imgproc.putText(
                mat,
                line2,
                new Point(leftPadding, line2Y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                new Scalar(0, 0, 0, 255),
                thicknessShadow
        );

        // Teks putih utama
        Imgproc.putText(
                mat,
                line1,
                new Point(leftPadding, line1Y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                new Scalar(255, 255, 255, 255),
                thicknessMain
        );

        Imgproc.putText(
                mat,
                line2,
                new Point(leftPadding, line2Y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                new Scalar(255, 255, 255, 255),
                thicknessMain
        );
    }

    private Bitmap loadBitmapFromUri(Uri uri) {
        try (InputStream inputStream = getContentResolver().openInputStream(uri)) {
            return BitmapFactory.decodeStream(inputStream);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private Bitmap rotateBitmapIfRequired(Bitmap bitmap, Uri imageUri) {
        try (InputStream inputStream = getContentResolver().openInputStream(imageUri)) {
            if (inputStream == null) return bitmap;

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
            e.printStackTrace();
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

            File imageFile = new File(picturesDir, "GAL_" + fileName + ".jpg");

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();

            Toast.makeText(this, "Gambar berhasil disimpan", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show();
        }
    }
}