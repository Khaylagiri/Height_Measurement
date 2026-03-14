package com.example.heightmeasurement;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class GalleryOpenCvActivity extends AppCompatActivity {

    private static final String TAG = "PERSPECTIVE";

    private ImageView imageViewResult;
    private CropOverlayView cropOverlayView;
    private Button btnPerspective, btnSaveGalleryImage;

    private Bitmap currentBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery_opencv);

        imageViewResult = findViewById(R.id.imageViewResult);
        cropOverlayView = findViewById(R.id.cropOverlayView);
        btnPerspective = findViewById(R.id.btnPerspective);
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
        currentBitmap = loadAndRotateBitmap(imageUri);

        if (currentBitmap == null) {
            Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        imageViewResult.setImageBitmap(currentBitmap);

        imageViewResult.post(() -> {
            Point[] detected = detectInitialBoardCorners(currentBitmap);
            if (detected != null) {
                cropOverlayView.setPoints(detected[0], detected[1], detected[2], detected[3]);
            } else {
                int w = cropOverlayView.getWidth();
                int h = cropOverlayView.getHeight();

                cropOverlayView.setPoints(
                        new Point(120, 180),
                        new Point(w - 120, 180),
                        new Point(w - 120, h - 280),
                        new Point(120, h - 280)
                );

                Toast.makeText(this, "Deteksi awal gagal, atur kotak manual", Toast.LENGTH_SHORT).show();
            }
        });

        btnPerspective.setOnClickListener(v -> {
            Log.d(TAG, "Perspective button clicked");

            Point[] overlayPoints = cropOverlayView.getPoints();
            Bitmap warped = warpFromOverlayPoints(currentBitmap, overlayPoints);

            if (warped != null) {
                currentBitmap = warped;
                imageViewResult.setImageBitmap(currentBitmap);

                imageViewResult.post(() -> {
                    int w = cropOverlayView.getWidth();
                    int h = cropOverlayView.getHeight();
                    cropOverlayView.setPoints(
                            new Point(40, 40),
                            new Point(w - 40, 40),
                            new Point(w - 40, h - 180),
                            new Point(40, h - 180)
                    );
                });

                Toast.makeText(this, "Perspective berhasil", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Gagal melakukan perspective", Toast.LENGTH_SHORT).show();
            }
        });

        btnSaveGalleryImage.setOnClickListener(v -> {
            if (currentBitmap != null) {
                saveBitmapToAppFiles(currentBitmap);
            }
        });
    }

    private Bitmap loadAndRotateBitmap(Uri imageUri) {
        try {
            Bitmap originalBitmap = loadBitmapFromUri(imageUri);
            if (originalBitmap == null) return null;
            return rotateBitmapIfRequired(originalBitmap, imageUri);
        } catch (Exception e) {
            Log.e(TAG, "loadAndRotateBitmap error", e);
            return null;
        }
    }

    private Point[] detectInitialBoardCorners(Bitmap bitmap) {
        try {
            Mat srcMat = new Mat();
            Utils.bitmapToMat(bitmap, srcMat);

            Mat gray = new Mat();
            Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY);

            Mat thresh = new Mat();
            Imgproc.threshold(gray, thresh, 180, 255, Imgproc.THRESH_BINARY);

            int width = thresh.cols();
            int height = thresh.rows();

            Mat topRoi = new Mat(thresh, new Rect(0, 0, width, height / 2));
            Mat bottomRoi = new Mat(thresh, new Rect(0, height / 2, width, height - (height / 2)));

            Point[] topCorners = findPaperCorners(topRoi, 0);
            Point[] bottomCorners = findPaperCorners(bottomRoi, height / 2);

            srcMat.release();
            gray.release();
            thresh.release();
            topRoi.release();
            bottomRoi.release();

            if (topCorners == null || bottomCorners == null) {
                return null;
            }

            Point tl = topCorners[0];
            Point tr = topCorners[1];
            Point br = bottomCorners[2];
            Point bl = bottomCorners[3];

            return mapImagePointsToViewPoints(bitmap, tl, tr, br, bl);

        } catch (Exception e) {
            Log.e(TAG, "detectInitialBoardCorners error", e);
            return null;
        }
    }

    private Point[] mapImagePointsToViewPoints(Bitmap bitmap, Point tl, Point tr, Point br, Point bl) {
        int imageW = bitmap.getWidth();
        int imageH = bitmap.getHeight();

        int viewW = imageViewResult.getWidth();
        int viewH = imageViewResult.getHeight();

        float scale = Math.min((float) viewW / imageW, (float) viewH / imageH);
        float displayedW = imageW * scale;
        float displayedH = imageH * scale;

        float offsetX = (viewW - displayedW) / 2f;
        float offsetY = (viewH - displayedH) / 2f;

        Point vTl = new Point(offsetX + tl.x * scale, offsetY + tl.y * scale);
        Point vTr = new Point(offsetX + tr.x * scale, offsetY + tr.y * scale);
        Point vBr = new Point(offsetX + br.x * scale, offsetY + br.y * scale);
        Point vBl = new Point(offsetX + bl.x * scale, offsetY + bl.y * scale);

        return new Point[]{vTl, vTr, vBr, vBl};
    }

    private Bitmap warpFromOverlayPoints(Bitmap bitmap, Point[] overlayPoints) {
        try {
            Mat srcMat = new Mat();
            Utils.bitmapToMat(bitmap, srcMat);

            int imageW = srcMat.cols();
            int imageH = srcMat.rows();

            int viewW = imageViewResult.getWidth();
            int viewH = imageViewResult.getHeight();

            float scale = Math.min((float) viewW / imageW, (float) viewH / imageH);
            float displayedW = imageW * scale;
            float displayedH = imageH * scale;

            float offsetX = (viewW - displayedW) / 2f;
            float offsetY = (viewH - displayedH) / 2f;

            Point tl = new Point((overlayPoints[0].x - offsetX) / scale, (overlayPoints[0].y - offsetY) / scale);
            Point tr = new Point((overlayPoints[1].x - offsetX) / scale, (overlayPoints[1].y - offsetY) / scale);
            Point br = new Point((overlayPoints[2].x - offsetX) / scale, (overlayPoints[2].y - offsetY) / scale);
            Point bl = new Point((overlayPoints[3].x - offsetX) / scale, (overlayPoints[3].y - offsetY) / scale);

            tl.x = clamp(tl.x, 0, imageW - 1);
            tl.y = clamp(tl.y, 0, imageH - 1);
            tr.x = clamp(tr.x, 0, imageW - 1);
            tr.y = clamp(tr.y, 0, imageH - 1);
            br.x = clamp(br.x, 0, imageW - 1);
            br.y = clamp(br.y, 0, imageH - 1);
            bl.x = clamp(bl.x, 0, imageW - 1);
            bl.y = clamp(bl.y, 0, imageH - 1);

            double widthTop = distance(tl, tr);
            double widthBottom = distance(bl, br);
            double maxWidth = Math.max(widthTop, widthBottom);

            double heightLeft = distance(tl, bl);
            double heightRight = distance(tr, br);
            double maxHeight = Math.max(heightLeft, heightRight);

            Log.d(TAG, "Warp size: " + maxWidth + " x " + maxHeight);

            MatOfPoint2f srcPoints = new MatOfPoint2f(tl, tr, br, bl);
            MatOfPoint2f dstPoints = new MatOfPoint2f(
                    new Point(0, 0),
                    new Point(maxWidth - 1, 0),
                    new Point(maxWidth - 1, maxHeight - 1),
                    new Point(0, maxHeight - 1)
            );

            Mat matrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Mat warped = new Mat();

            Imgproc.warpPerspective(srcMat, warped, matrix, new Size(maxWidth, maxHeight));

            Bitmap result = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(warped, result);

            srcMat.release();
            srcPoints.release();
            dstPoints.release();
            matrix.release();
            warped.release();

            return result;

        } catch (Exception e) {
            Log.e(TAG, "warpFromOverlayPoints error", e);
            return null;
        }
    }

    private Point[] findPaperCorners(Mat roiMat, int yOffset) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(
                roiMat.clone(),
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        hierarchy.release();

        if (contours.isEmpty()) {
            return null;
        }

        double bestArea = 0;
        Point[] bestCorners = null;

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < roiMat.cols() * roiMat.rows() * 0.10) {
                contour.release();
                continue;
            }

            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double peri = Imgproc.arcLength(contour2f, true);

            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true);

            Point[] pts = approx.toArray();

            if (pts.length == 4 && area > bestArea) {
                for (Point p : pts) {
                    p.y += yOffset;
                }
                bestCorners = orderPoints(pts);
                bestArea = area;
            }

            contour2f.release();
            approx.release();
            contour.release();
        }

        return bestCorners;
    }

    private Point[] orderPoints(Point[] pts) {
        Point[] ordered = new Point[4];

        double minSum = Double.MAX_VALUE;
        double maxSum = -Double.MAX_VALUE;
        double minDiff = Double.MAX_VALUE;
        double maxDiff = -Double.MAX_VALUE;

        for (Point p : pts) {
            double sum = p.x + p.y;
            double diff = p.x - p.y;

            if (sum < minSum) {
                minSum = sum;
                ordered[0] = p;
            }
            if (sum > maxSum) {
                maxSum = sum;
                ordered[2] = p;
            }
            if (diff > maxDiff) {
                maxDiff = diff;
                ordered[1] = p;
            }
            if (diff < minDiff) {
                minDiff = diff;
                ordered[3] = p;
            }
        }

        return ordered;
    }

    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    private double clamp(double val, double min, double max) {
        return Math.max(min, Math.min(max, val));
    }

    private Bitmap loadBitmapFromUri(Uri uri) {
        try (InputStream inputStream = getContentResolver().openInputStream(uri)) {
            return BitmapFactory.decodeStream(inputStream);
        } catch (Exception e) {
            Log.e(TAG, "loadBitmapFromUri error", e);
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
            Log.e(TAG, "rotateBitmapIfRequired error", e);
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
            Log.e(TAG, "saveBitmapToAppFiles error", e);
            Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show();
        }
    }
}