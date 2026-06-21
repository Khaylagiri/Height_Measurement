package com.example.heightmeasurement;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.exifinterface.media.ExifInterface;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.DetectorParameters;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class PerspectiveCorrectionActivity extends AppCompatActivity {

    private static final String TAG = "PerspectiveCorrection";

    private static final int MARKER_DICT = Objdetect.DICT_6X6_1000;

    private static final double PX_PER_CELL = 120.0;

    private static final double BOARD_COLS = 11.0;
    private static final double TOP_BOARD_ROWS = 8.0;
    private static final double BOTTOM_BOARD_OFFSET_Y = 11.35;
    private static final double BOTTOM_BOARD_ROWS = 8.0;

    private static final double LEFT_RIGHT_MARGIN_CELLS = 0.000;
    private static final double TOP_MARGIN_CELLS = 0.35;
    private static final double BOTTOM_EXTRA_CELLS = 5.60;

    private static final double BOARD_REAL_HEIGHT_CM = 200.0;
    private static final double BOARD_HEIGHT_CELLS = 19.35; // 8.0 (top) + 3.35 (gap) + 8.0 (bottom)

    private static final double MARKER_SIZE_CELLS = 0.74;

    /*
     * Ukur dari pusat optik kamera ke bidang papan dan dari subjek ke papan.
     * Nilai ini harus sama dengan kondisi fisik saat foto diambil.
     * Pada konfigurasi saat ini subjek diasumsikan 35 cm di depan papan.
     * Ukur langsung dan ubah angka ini bila posisi tumit berbeda.
     */
    private static final double CAMERA_DISTANCE_CM = 350.0;
    private static final double SUBJECT_DISTANCE_FROM_BOARD_CM = 35.0;

    private static final double MAX_MEAN_REPROJECTION_ERROR_PX = 6.0;
    private static final double MAX_POINT_REPROJECTION_ERROR_PX = 15.0;

    private static final int OUTPUT_WIDTH_PX =
            (int) Math.round((BOARD_COLS + LEFT_RIGHT_MARGIN_CELLS * 2.0) * PX_PER_CELL);

    private static final int OUTPUT_HEIGHT_PX =
            (int) Math.round(
                    (TOP_MARGIN_CELLS
                            + BOTTOM_BOARD_OFFSET_Y
                            + BOTTOM_BOARD_ROWS
                            + BOTTOM_EXTRA_CELLS) * PX_PER_CELL
            );

    private ImageView imageViewResult;
    private Button btnPerspective;
    private Button btnSaveGalleryImage;

    private Bitmap originalBitmap;
    private Bitmap currentBitmap;

    private boolean perspectiveSuccess = false;
    private String lastPerspectiveError = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_perspective_correction);

        imageViewResult = findViewById(R.id.imageViewResult);
        btnPerspective = findViewById(R.id.btnPerspective);
        btnSaveGalleryImage = findViewById(R.id.btnSaveGalleryImage);

        imageViewResult.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageViewResult.setAdjustViewBounds(true);

        /*
         * Tombol awal namanya Perspective Correction.
         * Tombol ini nanti hilang setelah perspective berhasil.
         */
        btnPerspective.setText("Perspective Correction");
        btnPerspective.setVisibility(View.VISIBLE);

        /*
         * Tombol yang tadinya Simpan sekarang jadi Next.
         * Tombol Next disembunyikan dulu.
         * Nanti muncul setelah perspective berhasil.
         */
        btnSaveGalleryImage.setText("Next");
        btnSaveGalleryImage.setVisibility(View.GONE);

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
        originalBitmap = loadAndRotateBitmap(imageUri);

        if (originalBitmap == null) {
            Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        currentBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        imageViewResult.setImageBitmap(currentBitmap);

        btnPerspective.setOnClickListener(v -> runPerspectiveOnly());

        /*
         * Klik Next:
         * hasil perspective dikirim ke MediaPipeActivity.
         */
        btnSaveGalleryImage.setOnClickListener(v -> openMediaPipeWithPerspectiveResult());
    }

    private void runPerspectiveOnly() {
        try {
            btnPerspective.setEnabled(false);
            btnSaveGalleryImage.setVisibility(View.GONE);
            perspectiveSuccess = false;
            lastPerspectiveError = "";

            Bitmap result = autoPerspectiveCamScannerStyle(originalBitmap);

            if (result == null) {
                String errorMessage = lastPerspectiveError == null || lastPerspectiveError.trim().isEmpty()
                        ? "Perspective gagal. Marker acuan papan belum cukup untuk menghitung bidang."
                        : "Perspective gagal: " + lastPerspectiveError;

                Toast.makeText(
                        this,
                        errorMessage,
                        Toast.LENGTH_LONG
                ).show();

                btnPerspective.setVisibility(View.VISIBLE);
                return;
            }

            currentBitmap = result;
            imageViewResult.setImageBitmap(currentBitmap);

            perspectiveSuccess = true;

            /*
             * Setelah perspective berhasil:
             * tombol Perspective Correction dihilangkan.
             */
            btnPerspective.setVisibility(View.GONE);

            /*
             * Setelah perspective berhasil:
             * tombol Next dimunculkan.
             */
            btnSaveGalleryImage.setVisibility(View.VISIBLE);

            Toast.makeText(
                    this,
                    "Perspective berhasil",
                    Toast.LENGTH_SHORT
            ).show();

        } catch (Exception e) {
            Log.e(TAG, "runPerspectiveOnly error", e);
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();

            btnPerspective.setVisibility(View.VISIBLE);

        } finally {
            btnPerspective.setEnabled(true);
        }
    }

    private void openMediaPipeWithPerspectiveResult() {
        if (!perspectiveSuccess || currentBitmap == null) {
            Toast.makeText(this, "Jalankan perspective sampai berhasil dulu", Toast.LENGTH_LONG).show();
            return;
        }

        if (currentBitmap.getWidth() != OUTPUT_WIDTH_PX
                || currentBitmap.getHeight() != OUTPUT_HEIGHT_PX) {
            Toast.makeText(
                    this,
                    "Ukuran hasil perspective tidak valid: "
                            + currentBitmap.getWidth() + "x" + currentBitmap.getHeight(),
                    Toast.LENGTH_LONG
            ).show();
            return;
        }

        String imagePath = saveBitmapToCacheForMediaPipe(currentBitmap);

        if (imagePath == null) {
            Toast.makeText(this, "Gagal menyiapkan gambar untuk MediaPipe", Toast.LENGTH_LONG).show();
            return;
        }

        Intent intent = new Intent(PerspectiveCorrectionActivity.this, MediaPipeActivity.class);
        intent.putExtra("image_path", imagePath);

        // Semua nilai skala berasal dari bitmap warped yang sama.
        double boardTopY = TOP_MARGIN_CELLS * PX_PER_CELL;
        double boardPixelHeight = BOARD_HEIGHT_CELLS * PX_PER_CELL;
        double boardBottomY = boardTopY + boardPixelHeight;
        double cmPerPixel = BOARD_REAL_HEIGHT_CM / boardPixelHeight;

        intent.putExtra("board_top_y", boardTopY);
        intent.putExtra("board_bottom_y", boardBottomY);
        intent.putExtra("board_pixel_height", boardPixelHeight);
        intent.putExtra("cm_per_pixel", cmPerPixel);
        intent.putExtra("board_real_height_cm", BOARD_REAL_HEIGHT_CM);
        intent.putExtra("output_width_px", OUTPUT_WIDTH_PX);
        intent.putExtra("output_height_px", OUTPUT_HEIGHT_PX);
        intent.putExtra("camera_distance_cm", CAMERA_DISTANCE_CM);
        intent.putExtra("subject_distance_from_board_cm", SUBJECT_DISTANCE_FROM_BOARD_CM);

        /*
         * false artinya:
         * gambar hasil perspective langsung tampil di MediaPipe,
         * tapi MediaPipe belum otomatis proses.
         * Jadi user tinggal klik tombol Proses MediaPipe.
         */
        intent.putExtra("auto_process", false);

        startActivity(intent);
    }

    private String saveBitmapToCacheForMediaPipe(Bitmap bitmap) {
        FileOutputStream fos = null;

        try {
            File cacheDir = new File(getCacheDir(), "perspective_result");

            if (!cacheDir.exists()) {
                cacheDir.mkdirs();
            }

            String fileName = new SimpleDateFormat(
                    "yyyyMMdd_HHmmss",
                    Locale.getDefault()
            ).format(new Date());

            File imageFile = new File(
                    cacheDir,
                    "PERSPECTIVE_TO_MEDIAPIPE_" + fileName + ".png"
            );

            fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();

            return imageFile.getAbsolutePath();

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToCacheForMediaPipe error", e);
            return null;

        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    private Bitmap autoPerspectiveCamScannerStyle(Bitmap bitmap) {
        Mat srcMat = new Mat();
        Mat gray = new Mat();
        Mat warped = new Mat();

        Mat ids = new Mat();
        List<Mat> corners = new ArrayList<>();
        List<Mat> rejected = new ArrayList<>();

        MatOfPoint2f imagePointsMat = null;
        MatOfPoint2f boardPointsMat = null;
        Mat homography = new Mat();

        try {
            Utils.bitmapToMat(bitmap, srcMat);

            if (srcMat.channels() == 4) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (srcMat.channels() == 3) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = srcMat.clone();
            }

            /*
             * CLAHE/equalize tidak selalu diperlukan. equalizeHist tetap dipakai
             * karena gambar masukan berupa citra 8-bit grayscale.
             */
            Imgproc.equalizeHist(gray, gray);

            Dictionary dictionary = Objdetect.getPredefinedDictionary(MARKER_DICT);
            DetectorParameters parameters = createArucoDetectorParameters();
            ArucoDetector detector = new ArucoDetector(dictionary, parameters);

            detector.detectMarkers(gray, corners, ids, rejected);

            if (ids.empty() || corners.isEmpty()) {
                return failPerspective("tidak ada marker ArUco yang berhasil dibaca");
            }

            /*
             * PENTING:
             * Homography dihitung dari TITIK TENGAH marker, bukan keempat sudutnya.
             * Ini menghindari kegagalan akibat ukuran cetak marker, white border,
             * atau urutan sudut marker yang berbeda.
             */
            List<Point> imageCenters = new ArrayList<>();
            List<Point> boardCenters = new ArrayList<>();

            int usedMarkers = 0;
            int usedTopMarkers = 0;
            int usedBottomMarkers = 0;

            double minWorldX = Double.MAX_VALUE;
            double maxWorldX = -Double.MAX_VALUE;
            double minWorldY = Double.MAX_VALUE;
            double maxWorldY = -Double.MAX_VALUE;

            for (int i = 0; i < ids.rows(); i++) {
                int id = (int) ids.get(i, 0)[0];

                Point boardCenter = getMarkerWorldCenter(id);
                if (boardCenter == null) {
                    // Marker terdeteksi, tetapi ID-nya tidak terdaftar pada layout papan.
                    continue;
                }

                Point imageCenter = getMarkerImageCenter(corners.get(i));
                if (imageCenter == null) {
                    continue;
                }

                imageCenters.add(imageCenter);
                boardCenters.add(boardCenter);
                usedMarkers++;

                if (getTopBoardRowCol(id) != null) {
                    usedTopMarkers++;
                } else if (getBottomBoardRowCol(id) != null) {
                    usedBottomMarkers++;
                }

                minWorldX = Math.min(minWorldX, boardCenter.x);
                maxWorldX = Math.max(maxWorldX, boardCenter.x);
                minWorldY = Math.min(minWorldY, boardCenter.y);
                maxWorldY = Math.max(maxWorldY, boardCenter.y);
            }

            Log.d(
                    TAG,
                    "Detected=" + ids.rows()
                            + ", registered=" + usedMarkers
                            + ", top=" + usedTopMarkers
                            + ", bottom=" + usedBottomMarkers
            );

            if (usedMarkers < 4) {
                return failPerspective(
                        "hanya " + usedMarkers
                                + " marker yang cocok dengan daftar ID papan; minimal 4 diperlukan"
                );
            }

            /*
             * Badan boleh menutup sebagian marker.
             * Cukup ada marker terdaftar pada papan atas dan papan bawah.
             */
            if (usedTopMarkers < 1 || usedBottomMarkers < 1) {
                return failPerspective(
                        "marker yang terbaca harus mencakup papan atas dan papan bawah"
                );
            }

            double horizontalCoverage = maxWorldX - minWorldX;
            double verticalCoverage = maxWorldY - minWorldY;
            double boardWidthPx = BOARD_COLS * PX_PER_CELL;
            double boardHeightPx = BOARD_HEIGHT_CELLS * PX_PER_CELL;

            Log.d(
                    TAG,
                    String.format(
                            Locale.US,
                            "Coverage horizontal=%.1f px, vertical=%.1f px",
                            horizontalCoverage,
                            verticalCoverage
                    )
            );

            /*
             * Threshold dibuat moderat. Marker tidak harus berada tepat di empat sudut,
             * tetapi tidak boleh semuanya berkumpul pada area sempit.
             */
            if (horizontalCoverage < boardWidthPx * 0.25) {
                return failPerspective("sebaran marker kiri-kanan terlalu sempit");
            }

            if (verticalCoverage < boardHeightPx * 0.35) {
                return failPerspective("sebaran marker atas-bawah terlalu sempit");
            }

            imagePointsMat = new MatOfPoint2f(
                    imageCenters.toArray(new Point[0])
            );
            boardPointsMat = new MatOfPoint2f(
                    boardCenters.toArray(new Point[0])
            );

            homography = Calib3d.findHomography(
                    imagePointsMat,
                    boardPointsMat,
                    Calib3d.RANSAC,
                    10.0
            );

            if (homography.empty()) {
                return failPerspective("matriks homography tidak dapat dihitung");
            }

            /*
             * Reprojection error hanya dicatat sebagai diagnostik.
             * Jangan langsung menggagalkan perspective karena papan terdiri dari
             * dua lembar dan pemasangan fisiknya bisa memiliki sedikit deviasi.
             */
            ReprojectionError reprojectionError = calculateReprojectionError(
                    imagePointsMat,
                    boardPointsMat,
                    homography
            );

            Log.d(
                    TAG,
                    String.format(
                            Locale.US,
                            "Center reprojection error mean=%.2f px, max=%.2f px",
                            reprojectionError.meanPx,
                            reprojectionError.maxPx
                    )
            );

            Imgproc.warpPerspective(
                    srcMat,
                    warped,
                    homography,
                    new Size(OUTPUT_WIDTH_PX, OUTPUT_HEIGHT_PX),
                    Imgproc.INTER_LINEAR
            );

            if (warped.empty()
                    || warped.cols() != OUTPUT_WIDTH_PX
                    || warped.rows() != OUTPUT_HEIGHT_PX) {
                return failPerspective("ukuran hasil warp tidak sesuai");
            }

            // Jangan shift, crop, atau resize lagi.
            Bitmap result = Bitmap.createBitmap(
                    OUTPUT_WIDTH_PX,
                    OUTPUT_HEIGHT_PX,
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(warped, result);
            return result;

        } catch (Exception e) {
            Log.e(TAG, "autoPerspectiveCamScannerStyle error", e);
            return failPerspective(
                    e.getMessage() == null
                            ? "terjadi kesalahan saat memproses perspective"
                            : e.getMessage()
            );

        } finally {
            for (Mat c : corners) {
                c.release();
            }

            for (Mat r : rejected) {
                r.release();
            }

            if (imagePointsMat != null) {
                imagePointsMat.release();
            }

            if (boardPointsMat != null) {
                boardPointsMat.release();
            }

            ids.release();
            homography.release();
            srcMat.release();
            gray.release();
            warped.release();
        }
    }

    private Bitmap failPerspective(String reason) {
        lastPerspectiveError = reason;
        Log.w(TAG, "Perspective rejected: " + reason);
        return null;
    }

    private Point getMarkerImageCenter(Mat markerMat) {
        try {
            double sumX = 0.0;
            double sumY = 0.0;
            int validPointCount = 0;

            for (int i = 0; i < 4; i++) {
                double[] xy = markerMat.get(0, i);

                if (xy == null || xy.length < 2) {
                    xy = markerMat.get(i, 0);
                }

                if (xy == null || xy.length < 2) {
                    continue;
                }

                sumX += xy[0];
                sumY += xy[1];
                validPointCount++;
            }

            if (validPointCount != 4) {
                return null;
            }

            return new Point(
                    sumX / validPointCount,
                    sumY / validPointCount
            );

        } catch (Exception e) {
            Log.e(TAG, "getMarkerImageCenter error", e);
            return null;
        }
    }

    private ReprojectionError calculateReprojectionError(
            MatOfPoint2f imagePoints,
            MatOfPoint2f expectedBoardPoints,
            Mat homography
    ) {
        MatOfPoint2f projected = new MatOfPoint2f();

        try {
            Core.perspectiveTransform(imagePoints, projected, homography);

            Point[] actual = projected.toArray();
            Point[] expected = expectedBoardPoints.toArray();

            if (actual.length == 0 || actual.length != expected.length) {
                return new ReprojectionError(false, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
            }

            double total = 0.0;
            double max = 0.0;

            for (int i = 0; i < actual.length; i++) {
                double dx = actual[i].x - expected[i].x;
                double dy = actual[i].y - expected[i].y;
                double error = Math.sqrt(dx * dx + dy * dy);
                total += error;
                max = Math.max(max, error);
            }

            return new ReprojectionError(true, total / actual.length, max);

        } finally {
            projected.release();
        }
    }

    private Point[] getMarkerImageCornersOrdered(Mat markerMat) {
        try {
            Point[] raw = new Point[4];

            for (int i = 0; i < 4; i++) {
                double[] xy = markerMat.get(0, i);

                if (xy == null || xy.length < 2) {
                    xy = markerMat.get(i, 0);
                }

                if (xy == null || xy.length < 2) {
                    return null;
                }

                raw[i] = new Point(xy[0], xy[1]);
            }

            return orderCornersTopLeftTopRightBottomRightBottomLeft(raw);

        } catch (Exception e) {
            Log.e(TAG, "getMarkerImageCornersOrdered error", e);
            return null;
        }
    }

    private Point[] orderCornersTopLeftTopRightBottomRightBottomLeft(Point[] pts) {
        if (pts == null || pts.length != 4) {
            return null;
        }

        Point tl = null;
        Point tr = null;
        Point br = null;
        Point bl = null;

        double minSum = Double.MAX_VALUE;
        double maxSum = -Double.MAX_VALUE;
        double minDiff = Double.MAX_VALUE;
        double maxDiff = -Double.MAX_VALUE;

        for (Point p : pts) {
            double sum = p.x + p.y;
            double diff = p.y - p.x;

            if (sum < minSum) {
                minSum = sum;
                tl = p;
            }

            if (sum > maxSum) {
                maxSum = sum;
                br = p;
            }

            if (diff < minDiff) {
                minDiff = diff;
                tr = p;
            }

            if (diff > maxDiff) {
                maxDiff = diff;
                bl = p;
            }
        }

        if (tl == null || tr == null || br == null || bl == null) {
            return null;
        }

        return new Point[]{tl, tr, br, bl};
    }

    private Point[] getMarkerWorldCorners(int id) {
        Point center = getMarkerWorldCenter(id);

        if (center == null) {
            return null;
        }

        double markerSizePx = MARKER_SIZE_CELLS * PX_PER_CELL;
        double half = markerSizePx / 2.0;

        return new Point[]{
                new Point(center.x - half, center.y - half),
                new Point(center.x + half, center.y - half),
                new Point(center.x + half, center.y + half),
                new Point(center.x - half, center.y + half)
        };
    }

    private Point getMarkerWorldCenter(int id) {
        Integer[] rcTop = getTopBoardRowCol(id);

        if (rcTop != null) {
            double x = (LEFT_RIGHT_MARGIN_CELLS + rcTop[1] + 0.5) * PX_PER_CELL;
            double y = (TOP_MARGIN_CELLS + rcTop[0] + 0.5) * PX_PER_CELL;
            return new Point(x, y);
        }

        Integer[] rcBottom = getBottomBoardRowCol(id);

        if (rcBottom != null) {
            double x = (LEFT_RIGHT_MARGIN_CELLS + rcBottom[1] + 0.5) * PX_PER_CELL;
            double y = (TOP_MARGIN_CELLS + BOTTOM_BOARD_OFFSET_Y + rcBottom[0] + 0.5) * PX_PER_CELL;
            return new Point(x, y);
        }

        return null;
    }

    private Integer[] getTopBoardRowCol(int id) {
        switch (id) {
            case 103: return new Integer[]{0, 0};
            case 115: return new Integer[]{0, 2};
            case 123: return new Integer[]{0, 4};
            case 131: return new Integer[]{0, 6};
            case 139: return new Integer[]{0, 8};
            case 143: return new Integer[]{0, 10};

            case 107: return new Integer[]{1, 1};
            case 111: return new Integer[]{1, 3};
            case 119: return new Integer[]{1, 5};
            case 127: return new Integer[]{1, 7};
            case 135: return new Integer[]{1, 9};

            case 102: return new Integer[]{2, 0};
            case 114: return new Integer[]{2, 2};
            case 122: return new Integer[]{2, 4};
            case 130: return new Integer[]{2, 6};
            case 138: return new Integer[]{2, 8};
            case 142: return new Integer[]{2, 10};

            case 106: return new Integer[]{3, 1};
            case 110: return new Integer[]{3, 3};
            case 118: return new Integer[]{3, 5};
            case 126: return new Integer[]{3, 7};
            case 134: return new Integer[]{3, 9};

            case 101: return new Integer[]{4, 0};
            case 113: return new Integer[]{4, 2};
            case 121: return new Integer[]{4, 4};
            case 129: return new Integer[]{4, 6};
            case 137: return new Integer[]{4, 8};
            case 141: return new Integer[]{4, 10};

            case 105: return new Integer[]{5, 1};
            case 109: return new Integer[]{5, 3};
            case 117: return new Integer[]{5, 5};
            case 125: return new Integer[]{5, 7};
            case 133: return new Integer[]{5, 9};

            case 100: return new Integer[]{6, 0};
            case 112: return new Integer[]{6, 2};
            case 120: return new Integer[]{6, 4};
            case 128: return new Integer[]{6, 6};
            case 136: return new Integer[]{6, 8};
            case 140: return new Integer[]{6, 10};

            case 104: return new Integer[]{7, 1};
            case 108: return new Integer[]{7, 3};
            case 116: return new Integer[]{7, 5};
            case 124: return new Integer[]{7, 7};
            case 132: return new Integer[]{7, 9};

            default: return null;
        }
    }

    private Integer[] getBottomBoardRowCol(int id) {
        switch (id) {
            case 403: return new Integer[]{0, 0};
            case 435: return new Integer[]{0, 10};

            case 407: return new Integer[]{1, 1};
            case 431: return new Integer[]{1, 9};
            case 439: return new Integer[]{1, 10};

            case 402: return new Integer[]{2, 0};
            case 410: return new Integer[]{2, 2};
            case 430: return new Integer[]{2, 8};
            case 434: return new Integer[]{2, 10};

            case 406: return new Integer[]{3, 1};
            case 438: return new Integer[]{3, 10};

            case 401: return new Integer[]{4, 0};
            case 409: return new Integer[]{4, 2};
            case 433: return new Integer[]{4, 10};

            case 400: return new Integer[]{5, 0};
            case 405: return new Integer[]{5, 2};
            case 429: return new Integer[]{5, 8};
            case 437: return new Integer[]{5, 10};

            case 404: return new Integer[]{6, 2};
            case 432: return new Integer[]{6, 8};

            case 408: return new Integer[]{7, 0};
            case 428: return new Integer[]{7, 8};
            case 436: return new Integer[]{7, 10};

            default: return null;
        }
    }

    private DetectorParameters createArucoDetectorParameters() {
        DetectorParameters parameters = new DetectorParameters();

        parameters.set_adaptiveThreshWinSizeMin(3);
        parameters.set_adaptiveThreshWinSizeMax(53);
        parameters.set_adaptiveThreshWinSizeStep(4);

        parameters.set_minMarkerPerimeterRate(0.004);
        parameters.set_maxMarkerPerimeterRate(4.0);

        parameters.set_polygonalApproxAccuracyRate(0.06);
        parameters.set_minCornerDistanceRate(0.005);
        parameters.set_minDistanceToBorder(1);

        parameters.set_cornerRefinementMethod(1);
        parameters.set_cornerRefinementWinSize(7);
        parameters.set_cornerRefinementMaxIterations(80);
        parameters.set_cornerRefinementMinAccuracy(0.01);

        return parameters;
    }

    private Bitmap loadAndRotateBitmap(Uri imageUri) {
        try {
            Bitmap bitmap = loadBitmapFromUri(imageUri);

            if (bitmap == null) {
                return null;
            }

            return rotateBitmapIfRequired(bitmap, imageUri);

        } catch (Exception e) {
            Log.e(TAG, "loadAndRotateBitmap error", e);
            return null;
        }
    }

    private Bitmap loadBitmapFromUri(Uri uri) {
        try (InputStream inputStream = getContentResolver().openInputStream(uri)) {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;

            return BitmapFactory.decodeStream(inputStream, null, options);

        } catch (Exception e) {
            Log.e(TAG, "loadBitmapFromUri error", e);
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
            Log.e(TAG, "rotateBitmapIfRequired error", e);
            return bitmap;
        }
    }
    private static class ReprojectionError {
        final boolean valid;
        final double meanPx;
        final double maxPx;

        ReprojectionError(boolean valid, double meanPx, double maxPx) {
            this.valid = valid;
            this.meanPx = meanPx;
            this.maxPx = maxPx;
        }
    }

}
