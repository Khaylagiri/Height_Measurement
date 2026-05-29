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
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
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
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class GalleryOpenCvActivity extends AppCompatActivity {

    private static final String TAG = "HEIGHT_MEASURE";
    private static final int MARKER_DICT = Objdetect.DICT_6X6_1000;
    private static final int MAX_WARP_W = 2200;
    private static final int MAX_WARP_H = 3600;

    private static final double CELL_PX = 100.0;
    private static final int OUTPUT_WIDTH_PX = 11 * 100; // 1100

    /*
     * Dinaikkan agar kaki tidak terpotong.
     * Papan atas 8 baris = 800 px
     * Gap antara papan ~= BOTTOM_BOARD_OFFSET_Y_PX
     * Papan bawah 8 baris = 800 px
     * Area kaki di bawah papan bawah ~= 600 px tambahan
     *
     * Total: sekitar 3200-3400 px cukup.
     * Kalau masih kepotong, naikkan lagi.
     */
    private static final int OUTPUT_HEIGHT_PX = 3400;

    /*
     * Offset Y papan bawah di koordinat output perspective.
     * Ini jarak dari baris pertama papan atas ke baris pertama papan bawah.
     *
     * WAJIB disesuaikan dengan jarak fisik papan atas ke papan bawah.
     *
     * Cara menghitung:
     * - Ukur jarak dari tepi atas papan atas ke tepi atas papan bawah (cm).
     * - Papan atas tinggi fisik = 8 baris x ukuran_cell_cm.
     * - Kalau ukuran cell = 1 cm, papan atas = 8 cm.
     * - Gap antara papan misalnya 4.5 cm.
     * - Total offset = (8 + 4.5) / 1 * 100 = 1250 px.
     *
     * Sesuaikan sesuai papan asli kamu.
     */
    private static final double BOTTOM_BOARD_OFFSET_Y_PX = 1250.0;

    private ImageView imageViewResult;
    private Button btnPerspective;
    private Button btnSaveGalleryImage;

    private Bitmap originalBitmap;
    private Bitmap currentBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery_opencv);

        Log.d(TAG, "=== onCreate ===");

        imageViewResult = findViewById(R.id.imageViewResult);
        btnPerspective = findViewById(R.id.btnPerspective);
        btnSaveGalleryImage = findViewById(R.id.btnSaveGalleryImage);

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initLocal() gagal");
            Toast.makeText(this, "OpenCV gagal diinisialisasi", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        String uriString = getIntent().getStringExtra("image_uri");
        Log.d(TAG, "image_uri = " + uriString);

        if (uriString == null) {
            Toast.makeText(this, "Gambar tidak ditemukan", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        Uri imageUri = Uri.parse(uriString);
        originalBitmap = loadAndRotateBitmap(imageUri);

        if (originalBitmap == null) {
            Log.e(TAG, "originalBitmap null");
            Toast.makeText(this, "Gagal membuka gambar", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        Log.d(TAG, "originalBitmap size = "
                + originalBitmap.getWidth() + "x" + originalBitmap.getHeight());

        currentBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
        imageViewResult.setImageBitmap(currentBitmap);

        btnPerspective.setOnClickListener(v -> runPerspectiveOnly());

        btnSaveGalleryImage.setOnClickListener(v -> {
            if (currentBitmap != null) {
                saveBitmapToAppFiles(currentBitmap);
            }
        });
    }

    // ==================== PERSPECTIVE ONLY ====================

    private void runPerspectiveOnly() {
        Log.d(TAG, "=== runPerspectiveOnly START ===");

        try {
            btnPerspective.setEnabled(false);

            Bitmap warped = autoPerspectiveFromAllBoardHomography(originalBitmap);

            if (warped == null) {
                Log.e(TAG, "Homography failed, trying fallback TwoBoards");
                warped = autoPerspectiveFromTwoBoards(originalBitmap);
            }

            if (warped == null) {
                Log.e(TAG, "All perspective methods failed");
                Toast.makeText(this,
                        "Gagal auto perspective. Pastikan marker terlihat jelas.",
                        Toast.LENGTH_LONG).show();
                return;
            }

            Log.d(TAG, "warped size = " + warped.getWidth() + "x" + warped.getHeight());

            currentBitmap = warped;
            imageViewResult.setImageBitmap(currentBitmap);

            Toast.makeText(this, "Perspective berhasil", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "runPerspectiveOnly error", e);
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        } finally {
            btnPerspective.setEnabled(true);
            Log.d(TAG, "=== runPerspectiveOnly END ===");
        }
    }

    // ==================== HOMOGRAPHY ATAS + BAWAH ====================

    private Bitmap autoPerspectiveFromAllBoardHomography(Bitmap bitmap) {
        Log.d(TAG, "=== autoPerspectiveFromAllBoardHomography START ===");

        Mat srcMat = new Mat();
        Mat gray = new Mat();
        Mat warped = new Mat();
        Mat ids = new Mat();
        List<Mat> corners = new ArrayList<>();
        List<Mat> rejected = new ArrayList<>();
        MatOfPoint2f srcPoints = null;
        MatOfPoint2f dstPoints = null;
        Mat H = new Mat();

        try {
            Utils.bitmapToMat(bitmap, srcMat);

            if (srcMat.channels() == 4) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (srcMat.channels() == 3) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = srcMat.clone();
            }

            Imgproc.equalizeHist(gray, gray);

            Dictionary dictionary = Objdetect.getPredefinedDictionary(MARKER_DICT);
            DetectorParameters parameters = createArucoDetectorParameters();
            ArucoDetector detector = new ArucoDetector(dictionary, parameters);
            detector.detectMarkers(gray, corners, ids, rejected);

            if (ids.empty()) {
                Log.e(TAG, "No ArUco detected");
                return null;
            }

            List<Point> imagePoints = new ArrayList<>();
            List<Point> worldPoints = new ArrayList<>();

            int topCount = 0;
            int bottomCount = 0;

            for (int i = 0; i < ids.rows(); i++) {
                int id = (int) ids.get(i, 0)[0];

                Point worldCenter = lookupBoardCenterPx(id);
                if (worldCenter == null) {
                    Log.d(TAG, "Skipping unknown board ID: " + id);
                    continue;
                }

                Mat c = corners.get(i);
                double[] xy0 = c.get(0, 0);
                double[] xy1 = c.get(0, 1);
                double[] xy2 = c.get(0, 2);
                double[] xy3 = c.get(0, 3);

                double cx = (xy0[0] + xy1[0] + xy2[0] + xy3[0]) / 4.0;
                double cy = (xy0[1] + xy1[1] + xy2[1] + xy3[1]) / 4.0;

                imagePoints.add(new Point(cx, cy));
                worldPoints.add(worldCenter);

                if (id >= 100 && id < 200) topCount++;
                if (id >= 400 && id < 500) bottomCount++;

                Log.d(TAG, "H point id=" + id
                        + " image=(" + cx + "," + cy + ")"
                        + " world=(" + worldCenter.x + "," + worldCenter.y + ")");
            }

            Log.d(TAG, "Homography points total=" + imagePoints.size()
                    + " top=" + topCount + " bottom=" + bottomCount);

            if (imagePoints.size() < 4) {
                Log.e(TAG, "Not enough markers for homography: " + imagePoints.size());
                return null;
            }

            srcPoints = new MatOfPoint2f(imagePoints.toArray(new Point[0]));
            dstPoints = new MatOfPoint2f(worldPoints.toArray(new Point[0]));

            H = Calib3d.findHomography(srcPoints, dstPoints, Calib3d.RANSAC, 5.0);

            if (H.empty()) {
                Log.e(TAG, "findHomography failed");
                return null;
            }

            Imgproc.warpPerspective(
                    srcMat,
                    warped,
                    H,
                    new Size(OUTPUT_WIDTH_PX, OUTPUT_HEIGHT_PX)
            );

            Bitmap result = Bitmap.createBitmap(
                    warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888
            );
            Utils.matToBitmap(warped, result);

            Log.d(TAG, "=== autoPerspectiveFromAllBoardHomography END ===");
            return result;

        } catch (Exception e) {
            Log.e(TAG, "autoPerspectiveFromAllBoardHomography error", e);
            return null;
        } finally {
            for (Mat c : corners) c.release();
            for (Mat r : rejected) r.release();
            if (srcPoints != null) srcPoints.release();
            if (dstPoints != null) dstPoints.release();
            H.release();
            ids.release();
            gray.release();
            srcMat.release();
            warped.release();
        }
    }

    // ==================== LOOKUP KOORDINAT MARKER ====================

    private Point lookupBoardCenterPx(int id) {
        if (id >= 100 && id < 200) {
            return lookupTopBoardCenterPx(id);
        }
        if (id >= 400 && id < 500) {
            return lookupBottomBoardCenterPx(id);
        }
        return null;
    }

    private Point lookupTopBoardCenterPx(int id) {
        Integer[] rc = lookupTopBoardRowCol(id);
        if (rc == null) return null;
        return new Point(
                (rc[1] + 0.5) * CELL_PX,
                (rc[0] + 0.5) * CELL_PX
        );
    }

    private Point lookupBottomBoardCenterPx(int id) {
        Integer[] rc = lookupBottomBoardRowCol(id);
        if (rc == null) return null;
        return new Point(
                (rc[1] + 0.5) * CELL_PX,
                BOTTOM_BOARD_OFFSET_Y_PX + (rc[0] + 0.5) * CELL_PX
        );
    }

    // Layout papan atas
    private Integer[] lookupTopBoardRowCol(int id) {
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

    // Layout papan bawah — WAJIB cocokkan dengan papan asli
    private Integer[] lookupBottomBoardRowCol(int id) {
        switch (id) {
            case 403: return new Integer[]{0, 0};
            case 435: return new Integer[]{0, 10};

            case 439: return new Integer[]{1, 10};

            case 402: return new Integer[]{2, 0};
            case 434: return new Integer[]{2, 10};

            case 438: return new Integer[]{3, 10};

            case 401: return new Integer[]{4, 0};
            case 433: return new Integer[]{4, 10};

            case 400: return new Integer[]{5, 0};
            case 405: return new Integer[]{5, 2};
            case 429: return new Integer[]{5, 8};
            case 437: return new Integer[]{5, 10};

            case 404: return new Integer[]{6, 2};
            case 432: return new Integer[]{6, 8};

            case 428: return new Integer[]{7, 8};
            case 436: return new Integer[]{7, 10};

            default: return null;
        }
    }

    // ==================== ARUCO PARAMETERS ====================

    private DetectorParameters createArucoDetectorParameters() {
        DetectorParameters parameters = new DetectorParameters();
        parameters.set_adaptiveThreshWinSizeMin(3);
        parameters.set_adaptiveThreshWinSizeMax(53);
        parameters.set_adaptiveThreshWinSizeStep(4);
        parameters.set_minMarkerPerimeterRate(0.005);
        parameters.set_maxMarkerPerimeterRate(4.0);
        parameters.set_polygonalApproxAccuracyRate(0.08);
        parameters.set_minCornerDistanceRate(0.005);
        parameters.set_minDistanceToBorder(1);
        parameters.set_cornerRefinementMethod(1);
        parameters.set_cornerRefinementWinSize(7);
        parameters.set_cornerRefinementMaxIterations(80);
        parameters.set_cornerRefinementMinAccuracy(0.01);
        return parameters;
    }

    // ==================== FALLBACK: BOUNDING BOX DUA PAPAN ====================

    private Bitmap autoPerspectiveFromTwoBoards(Bitmap bitmap) {
        Log.d(TAG, "=== autoPerspectiveFromTwoBoards START ===");

        Mat srcMat = new Mat();
        Mat gray = new Mat();
        Mat warped = new Mat();

        try {
            Utils.bitmapToMat(bitmap, srcMat);

            if (srcMat.channels() == 4) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (srcMat.channels() == 3) {
                Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = srcMat.clone();
            }

            Point[] boardCorners = detectTwoBoardCorners(gray);
            if (boardCorners == null) {
                Log.e(TAG, "detectTwoBoardCorners returned null");
                return null;
            }

            Point tl = boardCorners[0];
            Point tr = boardCorners[1];
            Point br = boardCorners[2];
            Point bl = boardCorners[3];

            double widthTop = distance(tl, tr);
            double widthBottom = distance(bl, br);
            double maxWidth = Math.max(widthTop, widthBottom);

            double heightLeft = distance(tl, bl);
            double heightRight = distance(tr, br);
            double maxHeight = Math.max(heightLeft, heightRight);

            double scale = Math.min(
                    (double) MAX_WARP_W / Math.max(1.0, maxWidth),
                    (double) MAX_WARP_H / Math.max(1.0, maxHeight)
            );

            if (scale < 1.0) {
                maxWidth = Math.max(800, Math.round(maxWidth * scale));
                maxHeight = Math.max(1600, Math.round(maxHeight * scale));
            }

            MatOfPoint2f srcPoints = new MatOfPoint2f(tl, tr, br, bl);
            MatOfPoint2f dstPoints = new MatOfPoint2f(
                    new Point(0, 0),
                    new Point(maxWidth - 1, 0),
                    new Point(maxWidth - 1, maxHeight - 1),
                    new Point(0, maxHeight - 1)
            );

            Mat H = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Imgproc.warpPerspective(srcMat, warped, H, new Size(maxWidth, maxHeight));

            Bitmap result = Bitmap.createBitmap(
                    warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888
            );
            Utils.matToBitmap(warped, result);

            srcPoints.release();
            dstPoints.release();
            H.release();

            Log.d(TAG, "=== autoPerspectiveFromTwoBoards END ===");
            return result;

        } catch (Exception e) {
            Log.e(TAG, "autoPerspectiveFromTwoBoards error", e);
            return null;
        } finally {
            gray.release();
            srcMat.release();
            warped.release();
        }
    }

    private Point[] detectTwoBoardCorners(Mat gray) {
        Log.d(TAG, "=== detectTwoBoardCorners START ===");

        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();
        List<Mat> rejected = new ArrayList<>();

        try {
            Dictionary dictionary = Objdetect.getPredefinedDictionary(MARKER_DICT);
            DetectorParameters parameters = createArucoDetectorParameters();
            ArucoDetector detector = new ArucoDetector(dictionary, parameters);
            detector.detectMarkers(gray, corners, ids, rejected);

            if (ids.empty() || corners.size() < 8) {
                Log.e(TAG, "detectTwoBoardCorners failed: not enough markers");
                return null;
            }

            List<Point> centers = new ArrayList<>();
            double avgMarkerSide = 0.0;

            for (Mat cornerMat : corners) {
                double[] xy0 = cornerMat.get(0, 0);
                double[] xy1 = cornerMat.get(0, 1);
                double[] xy2 = cornerMat.get(0, 2);
                double[] xy3 = cornerMat.get(0, 3);

                double cx = (xy0[0] + xy1[0] + xy2[0] + xy3[0]) / 4.0;
                double cy = (xy0[1] + xy1[1] + xy2[1] + xy3[1]) / 4.0;
                centers.add(new Point(cx, cy));

                double s1 = distance(new Point(xy0[0], xy0[1]), new Point(xy1[0], xy1[1]));
                double s2 = distance(new Point(xy1[0], xy1[1]), new Point(xy2[0], xy2[1]));
                double s3 = distance(new Point(xy2[0], xy2[1]), new Point(xy3[0], xy3[1]));
                double s4 = distance(new Point(xy3[0], xy3[1]), new Point(xy0[0], xy0[1]));
                avgMarkerSide += (s1 + s2 + s3 + s4) / 4.0;
            }

            avgMarkerSide /= Math.max(1, corners.size());

            int sidePad = Math.max(80, (int) Math.round(avgMarkerSide * 1.2));
            int topPad = Math.max(60, (int) Math.round(avgMarkerSide * 0.8));
            int bottomPad = Math.max(380, (int) Math.round(avgMarkerSide * 5.0));

            double medianY = computeMedianY(centers);

            List<Point> topPoints = new ArrayList<>();
            List<Point> bottomPoints = new ArrayList<>();

            for (Mat cornerMat : corners) {
                double[] xy0 = cornerMat.get(0, 0);
                double[] xy1 = cornerMat.get(0, 1);
                double[] xy2 = cornerMat.get(0, 2);
                double[] xy3 = cornerMat.get(0, 3);

                double cy = (xy0[1] + xy1[1] + xy2[1] + xy3[1]) / 4.0;

                if (cy < medianY) {
                    topPoints.add(new Point(xy0[0], xy0[1]));
                    topPoints.add(new Point(xy1[0], xy1[1]));
                    topPoints.add(new Point(xy2[0], xy2[1]));
                    topPoints.add(new Point(xy3[0], xy3[1]));
                } else {
                    bottomPoints.add(new Point(xy0[0], xy0[1]));
                    bottomPoints.add(new Point(xy1[0], xy1[1]));
                    bottomPoints.add(new Point(xy2[0], xy2[1]));
                    bottomPoints.add(new Point(xy3[0], xy3[1]));
                }
            }

            if (topPoints.size() < 4 || bottomPoints.size() < 4) {
                Log.e(TAG, "topPoints or bottomPoints < 4");
                return null;
            }

            Rect topRect = boundingRectCustom(
                    topPoints, sidePad, topPad, sidePad, sidePad,
                    gray.cols(), gray.rows()
            );

            Rect bottomRect = boundingRectCustom(
                    bottomPoints, sidePad, sidePad, sidePad, bottomPad,
                    gray.cols(), gray.rows()
            );

            if (topRect == null || bottomRect == null) {
                Log.e(TAG, "topRect or bottomRect null");
                return null;
            }

            Point tl = new Point(topRect.x, topRect.y);
            Point tr = new Point(topRect.x + topRect.width, topRect.y);
            Point br = new Point(
                    bottomRect.x + bottomRect.width,
                    bottomRect.y + bottomRect.height
            );
            Point bl = new Point(
                    bottomRect.x,
                    bottomRect.y + bottomRect.height
            );

            Log.d(TAG, "=== detectTwoBoardCorners END ===");
            return new Point[]{tl, tr, br, bl};

        } catch (Exception e) {
            Log.e(TAG, "detectTwoBoardCorners error", e);
            return null;
        } finally {
            for (Mat c : corners) c.release();
            for (Mat r : rejected) r.release();
            ids.release();
        }
    }

    // ==================== UTILITY ====================

    private double computeMedianY(List<Point> points) {
        List<Double> ys = new ArrayList<>();
        for (Point p : points) ys.add(p.y);
        Collections.sort(ys);
        int n = ys.size();
        if (n == 0) return 0;
        if (n % 2 == 1) return ys.get(n / 2);
        return (ys.get(n / 2 - 1) + ys.get(n / 2)) / 2.0;
    }

    private Rect boundingRectCustom(
            List<Point> pts,
            int leftPad, int topPad,
            int rightPad, int bottomPad,
            int imageW, int imageH
    ) {
        if (pts == null || pts.isEmpty()) return null;

        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;

        for (Point p : pts) {
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
        }

        int x1 = Math.max(0, (int) Math.floor(minX) - leftPad);
        int y1 = Math.max(0, (int) Math.floor(minY) - topPad);
        int x2 = Math.min(imageW - 1, (int) Math.ceil(maxX) + rightPad);
        int y2 = Math.min(imageH - 1, (int) Math.ceil(maxY) + bottomPad);

        int w = x2 - x1;
        int h = y2 - y1;
        if (w <= 0 || h <= 0) return null;

        return new Rect(x1, y1, w, h);
    }

    private double distance(Point p1, Point p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }

    // ==================== LOAD & ROTATE BITMAP ====================

    private Bitmap loadAndRotateBitmap(Uri imageUri) {
        try {
            Bitmap original = loadBitmapFromUri(imageUri);
            if (original == null) return null;
            return rotateBitmapIfRequired(original, imageUri);
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
                    bitmap, 0, 0,
                    bitmap.getWidth(), bitmap.getHeight(),
                    matrix, true
            );

        } catch (Exception e) {
            Log.e(TAG, "rotateBitmapIfRequired error", e);
            return bitmap;
        }
    }

    // ==================== SAVE ====================

    private void saveBitmapToAppFiles(Bitmap bitmap) {
        try {
            File picturesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
            if (picturesDir != null && !picturesDir.exists()) {
                picturesDir.mkdirs();
            }

            String fileName = new SimpleDateFormat(
                    "yyyyMMdd_HHmmss", Locale.getDefault()
            ).format(new Date());

            File imageFile = new File(picturesDir, "GAL_" + fileName + ".jpg");

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos);
            fos.flush();
            fos.close();

            Log.d(TAG, "Saved image to: " + imageFile.getAbsolutePath());
            Toast.makeText(this, "Gambar berhasil disimpan", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToAppFiles error", e);
            Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show();
        }
    }
}
