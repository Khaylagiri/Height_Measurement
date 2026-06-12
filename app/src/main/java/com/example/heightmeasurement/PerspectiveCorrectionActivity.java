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
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
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

    private static final double MARKER_SIZE_CELLS = 0.74;

    private static final int RESULT_SHIFT_X_PX = -100;

    private static final int CROP_WHITE_THRESHOLD = 245;

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_perspective_correction);

        imageViewResult = findViewById(R.id.imageViewResult);
        btnPerspective = findViewById(R.id.btnPerspective);
        btnSaveGalleryImage = findViewById(R.id.btnSaveGalleryImage);

        imageViewResult.setScaleType(ImageView.ScaleType.FIT_CENTER);
        imageViewResult.setAdjustViewBounds(true);

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

        btnSaveGalleryImage.setOnClickListener(v -> {
            if (currentBitmap != null) {
                saveBitmapToAppFiles(currentBitmap);
            }
        });
    }

    private void runPerspectiveOnly() {
        try {
            btnPerspective.setEnabled(false);

            Bitmap result = autoPerspectiveCamScannerStyle(originalBitmap);

            if (result == null) {
                Toast.makeText(
                        this,
                        "Perspective gagal. Pastikan marker ArUco terlihat jelas dan tidak terlalu tertutup badan.",
                        Toast.LENGTH_LONG
                ).show();
                return;
            }

            currentBitmap = result;
            imageViewResult.setImageBitmap(currentBitmap);

            Toast.makeText(
                    this,
                    "Perspective berhasil",
                    Toast.LENGTH_SHORT
            ).show();

        } catch (Exception e) {
            Log.e(TAG, "runPerspectiveOnly error", e);
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        } finally {
            btnPerspective.setEnabled(true);
        }
    }

    private Bitmap autoPerspectiveCamScannerStyle(Bitmap bitmap) {
        Mat srcMat = new Mat();
        Mat gray = new Mat();
        Mat warped = new Mat();
        Mat shiftedWarped = null;
        Mat croppedWarped = null;

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

            Imgproc.equalizeHist(gray, gray);

            Dictionary dictionary = Objdetect.getPredefinedDictionary(MARKER_DICT);
            DetectorParameters parameters = createArucoDetectorParameters();
            ArucoDetector detector = new ArucoDetector(dictionary, parameters);

            detector.detectMarkers(gray, corners, ids, rejected);

            if (ids.empty() || corners.isEmpty()) {
                return null;
            }

            List<Point> imagePoints = new ArrayList<>();
            List<Point> boardPoints = new ArrayList<>();

            int usedMarkers = 0;

            for (int i = 0; i < ids.rows(); i++) {
                int id = (int) ids.get(i, 0)[0];

                Point[] worldCorners = getMarkerWorldCorners(id);
                if (worldCorners == null) {
                    continue;
                }

                Point[] imageCorners = getMarkerImageCornersOrdered(corners.get(i));
                if (imageCorners == null) {
                    continue;
                }

                for (int k = 0; k < 4; k++) {
                    imagePoints.add(imageCorners[k]);
                    boardPoints.add(worldCorners[k]);
                }

                usedMarkers++;
            }

            Log.d(TAG, "Detected markers: " + corners.size() + ", used markers: " + usedMarkers);

            if (usedMarkers < 4 || imagePoints.size() < 16) {
                return null;
            }

            imagePointsMat = new MatOfPoint2f(imagePoints.toArray(new Point[0]));
            boardPointsMat = new MatOfPoint2f(boardPoints.toArray(new Point[0]));

            homography = Calib3d.findHomography(
                    imagePointsMat,
                    boardPointsMat,
                    Calib3d.RANSAC,
                    5.0
            );

            if (homography.empty()) {
                return null;
            }

            Imgproc.warpPerspective(
                    srcMat,
                    warped,
                    homography,
                    new Size(OUTPUT_WIDTH_PX, OUTPUT_HEIGHT_PX),
                    Imgproc.INTER_LINEAR
            );

            shiftedWarped = shiftResultX(warped, RESULT_SHIFT_X_PX);

            croppedWarped = cropHorizontalBlankBorders(shiftedWarped);

            Bitmap result = Bitmap.createBitmap(
                    croppedWarped.cols(),
                    croppedWarped.rows(),
                    Bitmap.Config.ARGB_8888
            );

            Utils.matToBitmap(croppedWarped, result);
            return result;

        } catch (Exception e) {
            Log.e(TAG, "autoPerspectiveCamScannerStyle error", e);
            return null;
        } finally {
            for (Mat c : corners) {
                c.release();
            }

            for (Mat r : rejected) {
                r.release();
            }

            if (imagePointsMat != null) imagePointsMat.release();
            if (boardPointsMat != null) boardPointsMat.release();

            ids.release();
            homography.release();
            srcMat.release();
            gray.release();
            warped.release();

            if (shiftedWarped != null) {
                shiftedWarped.release();
            }

            if (croppedWarped != null) {
                croppedWarped.release();
            }
        }
    }

    private Mat shiftResultX(Mat input, int shiftXPx) {
        Mat output = new Mat(input.rows(), input.cols(), input.type());

        Mat translation = Mat.eye(2, 3, CvType.CV_64F);

        translation.put(0, 2, shiftXPx);
        translation.put(1, 2, 0);

        Imgproc.warpAffine(
                input,
                output,
                translation,
                input.size(),
                Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT,
                new Scalar(255, 255, 255, 255)
        );

        translation.release();

        return output;
    }

    private Mat cropHorizontalBlankBorders(Mat input) {
        Mat gray = new Mat();
        Mat mask = new Mat();

        try {
            if (input.channels() == 4) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGBA2GRAY);
            } else if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            Imgproc.threshold(
                    gray,
                    mask,
                    CROP_WHITE_THRESHOLD,
                    255,
                    Imgproc.THRESH_BINARY_INV
            );

            int left = -1;
            int right = -1;

            int minPixelsPerColumn = Math.max(5, input.rows() / 400);

            for (int x = 0; x < mask.cols(); x++) {
                Mat col = mask.col(x);
                int count = Core.countNonZero(col);
                col.release();

                if (count > minPixelsPerColumn) {
                    left = x;
                    break;
                }
            }

            for (int x = mask.cols() - 1; x >= 0; x--) {
                Mat col = mask.col(x);
                int count = Core.countNonZero(col);
                col.release();

                if (count > minPixelsPerColumn) {
                    right = x;
                    break;
                }
            }

            if (left < 0 || right < 0 || right <= left) {
                return input.clone();
            }

            int padding = 0;

            left = Math.max(0, left - padding);
            right = Math.min(input.cols() - 1, right + padding);

            Rect cropRect = new Rect(
                    left,
                    0,
                    right - left + 1,
                    input.rows()
            );

            return new Mat(input, cropRect).clone();

        } finally {
            gray.release();
            mask.release();
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
        if (pts == null || pts.length != 4) return null;

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
                    "PERSPECTIVE_" + fileName + ".jpg"
            );

            FileOutputStream fos = new FileOutputStream(imageFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos);
            fos.flush();
            fos.close();

            Toast.makeText(this, "Gambar berhasil disimpan", Toast.LENGTH_SHORT).show();

        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToAppFiles error", e);
            Toast.makeText(this, "Gagal menyimpan gambar", Toast.LENGTH_SHORT).show();
        }
    }
}