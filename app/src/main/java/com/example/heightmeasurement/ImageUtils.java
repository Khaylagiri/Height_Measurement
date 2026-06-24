package com.example.heightmeasurement;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.util.Log;

import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

public final class ImageUtils {

    private static final String TAG = "ImageUtils";

    private ImageUtils() {
    }

    public static Bitmap decodeUri(Context context, Uri uri, int maxSidePx) {
        if (context == null || uri == null || maxSidePx <= 0) {
            return null;
        }

        try {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;

            try (InputStream input = context.getContentResolver().openInputStream(uri)) {
                if (input == null) {
                    return null;
                }
                BitmapFactory.decodeStream(input, null, bounds);
            }

            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                return null;
            }

            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            options.inSampleSize = calculateInSampleSize(
                    bounds.outWidth,
                    bounds.outHeight,
                    maxSidePx
            );

            Bitmap decoded;
            try (InputStream input = context.getContentResolver().openInputStream(uri)) {
                if (input == null) {
                    return null;
                }
                decoded = BitmapFactory.decodeStream(input, null, options);
            }

            if (decoded == null) {
                return null;
            }

            int rotation = readRotation(context, uri);
            Bitmap rotated = rotateBitmap(decoded, rotation);
            if (rotated != decoded && !decoded.isRecycled()) {
                decoded.recycle();
            }

            return scaleDownIfNeeded(rotated, maxSidePx);

        } catch (OutOfMemoryError oom) {
            Log.e(TAG, "decodeUri out of memory", oom);
            return null;
        } catch (Exception e) {
            Log.e(TAG, "decodeUri error", e);
            return null;
        }
    }

    public static Bitmap decodeFile(String path, int maxSidePx) {
        if (path == null || path.trim().isEmpty() || maxSidePx <= 0) {
            return null;
        }

        File file = new File(path);
        if (!file.exists() || !file.isFile()) {
            return null;
        }

        try {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(path, bounds);

            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                return null;
            }

            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            options.inSampleSize = calculateInSampleSize(
                    bounds.outWidth,
                    bounds.outHeight,
                    maxSidePx
            );

            Bitmap decoded = BitmapFactory.decodeFile(path, options);
            if (decoded == null) {
                return null;
            }

            int rotation = readRotation(file);
            Bitmap rotated = rotateBitmap(decoded, rotation);
            if (rotated != decoded && !decoded.isRecycled()) {
                decoded.recycle();
            }

            return scaleDownIfNeeded(rotated, maxSidePx);

        } catch (OutOfMemoryError oom) {
            Log.e(TAG, "decodeFile out of memory", oom);
            return null;
        } catch (Exception e) {
            Log.e(TAG, "decodeFile error", e);
            return null;
        }
    }

    private static int calculateInSampleSize(int width, int height, int maxSidePx) {
        int sample = 1;
        int largest = Math.max(width, height);

        while (largest / (sample * 2) >= maxSidePx) {
            sample *= 2;
        }

        return Math.max(1, sample);
    }

    private static Bitmap scaleDownIfNeeded(Bitmap bitmap, int maxSidePx) {
        if (bitmap == null) {
            return null;
        }

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int largest = Math.max(width, height);

        if (largest <= maxSidePx) {
            return bitmap;
        }

        double scale = maxSidePx / (double) largest;
        int newWidth = Math.max(1, (int) Math.round(width * scale));
        int newHeight = Math.max(1, (int) Math.round(height * scale));

        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
        if (scaled != bitmap && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return scaled;
    }

    private static int readRotation(Context context, Uri uri) {
        try (InputStream input = context.getContentResolver().openInputStream(uri)) {
            if (input == null) {
                return 0;
            }
            ExifInterface exif = new ExifInterface(input);
            return exifToDegrees(exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            ));
        } catch (Exception e) {
            Log.w(TAG, "readRotation uri failed", e);
            return 0;
        }
    }

    private static int readRotation(File file) {
        try (InputStream input = new FileInputStream(file)) {
            ExifInterface exif = new ExifInterface(input);
            return exifToDegrees(exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            ));
        } catch (Exception e) {
            return 0;
        }
    }

    private static int exifToDegrees(int orientation) {
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                return 90;
            case ExifInterface.ORIENTATION_ROTATE_180:
                return 180;
            case ExifInterface.ORIENTATION_ROTATE_270:
                return 270;
            default:
                return 0;
        }
    }

    private static Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        if (bitmap == null || degrees == 0) {
            return bitmap;
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);

        return Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.getWidth(),
                bitmap.getHeight(),
                matrix,
                true
        );
    }
}
