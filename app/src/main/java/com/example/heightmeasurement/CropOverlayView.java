package com.example.heightmeasurement;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import org.opencv.core.Point;

public class CropOverlayView extends View {

    private final Paint linePaint = new Paint();
    private final Paint pointPaint = new Paint();
    private final Paint fillPaint = new Paint();

    private final Point[] points = new Point[]{
            new Point(200, 200),
            new Point(800, 200),
            new Point(800, 1400),
            new Point(200, 1400)
    };

    private int activePoint = -1;
    private final float pointRadius = 28f;

    public CropOverlayView(Context context) {
        super(context);
        init();
    }

    public CropOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public CropOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        linePaint.setColor(Color.GREEN);
        linePaint.setStrokeWidth(6f);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setAntiAlias(true);

        pointPaint.setColor(Color.RED);
        pointPaint.setStyle(Paint.Style.FILL);
        pointPaint.setAntiAlias(true);

        fillPaint.setColor(Color.argb(60, 0, 255, 0));
        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setAntiAlias(true);
    }

    public void setPoints(Point tl, Point tr, Point br, Point bl) {
        points[0] = tl;
        points[1] = tr;
        points[2] = br;
        points[3] = bl;
        invalidate();
    }

    public Point[] getPoints() {
        return new Point[]{
                new Point(points[0].x, points[0].y),
                new Point(points[1].x, points[1].y),
                new Point(points[2].x, points[2].y),
                new Point(points[3].x, points[3].y)
        };
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        Path path = new Path();
        path.moveTo((float) points[0].x, (float) points[0].y);
        path.lineTo((float) points[1].x, (float) points[1].y);
        path.lineTo((float) points[2].x, (float) points[2].y);
        path.lineTo((float) points[3].x, (float) points[3].y);
        path.close();

        canvas.drawPath(path, fillPaint);
        canvas.drawPath(path, linePaint);

        for (Point p : points) {
            canvas.drawCircle((float) p.x, (float) p.y, pointRadius, pointPaint);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                activePoint = findNearestPoint(x, y);
                return activePoint != -1;

            case MotionEvent.ACTION_MOVE:
                if (activePoint != -1) {
                    points[activePoint].x = x;
                    points[activePoint].y = y;
                    invalidate();
                    return true;
                }
                return false;

            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                boolean handled = activePoint != -1;
                activePoint = -1;
                return handled;
        }

        return false;
    }

    private int findNearestPoint(float x, float y) {
        for (int i = 0; i < points.length; i++) {
            double dx = points[i].x - x;
            double dy = points[i].y - y;
            if (Math.sqrt(dx * dx + dy * dy) < 90) {
                return i;
            }
        }
        return -1;
    }
}