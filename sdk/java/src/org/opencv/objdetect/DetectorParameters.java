//
// This file is auto-generated. Please don't modify it!
//
package org.opencv.objdetect;

// C++: class DetectorParameters
/**
 * struct DetectorParameters is used by ArucoDetector
 */
public class DetectorParameters {

    protected final long nativeObj;
    protected DetectorParameters(long addr) { nativeObj = addr; }

    public long getNativeObjAddr() { return nativeObj; }

    // internal usage only
    public static DetectorParameters __fromPtr__(long addr) { return new DetectorParameters(addr); }

    public DetectorParameters() {
        nativeObj = DetectorParameters_0();
    }

    public int get_adaptiveThreshWinSizeMin() {
        return get_adaptiveThreshWinSizeMin_0(nativeObj);
    }

    public void set_adaptiveThreshWinSizeMin(int adaptiveThreshWinSizeMin) {
        set_adaptiveThreshWinSizeMin_0(nativeObj, adaptiveThreshWinSizeMin);
    }

    public int get_adaptiveThreshWinSizeMax() {
        return get_adaptiveThreshWinSizeMax_0(nativeObj);
    }

    public void set_adaptiveThreshWinSizeMax(int adaptiveThreshWinSizeMax) {
        set_adaptiveThreshWinSizeMax_0(nativeObj, adaptiveThreshWinSizeMax);
    }

    public int get_adaptiveThreshWinSizeStep() {
        return get_adaptiveThreshWinSizeStep_0(nativeObj);
    }

    public void set_adaptiveThreshWinSizeStep(int adaptiveThreshWinSizeStep) {
        set_adaptiveThreshWinSizeStep_0(nativeObj, adaptiveThreshWinSizeStep);
    }

    public double get_adaptiveThreshConstant() {
        return get_adaptiveThreshConstant_0(nativeObj);
    }

    public void set_adaptiveThreshConstant(double adaptiveThreshConstant) {
        set_adaptiveThreshConstant_0(nativeObj, adaptiveThreshConstant);
    }

    public double get_minMarkerPerimeterRate() {
        return get_minMarkerPerimeterRate_0(nativeObj);
    }

    public void set_minMarkerPerimeterRate(double minMarkerPerimeterRate) {
        set_minMarkerPerimeterRate_0(nativeObj, minMarkerPerimeterRate);
    }

    public double get_maxMarkerPerimeterRate() {
        return get_maxMarkerPerimeterRate_0(nativeObj);
    }

    public void set_maxMarkerPerimeterRate(double maxMarkerPerimeterRate) {
        set_maxMarkerPerimeterRate_0(nativeObj, maxMarkerPerimeterRate);
    }

    public double get_polygonalApproxAccuracyRate() {
        return get_polygonalApproxAccuracyRate_0(nativeObj);
    }

    public void set_polygonalApproxAccuracyRate(double polygonalApproxAccuracyRate) {
        set_polygonalApproxAccuracyRate_0(nativeObj, polygonalApproxAccuracyRate);
    }

    public double get_minCornerDistanceRate() {
        return get_minCornerDistanceRate_0(nativeObj);
    }

    public void set_minCornerDistanceRate(double minCornerDistanceRate) {
        set_minCornerDistanceRate_0(nativeObj, minCornerDistanceRate);
    }

    public int get_minDistanceToBorder() {
        return get_minDistanceToBorder_0(nativeObj);
    }

    public void set_minDistanceToBorder(int minDistanceToBorder) {
        set_minDistanceToBorder_0(nativeObj, minDistanceToBorder);
    }

    public double get_minMarkerDistanceRate() {
        return get_minMarkerDistanceRate_0(nativeObj);
    }

    public void set_minMarkerDistanceRate(double minMarkerDistanceRate) {
        set_minMarkerDistanceRate_0(nativeObj, minMarkerDistanceRate);
    }

    public float get_minGroupDistance() {
        return get_minGroupDistance_0(nativeObj);
    }

    public void set_minGroupDistance(float minGroupDistance) {
        set_minGroupDistance_0(nativeObj, minGroupDistance);
    }

    public int get_cornerRefinementMethod() {
        return get_cornerRefinementMethod_0(nativeObj);
    }

    public void set_cornerRefinementMethod(int cornerRefinementMethod) {
        set_cornerRefinementMethod_0(nativeObj, cornerRefinementMethod);
    }

    public int get_cornerRefinementWinSize() {
        return get_cornerRefinementWinSize_0(nativeObj);
    }

    public void set_cornerRefinementWinSize(int cornerRefinementWinSize) {
        set_cornerRefinementWinSize_0(nativeObj, cornerRefinementWinSize);
    }

    public float get_relativeCornerRefinmentWinSize() {
        return get_relativeCornerRefinmentWinSize_0(nativeObj);
    }

    public void set_relativeCornerRefinmentWinSize(float relativeCornerRefinmentWinSize) {
        set_relativeCornerRefinmentWinSize_0(nativeObj, relativeCornerRefinmentWinSize);
    }

    public int get_cornerRefinementMaxIterations() {
        return get_cornerRefinementMaxIterations_0(nativeObj);
    }

    public void set_cornerRefinementMaxIterations(int cornerRefinementMaxIterations) {
        set_cornerRefinementMaxIterations_0(nativeObj, cornerRefinementMaxIterations);
    }

    public double get_cornerRefinementMinAccuracy() {
        return get_cornerRefinementMinAccuracy_0(nativeObj);
    }

    public void set_cornerRefinementMinAccuracy(double cornerRefinementMinAccuracy) {
        set_cornerRefinementMinAccuracy_0(nativeObj, cornerRefinementMinAccuracy);
    }

    public int get_markerBorderBits() {
        return get_markerBorderBits_0(nativeObj);
    }

    public void set_markerBorderBits(int markerBorderBits) {
        set_markerBorderBits_0(nativeObj, markerBorderBits);
    }

    public int get_perspectiveRemovePixelPerCell() {
        return get_perspectiveRemovePixelPerCell_0(nativeObj);
    }

    public void set_perspectiveRemovePixelPerCell(int perspectiveRemovePixelPerCell) {
        set_perspectiveRemovePixelPerCell_0(nativeObj, perspectiveRemovePixelPerCell);
    }

    public double get_perspectiveRemoveIgnoredMarginPerCell() {
        return get_perspectiveRemoveIgnoredMarginPerCell_0(nativeObj);
    }

    public void set_perspectiveRemoveIgnoredMarginPerCell(double perspectiveRemoveIgnoredMarginPerCell) {
        set_perspectiveRemoveIgnoredMarginPerCell_0(nativeObj, perspectiveRemoveIgnoredMarginPerCell);
    }

    public double get_maxErroneousBitsInBorderRate() {
        return get_maxErroneousBitsInBorderRate_0(nativeObj);
    }

    public void set_maxErroneousBitsInBorderRate(double maxErroneousBitsInBorderRate) {
        set_maxErroneousBitsInBorderRate_0(nativeObj, maxErroneousBitsInBorderRate);
    }

    public double get_minOtsuStdDev() {
        return get_minOtsuStdDev_0(nativeObj);
    }

    public void set_minOtsuStdDev(double minOtsuStdDev) {
        set_minOtsuStdDev_0(nativeObj, minOtsuStdDev);
    }

    public double get_errorCorrectionRate() {
        return get_errorCorrectionRate_0(nativeObj);
    }

    public void set_errorCorrectionRate(double errorCorrectionRate) {
        set_errorCorrectionRate_0(nativeObj, errorCorrectionRate);
    }

    public float get_aprilTagQuadDecimate() {
        return get_aprilTagQuadDecimate_0(nativeObj);
    }

    public void set_aprilTagQuadDecimate(float aprilTagQuadDecimate) {
        set_aprilTagQuadDecimate_0(nativeObj, aprilTagQuadDecimate);
    }

    public float get_aprilTagQuadSigma() {
        return get_aprilTagQuadSigma_0(nativeObj);
    }

    public void set_aprilTagQuadSigma(float aprilTagQuadSigma) {
        set_aprilTagQuadSigma_0(nativeObj, aprilTagQuadSigma);
    }

    public int get_aprilTagMinClusterPixels() {
        return get_aprilTagMinClusterPixels_0(nativeObj);
    }

    public void set_aprilTagMinClusterPixels(int aprilTagMinClusterPixels) {
        set_aprilTagMinClusterPixels_0(nativeObj, aprilTagMinClusterPixels);
    }

    public int get_aprilTagMaxNmaxima() {
        return get_aprilTagMaxNmaxima_0(nativeObj);
    }

    public void set_aprilTagMaxNmaxima(int aprilTagMaxNmaxima) {
        set_aprilTagMaxNmaxima_0(nativeObj, aprilTagMaxNmaxima);
    }

    public float get_aprilTagCriticalRad() {
        return get_aprilTagCriticalRad_0(nativeObj);
    }

    public void set_aprilTagCriticalRad(float aprilTagCriticalRad) {
        set_aprilTagCriticalRad_0(nativeObj, aprilTagCriticalRad);
    }

    public float get_aprilTagMaxLineFitMse() {
        return get_aprilTagMaxLineFitMse_0(nativeObj);
    }

    public void set_aprilTagMaxLineFitMse(float aprilTagMaxLineFitMse) {
        set_aprilTagMaxLineFitMse_0(nativeObj, aprilTagMaxLineFitMse);
    }

    public int get_aprilTagMinWhiteBlackDiff() {
        return get_aprilTagMinWhiteBlackDiff_0(nativeObj);
    }

    public void set_aprilTagMinWhiteBlackDiff(int aprilTagMinWhiteBlackDiff) {
        set_aprilTagMinWhiteBlackDiff_0(nativeObj, aprilTagMinWhiteBlackDiff);
    }

    public int get_aprilTagDeglitch() {
        return get_aprilTagDeglitch_0(nativeObj);
    }

    public void set_aprilTagDeglitch(int aprilTagDeglitch) {
        set_aprilTagDeglitch_0(nativeObj, aprilTagDeglitch);
    }

    public boolean get_detectInvertedMarker() {
        return get_detectInvertedMarker_0(nativeObj);
    }

    public void set_detectInvertedMarker(boolean detectInvertedMarker) {
        set_detectInvertedMarker_0(nativeObj, detectInvertedMarker);
    }

    public boolean get_useAruco3Detection() {
        return get_useAruco3Detection_0(nativeObj);
    }

    public void set_useAruco3Detection(boolean useAruco3Detection) {
        set_useAruco3Detection_0(nativeObj, useAruco3Detection);
    }

    public int get_minSideLengthCanonicalImg() {
        return get_minSideLengthCanonicalImg_0(nativeObj);
    }

    public void set_minSideLengthCanonicalImg(int minSideLengthCanonicalImg) {
        set_minSideLengthCanonicalImg_0(nativeObj, minSideLengthCanonicalImg);
    }

    public float get_minMarkerLengthRatioOriginalImg() {
        return get_minMarkerLengthRatioOriginalImg_0(nativeObj);
    }

    public void set_minMarkerLengthRatioOriginalImg(float minMarkerLengthRatioOriginalImg) {
        set_minMarkerLengthRatioOriginalImg_0(nativeObj, minMarkerLengthRatioOriginalImg);
    }

    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
    }

    private static native long DetectorParameters_0();
    private static native int get_adaptiveThreshWinSizeMin_0(long nativeObj);
    private static native void set_adaptiveThreshWinSizeMin_0(long nativeObj, int adaptiveThreshWinSizeMin);
    private static native int get_adaptiveThreshWinSizeMax_0(long nativeObj);
    private static native void set_adaptiveThreshWinSizeMax_0(long nativeObj, int adaptiveThreshWinSizeMax);
    private static native int get_adaptiveThreshWinSizeStep_0(long nativeObj);
    private static native void set_adaptiveThreshWinSizeStep_0(long nativeObj, int adaptiveThreshWinSizeStep);
    private static native double get_adaptiveThreshConstant_0(long nativeObj);
    private static native void set_adaptiveThreshConstant_0(long nativeObj, double adaptiveThreshConstant);
    private static native double get_minMarkerPerimeterRate_0(long nativeObj);
    private static native void set_minMarkerPerimeterRate_0(long nativeObj, double minMarkerPerimeterRate);
    private static native double get_maxMarkerPerimeterRate_0(long nativeObj);
    private static native void set_maxMarkerPerimeterRate_0(long nativeObj, double maxMarkerPerimeterRate);
    private static native double get_polygonalApproxAccuracyRate_0(long nativeObj);
    private static native void set_polygonalApproxAccuracyRate_0(long nativeObj, double polygonalApproxAccuracyRate);
    private static native double get_minCornerDistanceRate_0(long nativeObj);
    private static native void set_minCornerDistanceRate_0(long nativeObj, double minCornerDistanceRate);
    private static native int get_minDistanceToBorder_0(long nativeObj);
    private static native void set_minDistanceToBorder_0(long nativeObj, int minDistanceToBorder);
    private static native double get_minMarkerDistanceRate_0(long nativeObj);
    private static native void set_minMarkerDistanceRate_0(long nativeObj, double minMarkerDistanceRate);
    private static native float get_minGroupDistance_0(long nativeObj);
    private static native void set_minGroupDistance_0(long nativeObj, float minGroupDistance);
    private static native int get_cornerRefinementMethod_0(long nativeObj);
    private static native void set_cornerRefinementMethod_0(long nativeObj, int cornerRefinementMethod);
    private static native int get_cornerRefinementWinSize_0(long nativeObj);
    private static native void set_cornerRefinementWinSize_0(long nativeObj, int cornerRefinementWinSize);
    private static native float get_relativeCornerRefinmentWinSize_0(long nativeObj);
    private static native void set_relativeCornerRefinmentWinSize_0(long nativeObj, float relativeCornerRefinmentWinSize);
    private static native int get_cornerRefinementMaxIterations_0(long nativeObj);
    private static native void set_cornerRefinementMaxIterations_0(long nativeObj, int cornerRefinementMaxIterations);
    private static native double get_cornerRefinementMinAccuracy_0(long nativeObj);
    private static native void set_cornerRefinementMinAccuracy_0(long nativeObj, double cornerRefinementMinAccuracy);
    private static native int get_markerBorderBits_0(long nativeObj);
    private static native void set_markerBorderBits_0(long nativeObj, int markerBorderBits);
    private static native int get_perspectiveRemovePixelPerCell_0(long nativeObj);
    private static native void set_perspectiveRemovePixelPerCell_0(long nativeObj, int perspectiveRemovePixelPerCell);
    private static native double get_perspectiveRemoveIgnoredMarginPerCell_0(long nativeObj);
    private static native void set_perspectiveRemoveIgnoredMarginPerCell_0(long nativeObj, double perspectiveRemoveIgnoredMarginPerCell);
    private static native double get_maxErroneousBitsInBorderRate_0(long nativeObj);
    private static native void set_maxErroneousBitsInBorderRate_0(long nativeObj, double maxErroneousBitsInBorderRate);
    private static native double get_minOtsuStdDev_0(long nativeObj);
    private static native void set_minOtsuStdDev_0(long nativeObj, double minOtsuStdDev);
    private static native double get_errorCorrectionRate_0(long nativeObj);
    private static native void set_errorCorrectionRate_0(long nativeObj, double errorCorrectionRate);
    private static native float get_aprilTagQuadDecimate_0(long nativeObj);
    private static native void set_aprilTagQuadDecimate_0(long nativeObj, float aprilTagQuadDecimate);
    private static native float get_aprilTagQuadSigma_0(long nativeObj);
    private static native void set_aprilTagQuadSigma_0(long nativeObj, float aprilTagQuadSigma);
    private static native int get_aprilTagMinClusterPixels_0(long nativeObj);
    private static native void set_aprilTagMinClusterPixels_0(long nativeObj, int aprilTagMinClusterPixels);
    private static native int get_aprilTagMaxNmaxima_0(long nativeObj);
    private static native void set_aprilTagMaxNmaxima_0(long nativeObj, int aprilTagMaxNmaxima);
    private static native float get_aprilTagCriticalRad_0(long nativeObj);
    private static native void set_aprilTagCriticalRad_0(long nativeObj, float aprilTagCriticalRad);
    private static native float get_aprilTagMaxLineFitMse_0(long nativeObj);
    private static native void set_aprilTagMaxLineFitMse_0(long nativeObj, float aprilTagMaxLineFitMse);
    private static native int get_aprilTagMinWhiteBlackDiff_0(long nativeObj);
    private static native void set_aprilTagMinWhiteBlackDiff_0(long nativeObj, int aprilTagMinWhiteBlackDiff);
    private static native int get_aprilTagDeglitch_0(long nativeObj);
    private static native void set_aprilTagDeglitch_0(long nativeObj, int aprilTagDeglitch);
    private static native boolean get_detectInvertedMarker_0(long nativeObj);
    private static native void set_detectInvertedMarker_0(long nativeObj, boolean detectInvertedMarker);
    private static native boolean get_useAruco3Detection_0(long nativeObj);
    private static native void set_useAruco3Detection_0(long nativeObj, boolean useAruco3Detection);
    private static native int get_minSideLengthCanonicalImg_0(long nativeObj);
    private static native void set_minSideLengthCanonicalImg_0(long nativeObj, int minSideLengthCanonicalImg);
    private static native float get_minMarkerLengthRatioOriginalImg_0(long nativeObj);
    private static native void set_minMarkerLengthRatioOriginalImg_0(long nativeObj, float minMarkerLengthRatioOriginalImg);
    private static native void delete(long nativeObj);
}
