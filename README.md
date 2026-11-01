# Human Height Measurement – Computer Vision

An Android mobile application developed to estimate human body height automatically from full-body images using Computer Vision techniques.

The application allows users to capture images directly using a smartphone camera or upload images from the gallery. It processes the images through ArUco Marker detection for scale reference, Perspective Correction using Homography and RANSAC, and MediaPipe Pose Landmarker to identify body landmarks and estimate human height.

### Key Features
- Capture images using a smartphone camera.
- Upload full-body images from the gallery.
- ArUco Marker detection for measurement scale reference.
- Perspective correction to reduce camera perspective distortion.
- Human pose detection using MediaPipe Pose Landmarker.
- Automatic height estimation based on head and foot positions.
- Display estimated height measurement results.

### Technologies
- Kotlin
- Android Studio
- OpenCV
- MediaPipe Pose Landmarker
- ArUco Marker
- Computer Vision
- Homography & RANSAC

### Evaluation
Tested on 75 images from 15 subjects with five posture variations.

- MAE before correction: 31.92 cm
- MAE after correction: 4.31 cm

The evaluation demonstrates the improvement achieved through corrected measurement planes and scale calibration.
