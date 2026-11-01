# Human Height Measurement Application – Computer Vision

An Android-based mobile application designed to estimate human body height automatically using Computer Vision technology. The application allows users to capture photos directly using a smartphone camera or upload full-body images from the gallery to obtain estimated height measurements.

## About the Project

This project focuses on developing a mobile-based human height measurement system that provides a practical alternative to conventional manual measurement methods.

The application processes full-body images by combining ArUco Marker detection as a measurement scale reference, Perspective Correction using Homography and RANSAC, and MediaPipe Pose Landmarker to identify body landmarks.

The system estimates body height by identifying the highest point of the head and the lowest point of the feet after image correction and scale calibration. The estimated measurement is then displayed through the Android application.

## Key Features

- **Camera Capture:** Capture full-body images directly using a smartphone camera.
- **Gallery Upload:** Select existing full-body images from the device gallery.
- **Image Processing:** Process images before performing height estimation.
- **ArUco Marker Detection:** Detect reference markers for measurement scale calibration.
- **Perspective Correction:** Correct image perspective using Homography and RANSAC.
- **Pose Estimation:** Detect 33 human body landmarks using MediaPipe Pose Landmarker.
- **Automatic Height Estimation:** Estimate body height based on head and foot positions.
- **Measurement Results:** Display the estimated height through the application interface.

## Application Demo

The following videos demonstrate the application using two different subject categories.

### 1. Children's Height Measurement
Demonstration of the application when measuring a child's height using a full-body image.

**Video Demo:**  
[![Watch Height Measurement Demo](https://drive.google.com/thumbnail?id=1oevGNQD3EXG4kVcBgvAgL0DzPlkeMxer&sz=w1000)](https://drive.google.com/file/d/1oevGNQD3EXG4kVcBgvAgL0DzPlkeMxer/view)

[▶ Watch Full Video](https://drive.google.com/file/d/1oevGNQD3EXG4kVcBgvAgL0DzPlkeMxer/view)

### 2. Adult Height Measurement
Demonstration of the application when measuring an adult's height using a full-body image.

**Video Demo:**  
[[Insert Adult Demo Video Link Here]](https://drive.google.com/file/d/1aijl-3IPbvONHYAWNAtANWZMed8CZoh_/view?usp=sharing)

## Technologies Used

- Kotlin
- Android Studio
- OpenCV
- MediaPipe Pose Landmarker
- ArUco Marker
- Homography
- RANSAC
- Computer Vision
- Image Processing

## Evaluation Results

The system was evaluated using 75 images from 15 subjects with five different posture variations.

| Evaluation | MAE |
|---|---:|
| Before Perspective and Scale Correction | 31.92 cm |
| After Perspective and Scale Correction | 4.31 cm |

The results show that applying corrected measurement planes and scale calibration reduces the overall Mean Absolute Error (MAE) from 31.92 cm to 4.31 cm.

## Project Objectives

- Develop an Android application for automated human height estimation.
- Implement Computer Vision techniques for image-based measurement.
- Integrate perspective correction and pose estimation.
- Provide a practical and accessible mobile measurement solution.

## Developer

Developed as an academic project using Android development and Computer Vision techniques.
