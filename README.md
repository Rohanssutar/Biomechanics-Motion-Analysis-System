# Biomechanics Motion and Analysis System for Boxing

## Overview

This is a computer vision-powered boxing technique analyzer built with Streamlit that helps user improve their boxing form. The application uses MediaPipe pose estimation to analyze boxing movements in real-time, comparing user techniques against professional reference poses. It provides detailed feedback on form, accuracy scores, and technique improvement suggestions for common boxing punches (jab, cross, hook, uppercut)

## User Preferences

Preffered communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **Interface Design**: Multi-column layout with expandable sidebar for analysis settings
- **Real-time Visualization**: Plotly-based interactive charts for performance metrics and technique analysis
- **Video Processing UI**: Upload interface with progress tracking and ETA display for video analysis
- **Feedback System**: Color-coded accuracy indicators and detailed technique improvement suggestions

### Core Processing Pipeline
- **Pose Estimation**: MediaPipe-based computer vision system optimized for upper body boxing movements
- **Video Processing**: Frame extraciton with configurable frame rates(5-15 FPS) and maximum frame limits for performance
- **Technique Analysis**: Rule-based system comparing user poses against professional reference angles
- **Performance Monitoring**: Built-in timing and memory usage tracking with optimization features

### Data Architecture
- **Pose Data Structure**: Dictionary-based landmark storage with calculated joint angles and body rotation metrics
- **Reference System**: Pre-computed professional boxing pose templates with angle tolerances and scoring weights
- **Analysis Pipeline**: Multi-stage comparison system with weighted scoring for different technique aspects
- **Session Management**: Streamlit session state for maintaining analyze instances and user data

### Performance Optimizations
- **Frame Rate Control**: Adjustable processing speeds to balance accuracy and performance
- **Memory Management**: Frame limiting (150 max frames) and temporary file cleanup
- **Model Complexity**: MediaPipe model complexity set to 1 for optimal speed / accuracy balance
- **Image Processing**: Optimized frame preprocessing for pose detection accuracy

## External Dependencies

### Computer Vision & ML
- **MediaPipe**: Google's machine learning framework for pose estimation and landmark detection
- **OpenCV (cv2)**: Video processing, frame manipulation, and computer vision operations
- **NumPy**: Mathematical operations for pose analysis and angle calculations

### Web Framework & Visualization
- **Streamlit**: Main web application framework with built-in UI components
- **Plotly**: Interactive data visualization for performance charts and analysis results
- **PIL (Pillow)**: Image processing and format conversion utilities

### Data Processing
- **Pandas**: Data structure management for analysis results and performance metrics
- **TempFile/OS**: Temporary file handling for video uploads and processing
- **PSUtil**: System performance monitoring (optional dependency for memory tracking)