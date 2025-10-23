import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from pose_estimator import PoseEstimator
from utils import PerformanceMonitor, optimize_image_for_pose_detection, display_progress_with_eta
import time

class VideoProcessor:
    def __init__(self):
        """Initialize pose estimator"""
        self.pose_estimator = PoseEstimator()
        self.performance_monitor = PerformanceMonitor()

    def process_video(self, video_path: str, frame_rate: int = 8, max_frames: int = 150, video_type: str = "video") -> Tuple[List[Dict], List[np.ndarray]]:
        self.performance_monitor.start_timer(f"process_{video_type}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Failed to open video: {video_path}")
            return [], []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        frame_skip = max(1, int(video_fps / frame_rate))

        poses = []
        frames = []
        frame_count = 0
        processed_count = 0

        progress_placeholder = st.empty()
        start_time = time.time()

        try:
            while cap.isOpened() and processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    optimized_frame = optimize_image_for_pose_detection(frame)

                    pose_data = self.pose_estimator.estimate_pose(optimized_frame)
                    poses.append(pose_data)
                    frames.append(frame.copy())
                    processed_count += 1

                    with progress_placeholder:
                        display_progress_with_eta(
                            processed_count,
                            min(max_frames, total_frames // frame_skip),
                            start_time,
                            f"Processing {video_type} video"
                        )
                frame_count += 1

                if processed_count >= max_frames:
                    break
        
        finally:
            cap.release()
            progress_placeholder.empty()

        processing_time = self.performance_monitor.end_timer(f"process {video_type}")

        valid_poses = []
        valid_frames = []

        for pose, frame in zip(poses, frames):
            if pose is not None:
                valid_poses.append(pose)
                valid_frames.append(frame)
        st.success(f"✅Processed {len(valid_poses)} valid frames from {video_type} video in {processing_time:.2f}s")
        return valid_poses, valid_frames
    
    def extract_video_info(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {'error': 'Cannot open video file'}

        try:
            info = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            return info
        finally:
            cap.release()

    def create_side_by_side_video(self, frames1: List[np.ndarray], frames2: List[np.ndarray], poses1: List[Dict], poses2: List[Dict], output_path: str) -> bool:
        """
        Create a side-by-side comparison video with pose overlays.
        
        Args:
            frames1: First video frames
            frames2: Second video frames
            poses1: First video poses
            poses2: Second video poses
            output_path: Path for output video
            
        Returns:
            Success status
        """
        if not frames1 or not frames2:
            return False
        
        min_frames = min(len(frames1), len(frames2))
        frame_height = max(frames1[0].shape[0] + frames2[0].shape[0])
        frame_width = frames1[0].shape[1] + frames2.shape[1]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 8.0, (frame_width, frame_height))

        try:
            for i in range(min_frames):
                frame1 = frames1[i]
                frame2 = frames2[i] if i < len(frames2) else np.zeros_like(frame1)

                # Resize frames
                if frame1.shape[0] != frame_height:
                    frame1 = cv2.resize(frame1, (frame1.shape[1], frame_height))
                if frame2.shape[0] != frame_height:
                    frame2 = cv2.resize(frame2, (frame2.shape[1], frame_height))

                if i < len(poses1) and poses1[i]:
                    frame1 = self.pose_estimator.draw_pose(frame1, poses1[i])
                if i < len(poses2) and poses2[i]:
                    frame2 = self.pose_estimator.draw_pose(frame2, poses2[i])

                combined_frame = np.hstack([frame1, frame2])

                out.write(combined_frame)
            return True
        
        except Exception as e:
            st.error(f"Error creating comparison video: {e}")
            return False
        
        finally:
            out.release()

    def get_frame_At_time(self, video_path: str, time_seconds: float) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened:
            return None
        
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
            ret, frame = cap.read()

            if ret:
                return frame
            return None
        
        finally:
            cap.release()

    def extract_key_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened:
            return []
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                num_frames = total_frames
            
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            key_frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    key_frames.append(frame)
            return key_frames
        
        finally:
            cap.release()