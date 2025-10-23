import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional

class PoseEstimator:
    def __init__(self):
        """Initialize mediapipe pose estimation with optimized settings for boxing analysis"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize pose model with optimized settings for speed
        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            model_complexity = 1,
            enable_segmentation = False,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )

        #Define boxing-relevant landmarks (upper body focus)
        self.boxing_landmarks = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'nose': self.mp_pose.PoseLandmark.NOSE,
        }
    
    def estimate_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Estimate pose from a single frame, focusing on upper body landmarks.
        Args:
            frame: Input video frame
        Returns:
            Dictionary containing pose landmarks and calculated angles, or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            #Extract relevant landmarks
            landmarks = {}
            for name, landmark_id in self.boxing_landmarks.items():
                landmark = results.pose_landmarks.landmark[landmark_id]
                landmarks[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            #Calculate boxing-relevant angles
            angles = self._calculate_boxing_angles(landmarks)

            return {
                'landmarks': landmarks,
                'angles': angles,
                'raw_landmarks': results.pose_landmarks
            }
        
        return None
    
    def _calculate_boxing_angles(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate key angles relevant for boxing technique analysis.
        Args:
            landmarks: Dictionary of landmark positions
        Returns:
            Dictionary of calculated angles
        """
        angles = {}

        try:
            # Left arm angles
            if all(key in landmarks for key in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angles['left_elbow_angle'] = self._calculate_angle(
                    landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
                )
            
            # Right arm angles
            if all(key in landmarks for key in ['right_shoulder', 'right_elbow', 'right_wrist']):
                angles['right_elbow_angle'] = self._calculate_angle(
                    landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
                )
            
            # Shoulder angles (arm elevation)
            if all(key in landmarks for key in ['left_hip', 'left_shoulder', 'left_elbow']):
                angles['left_shoulder_angle'] = self._calculate_angle(
                    landmarks['left_hip'], landmarks['left_shoulder'], landmarks['left_elbow']
                )
            
            if all(key in landmarks for key in ['right_hip', 'right_shoulder', 'right_elbow']):
                angles['right_shoulder_angle'] = self._calculate_angle(
                    landmarks['right_hip'], landmarks['right_shoulder'], landmarks['right_elbow']
                )

            # Body alignment (shoulder-hip alignment)
            if all(key in landmarks for key in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                angles['body_rotation'] = self._calculate_body_rotation(landmarks)
            
            # Wrist position relative to shoulder (punch extension)
            if all(key in landmarks for key in ['left_shoulder', 'left_wrist']):
                angles['left_punch_extension'] = self._calculate_punch_extension(
                    landmarks['left_shoulder'], landmarks['left_wrist']
                )

            if all(key in landmarks for key in ['right_shoulder', 'right_wrist']):
                angles['right_punch_extension'] = self._calculate_punch_extension(
                    landmarks['right_shoulder'], landmarks['right_wrist']
                )

        except Exception as e:
            print(f"Error calculating angles: {e}")
        return angles
    
    def _calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """
        Calculate angle between three points.
        Args:
            point1, point2, point3: Points with 'x', 'y' coordinates
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        p1 = np.array([point1['x'], point1['y']])
        p2 = np.array([point2['x'], point2['y']])
        p3 = np.array([point3['x'], point3['y']])

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0) #Handle numerical errors
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _calculate_body_rotation(self, landmarks: Dict) -> float:
        """Calculate body rotation angle based on shoulder-hip alignment."""
        shoulder_vector = np.array([
            landmarks['right_shoulder']['x'] - landmarks['left_shoulder']['x'],
            landmarks['right_shoulder']['y'] - landmarks['left_shoulder']['y']
        ])

        hip_vector = np.array([
            landmarks['right_hip']['x'] - landmarks['left_hip']['x'],
            landmarks['right_hip']['y'] - landmarks['left_hip']['y']
        ])

        # Calculate angle between shoulder and hip lines
        cos_angle = np.dot(shoulder_vector, hip_vector) / (
            np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _calculate_punch_extension(self, shoulder: Dict, wrist: Dict) -> float:
        """Calculate punch extension angle relative to horizontal."""
        vector = np.array([
            wrist['x'] - shoulder['x'],
            wrist['y'] - shoulder['y']
        ])

        #Calculate angle with horizontal (positive x-axis)
        horizontal = np.array([1, 0])
        cos_angle = np.dot(vector, horizontal) / np.linalg.norm(vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def draw_pose(self, frame: np.ndarray, pose_data: Dict, highlight_differences: Dict = None) -> np.ndarray:
        """
        Draw pose landmarks on the frame with boxing-specific annotations.
        Args:
            frame: Input frame
            pose_data: Pose data from estimate_pose
        Returns:
            Frame with pose annotations
        """
        annotated_frame = frame.copy()

        if 'raw_landmarks' in pose_data:
            if highlight_differences:
                # Use red for high differences, yellow for medium, green for good
                landmark_color = (0, 0, 255)
                connection_color = (0, 165, 255)
            else:
                # Default green for reference or good poses
                landmark_color = (0, 255, 0)
                connection_color = (0, 0, 255)

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_data['raw_landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=connection_color, thickness=2)
            )

            # Add angle annotations
            angles = pose_data.get('angles', {})
            landmarks = pose_data.get('landmarks', {})

            # Annotate key angles on the frame
            self._annotate_angles(annotated_frame, angles, landmarks)
        return annotated_frame
    
    def _annotate_angles(self, frame: np.ndarray, angles: Dict, landmarks: Dict):
        """Add angle annotations to the frame."""
        h, w = frame.shape[:2]

        # Annotate elbow angles
        if 'left_elbow_angle' in angles and 'left_elbow' in landmarks:
            pos = (int(landmarks['left_elbow']['x'] * w), int(landmarks['left_elbow']['y'] * h))
            cv2.putText(frame, f"L: {angles['left_elbow_angle']:.0f}°",
                        (pos[0] - 30, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        if 'right_elbow_angle' in angles and 'right_elbow' in landmarks:
            pos = (int(landmarks['right_elbow']['x'] * w), int(landmarks['right_elbow']['y'] * h))
            cv2.putText(frame, f"R: {angles['right_elbow_angle']:.0f}°",
                        (pos[0] + 10, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)