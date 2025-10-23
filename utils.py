import numpy as np
import cv2 
from typing import List, Dict, Tuple, Optional
import streamlit as st
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} exceuted in {execution_time:.2f} seconds")
        return result
    return wrapper

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}

    def start_timer(self, operation: str):
        self.timings[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        if operation in self.timings:
            duration = time.time() - self.timings[operation]
            del self.timings[operation]
            return duration
        return 0
    
    def get_memory_info(self) -> Dict:
        try:
            import psutil
            process = psutil.Process()
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'memory_percent': process.memory_percent()
            }
        except ImportError:
            return {'memory_mb': 0, 'memory_percent': 0}
        
def optimize_image_for_pose_detection(image: np.ndarray) -> np.ndarray:
    """
    Optimize image for better pose detection performance.
    Args:
        image: Input image
    Returns:
        Optimized image
    """
    # Ensure image is in correct format
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    
    #Resize if too large (maintain aspect ratio)
    height, width = image.shape[:2]
    max_dimension = 1280

    if max(height, width) > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    #Enhance constrast for better detection
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge and convert back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

def calculate_angle_between_points(p1: Tuple[float, float],
                                p2: Tuple[float, float],
                                p3: Tuple[float, float]) -> float:
    """
    Calculate angle between three points using vectorized operations.
    Args:
        p1, p2, p3: Points as (x, y) tuples
    Returns:
        Angle in degrees
    """
    # Convert to numpy arrays for vectorized operations
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    p3_arr = np.array(p3)

    # Calculate vectors
    v1 = p1_arr - p2_arr
    v2 = p3_arr - p2_arr

    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)

def batch_calculate_angles(points_batch: List[List[Tuple[float, float]]]) -> List[float]:
    """Calculate angles for multiple points sets using vectorized operations."""

    angles = []
    for points in points_batch:
        if len(points) == 3:
            angle = calculate_angle_between_points(points[0], points[1], points[2])
            angles.append(angle)
        else:
            angles.append(0.0)
    
    return angles

def smooth_angle_sequence(angles: List[float], window_size: int = 5) -> List[float]:
    """
    Apply smoothing to angle sequence to reduce noise.
    Args:
        angles: List of angle values
        window_size: Size of smoothing window
    Returns:
        Smoothed angle sequence
    """
    if len(angles) < window_size:
        return angles
    
    smoothed = []
    half_window = window_size // 2

    for i in range(len(angles)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(angles), i + half_window + 1)
        window_angles = angles[start_idx:end_idx]
        smoothed_angle = np.mean(window_angles)
        smoothed.append(smoothed_angle)
    return smoothed

def detect_key_frames(angle_sequences: Dict[str, List[float]], technique: str) -> List[int]:
    if not angle_sequences:
        return []
    
    primary_angles = {
        'jab': 'left_elbow_angle',
        'cross': 'right_elbow_angle', 
        'hook': 'left_elbow_angle',
        'uppercut': 'left_elbow_angle'
    }

    primary_angle = primary_angles.get(technique, 'left_elbow_angle')

    if primary_angle not in angle_sequences:
        primary_angle = next(iter(angle_sequences.keys()))

    angles = angle_sequences[primary_angle]

    if len(angles) < 3:
        return list(range(len(angles)))
    
    derivatives = np.diff(angles)
    key_frames = []

    for i in range(1, len(derivatives) - 1):
        if (derivatives[i] > derivatives[i-1] and derivatives[i] > derivatives[i+1]) or \
           (derivatives[i] > derivatives[i-1] and derivatives[i] > derivatives[i+1]):
            key_frames.append(i)

    key_frames = [0] + key_frames + [len(angles) - 1]
    key_frames = sorted(list(set(key_frames)))

    return key_frames

def create_accuracy_visualization(joint_accuracies: Dict[str, float]) -> Dict:
    # Prepare data for plotting
    joints = list(joint_accuracies.keys())
    accuracies = list(joint_accuracies.values())

    colors = []
    for accuracy in accuracies:
        if accuracy >= 80:
            colors.append('green')
        elif accuracy >= 60:
            colors.append('orange')
        else:
            colors.append('red')
    
    return {
        'joints': joints,
        'accuracies': accuracies,
        'colors': colors,
        'average_accuracy': np.mean(accuracies) if accuracies else 0
    }

def format_feedback_message(feedback: Dict[str, List[str]]) -> str:
    formatted_messages = []

    category_icons = {
        'arm_positioning': '💪🏻',
        'body_alignment': '🏃🏻',
        'punch_technique': '👊🏻',
        'general_tips': '💡'
    }

    for category, messages in feedback.items():
        if messages:
            icon = category_icons.get(category, '•')
            category_title = category.replace('_', ' ').title()
            formatted_messages.append(f"\n{icon} **{category_title}:**")

            for message in messages:
                formatted_messages.append(f"   • {message}")
    
    return '\n'.join(formatted_messages) if formatted_messages else "Great technique! Keep practicing!"

def validate_pose_data(pose_data: Dict) -> bool:
    if not pose_data:
        return False
    
    required_fields = ['landmarks', 'angles']
    if not all(field in pose_data for field in required_fields):
        return False
    
    landmarks = pose_data['landmarks']
    required_landmarks = ['left_shoulder', 'right_right_shoulder', 'left_elbow', 'right_elbow']

    if not all(landmark in landmarks for landmark in required_landmarks):
        return False
    
    angles = pose_data['angles']
    if len(angles) < 2:
        return False
    
    for angle in angles.values():
        if not (0 <= angle <= 180):
            return False
    
    return True

@st.cache_data
def load_cached_reference_poses():
    from reference_poses import ReferenceBoxingPoses
    return ReferenceBoxingPoses()

def display_progress_with_eta(current: int, total: int, start_time: float, operation: str = "Processing"):
    if total <= 0:
        return
    
    progress = current / total
    elapsed_time = time.time() - start_time

    if current > 0 and progress > 0:
        eta = (elapsed_time / progress) - elapsed_time
        eta_text = f" (ETA: {eta:.0f}s)" if eta > 1 else ""
    else:
        eta_text = ""

    progress = 0 if total == 0 else current / total
    progress = min(max(progress, 0), 1)
    st.progress(progress, text=f"{operation}: {current}/{total}{eta_text}")