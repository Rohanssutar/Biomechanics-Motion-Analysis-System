import numpy as np
from typing import List, Dict, Tuple
from reference_poses import ReferenceBoxingPoses
import pandas as pd

class BoxingAnalyzer:
    def __init__(self):
        """Initialize boxing analyser with reference poses."""
        self.reference_poses = ReferenceBoxingPoses()

        # Weights for different aspects of techniques
        self.angle_weights = {
            'left_elbow_angle': 1.0,
            'right_elbow_angle': 1.0,
            'left_shoulder_angle': 0.8,
            'right_shoulder_angle': 0.8,
            'body_rotation': 0.6,
            'left_punch_extension': 0.7,
            'right_punch_extension': 0.7
        }

        # Tolerance levels for angle comparison (degrees)
        self.tolerance_levels = {
            'excellent': 10,
            'good': 20,
            'fair': 30,
            'poor': float('inf')
        }

    def analyze_technique(self, user_poses: List[Dict], technique: str) -> Dict:
        """
        Analyze user's boxing technique against reference poses.
        Args:
            user_poses: List of pose dictionaries from pose estimation
            technique: Boxing technique name (jab, cross, hook, uppercut)
        Returns: 
            Analysis results with accuracy scores and feedback
        """
        # Get reference pose for the technique
        reference = self.reference_poses.get_reference_pose(technique)
        if not reference:
            raise ValueError(f"Unknown technique: {technique}")
        
        # Extract angles from user poses
        user_angles_sequence = self._extract_angles_sequence(user_poses)
        if not user_angles_sequence:
            return {
                'overall_accuracy': 0,
                'joint_accuracies': {},
                'feedback': {'error': ['No valid poses detected in the video']},
                'frame_scores': []
            }
        
        # Find the best match frame(s)
        frame_scores = self._calculate_frame_scores(user_angles_sequence, reference['angles'])

        # Get the best frame for detailed analysis
        best_frame_idx = np.argmax(frame_scores)
        best_user_angles = user_angles_sequence[best_frame_idx]

        # Calculate joint specific accuracies
        joint_accuracies = self._calculate_joint_accuracies(best_user_angles, reference['angles'])

        # Calculate overall accuracy
        overall_accuracy = self._calculate_overall_accuracy(joint_accuracies, self.angle_weights)

        # Generate detailed feedback
        feedback = self._generate_feedback(best_user_angles, reference, joint_accuracies)

        return {
            'overall_accuracy': overall_accuracy,
            'joint_accuracies': joint_accuracies,
            'feedback': feedback,
            'frame_scores': frame_scores,
            'best_frame_idx': best_frame_idx,
            'reference_used': technique,
            'total_frames_analyzed': len(user_angles_sequence)
        }
    
    def _extract_angles_sequence(self, user_poses: List[Dict]) -> List[Dict]:
        angles_sequence = []

        for pose in user_poses:
            if pose and 'angles' in pose:
                angles = pose['angles']
                # Only include poses with sufficient angle data
                if len(angles) >= 3:
                    angles_sequence.append(angles)
        
        return angles_sequence
    
    def _calculate_frame_scores(self, user_angles_sequence: List[Dict], reference_angles: Dict) -> List[float]:
        frame_scores = []

        for user_angles in user_angles_sequence:
            joint_accuracies = self._calculate_joint_accuracies(user_angles, reference_angles)
            overall_score = self._calculate_overall_accuracy(joint_accuracies, self.angle_weights)
            frame_scores.append(overall_score)

        return frame_scores
    
    def _calculate_joint_accuracies(self, user_angles: Dict, reference_angles: Dict) -> Dict[str, float]:
        joint_accuracies = {}

        for angle_name, reference_value in reference_angles.items():
            if angle_name in user_angles:
                user_value = user_angles[angle_name]
                accuracy = self._calculate_angle_accuracy(user_value, reference_value)
                joint_accuracies[angle_name] = accuracy
            else:
                joint_accuracies[angle_name] = 0

        return joint_accuracies
    
    def _calculate_angle_accuracy(self, user_angle: float, reference_angle: float) -> float:
        """
        Calculate accuracy percentage for a single angle comparison.
        """
        angle_diff = abs(user_angle - reference_angle)

        if angle_diff <= self.tolerance_levels['excellent']:
            return 100 - (angle_diff / self.tolerance_levels['excellent']) * 10
        elif angle_diff <= self.tolerance_levels['good']:
            return 90 - ((angle_diff - self.tolerance_levels['excellent']) / (self.tolerance_levels['good'] - self.tolerance_levels['excellent'])) * 20
        elif angle_diff <= self.tolerance_levels['fair']:
            return 70 - ((angle_diff - self.tolerance_levels['good']) / (self.tolerance_levels['fair'] - self.tolerance_levels['good'])) * 30
        else:
            return max(0, 40 - (angle_diff - self.tolerance_levels['fair']) / 5)
        
    def _calculate_overall_accuracy(self, joint_accuracies: Dict[str, float], weights: Dict[str, float]) -> float:
        # Calculate weights overall accuracy score.
        weighted_sum = 0
        total_weight = 0

        for angle_name, accuracy in joint_accuracies.items():
            weight = weights.get(angle_name, 0.5) 
            weighted_sum += accuracy * weight
            total_weight += weight

        if total_weight == 0:
            return 0 
        
        return weighted_sum / total_weight

    def _generate_feedback(self, user_angle: Dict, reference: Dict, joint_accuracies: Dict) -> Dict[str, List[str]]:
        feedback = {
            'arm_positioning': [],
            'body_alignment': [],
            'punch_alignment': [], 
            'general_tips': []
        }

        reference_angles = reference['angles']

        if 'left_elbow_angle' in joint_accuracies and joint_accuracies['left_elbow_angle'] < 70:
            user_val = user_angle.get('left_elbow_angle', 0)
            ref_val = reference_angles.get('left_elbow_angle', 0)
            if user_val < ref_val - 15:
                feedback['arm_positioning'].append("Left arm is too bent - extend your elbow more during the punch")
            elif user_val > ref_val + 15:
                feedback['arm_positioning'].append("Left arm is overextended - maintain slight bend in elbow")

        if 'right_elbow_angle' in joint_accuracies and joint_accuracies['right_elbow_angle'] < 70:
            user_val = user_angle.get('right_elbow_angle', 0)
            ref_val = reference_angles.get('right_elbow_angle', 0)
            if user_val < ref_val - 15:
                feedback['arm_positioning'].append("Right arm is too bent - extend your elbow more during the punch")
            elif user_val > ref_val + 15:
                feedback['arm_positioning'].append("right arm is overextended - maintain slight bend in elbow")

        # Analyze shoulder positioning
        if 'left_shoulder_angle' in joint_accuracies and joint_accuracies['left_shoulder_angle']:
            feedback['arm_positioning'].append("Left shoulder position needs adjustment - focus on proper shoulder rotation")
        
        if 'right_shoulder_angle' in joint_accuracies and joint_accuracies['right_shoulder_angle']:
            feedback['arm_positioning'].append("Right shoulder position needs adjustment - focus on proper shoulder rotation")

        if 'body_rotation' in joint_accuracies and joint_accuracies['body_rotation']:
            feedback['body_alignment'].append("Body rotation is incorrect - practice proper hip and shoulder coordination")

        # Analyze punch extension
        punch_extensions = ['left_punch_extension', 'right_punch_extension']
        for ext_angle in punch_extensions:
            if ext_angle in joint_accuracies and joint_accuracies[ext_angle] < 60:
                side = 'left' if 'left' in ext_angle else 'right'
                feedback['punch_technique'].append(f"Improve {side} punch extension - focus on straight line to target")

        # Technique specific tips
        technique_tips = reference.get('tips', [])
        feedback['general_tips'].extend(technique_tips)

        # Add encouragement
        overall_avg = np.mean(list(joint_accuracies.values()))
        if overall_avg >= 80:
            feedback['general_tips'].append("Excellent technique! Minor adjustments will perfect your form")
        elif overall_avg >= 60:
            feedback['general_tips'].append("Good foundation! Focus on the highlighted areas for improvement")
        else:
            feedback['general_tips'].append("Practice the basic mechanics - Focus on one element at a time")

        # Remove empty feedback
        feedback = {k: v for k, v in feedback.items() if v}
        return feedback
    
    def get_technique_comparison(self, user_poses: List[Dict]) -> Dict:
        """Compare user's pose against all avaliable techniques to suggest best match."""
        all_techniques = ['jab', 'cross', 'hook', 'uppercut']
        technique_scores = {}

        for technique in all_techniques:
            try:
                result = self.analyze_technique(user_poses, technique)
                technique_scores[technique] = result['overall_accuracy']
            except ValueError:
                technique_scores[technique] = 0

        best_match = max(technique_scores.keys(), key=lambda k: technique_scores[k])

        return {
            'technique_scores': technique_scores,
            'best_match': best_match,
            'confidence': technique_scores[best_match]
        }
    
    def compare_videos(self, ref_poses: List[Dict], user_poses: List[Dict], ref_frames: List, user_frames: List) -> Dict:
        """
        Compare two videos pose by pose for side-by-side analysis.
        
        Args:
            ref_poses: List of pose dictionaries from reference video
            user_poses: List of pose dictionaries from user video
            ref_frames: List of frames from reference video
            user_frames: List of frames from user video
            
        Returns:
            Comparison results with similarities and differences
        """
        # Synchronize video lengths
        min_length = min(len(ref_poses), len(user_poses))
        if min_length == 0:
            return {
                'overall_similarity': 0,
                'frames_compared': 0,
                'differences': {},
                'joint_similarities': {},
                'error': 'No valid poses to compare'
            }
        
        ref_angles_sequence = self._extract_angles_sequence(ref_poses[:min_length])
        user_angles_sequence = self._extract_angles_sequence(user_poses[:min_length])

        if not ref_angles_sequence or not user_angles_sequence:
            return {
                'overall_similarity': 0,
                'frames_compared': min_length,
                'differences': {},
                'joint_similarities': {},
                'error': 'Insufficient angle data for comparison'
            }
        
        # Calculate frame-by-frame similarities
        frame_similarites = []
        joint_similarities_total = {}
        joint_differences = {}

        for i in range(min(len(ref_angles_sequence), len(user_angles_sequence))):
            ref_angles = ref_angles_sequence[i]
            user_angles = user_angles_sequence[i]

            # Calculate similarities
            frame_joint_similarities = {}
            frame_joint_differences = {}

            for angle_name in ref_angles:
                if angle_name in user_angles:
                    ref_val = ref_angles[angle_name]
                    user_val = user_angles[angle_name]

                    angle_diff = abs(ref_val - user_val)
                    similarity = self._calculate_angle_accuracy(user_val, ref_val)

                    frame_joint_similarities[angle_name] = similarity
                    frame_joint_differences[angle_name] = angle_diff

                    if angle_name not in joint_similarities_total:
                        joint_similarities_total[angle_name] = []
                        joint_differences[angle_name] = []

                    joint_similarities_total[angle_name].append(similarity)
                    joint_differences[angle_name].append(angle_diff)

            if frame_joint_similarities:
                weighted_similarity = self._calculate_overall_accuracy(
                    frame_joint_similarities, self.angle_weights
                )
                frame_similarites.append(weighted_similarity)

        avg_joint_similarities = {}
        avg_joint_differences = {}

        for angle_name in joint_similarities_total:
            avg_joint_similarities[angle_name] = np.mean(joint_similarities_total[angle_name])
            avg_joint_differences[angle_name] = np.mean(joint_differences[angle_name])

        overall_similarity = np.mean(frame_similarites) if frame_similarites else 0

        differences = self._generate_video_differences(avg_joint_differences, avg_joint_similarities)

        return {
            'overall_similarity': overall_similarity,
            'frames_compared': min_length,
            'joint_similarities': avg_joint_similarities,
            'differences': differences,
            'frame_similarities': frame_similarites
        }
    
    def _generate_video_differences(self, joint_differences: Dict[str, float], joint_similarites: Dict[str, float]) -> Dict:
        """
        Generate detailed feedback for video comparison.
        
        Args:
            joint_differences: Average angle differences for each joint
            joint_similarities: Average similarities for each joint
            
        Returns:
            Dictionary with detailed differences and suggestions
        """
        differences = {}

        # Define angle-specific feedback
        angle_feedback = {
            'left_elbow_angle': {
                'name': 'Left Elbow',
                'high_diff': 'Left arm extension is significantly different from references',
                'med_diff': 'Left elbow angle needs adjustment',
                'suggestion': 'Focus on matching the reference arm extension during punch'
            },
            'right_elbow_angle': {
                'name': 'Right Elbow',
                'high_diff': 'Right arm extension differs significantly different from reference',
                'med_diff': 'Right elbow angle needs adjustment',
                'suggestion': 'Practice matching the reference right arm position'
            },
            'left_shoulder_angle': {
                'name': 'Left Shoulder',
                'high_diff': 'Left shoulder position is very different from reference',
                'mid_diff': 'Left shoulder alignment needs improvement',
                'suggestion': 'Work on shoulder rotation and positioning' 
            },
            'right_shoulder_angle': {
                'name': 'Right Shoulder',
                'high_diff': 'Right shoulder position differs significantly',
                'mid_diff': 'Right shoulder needs better alignment',
                'suggestion': 'Practice proper shoulder mechanics'
            },
            'body_rotation': {
                'name': 'Body Rotation',
                'high_diff': 'Body Rotation is very different from reference technique',
                'mid_diff': 'Body rotation needs adjustment',
                'suggestion': 'Focus on hip and torso coordination like in reference video'
            },
            'left_punch_extension': {
                'name': 'Left Punch Extension',
                'high_diff': 'Left punch trajectory differs significantly from reference',
                'mid_diff': 'Left punch extension needs work',
                'suggestion': 'Practice punch direction and extension angle'
            },
            'right_punch_extension': {
                'name': 'Right Punch Extension',
                'high_diff': 'Right punch trajectory is different from reference',
                'mid_diff': 'Right punch extension needs improvement',
                'suggestion': 'Match the reference punch angle and direction'
            }
        }

        for angle_name, avg_diff in joint_differences.items():
            if angle_name in angle_feedback:
                feedback_info = angle_feedback[angle_name]
                similarity = joint_similarites.get(angle_name, 0)

                if avg_diff > 30 or similarity < 50:
                    severity = "High"
                    description = feedback_info['high_diff']
                elif avg_diff > 15 or similarity < 70:
                    severity = "Medium"
                    description = feedback_info['mid_diff']
                else:
                    severity = "Low"
                    description = f"{feedback_info['name']} shows minor differences"
                
                differences[angle_name] = {
                    'difference': avg_diff,
                    'similarity': similarity,
                    'severity': severity,
                    'description': description,
                    'suggestion': feedback_info['suggestion']
                }
        return differences