import numpy as np
from typing import Dict, Optional

class ReferenceBoxingPoses:
    def __init__(self):
        """Initialize with pre-computed reference poses for boxing techniques."""
        self.reference_data = self._load_reference_poses()

    def _load_reference_poses(self) -> Dict:
        """
        Load pre-computed reference poses for common boxing techniques.
        These are based on professional boxing form analysis.
        """
        return {
            'jab': {
                'angles': {
                    'left_elbow_angle': 165.0,
                    'right_elbow_angle': 90.0,
                    'left_shoulder_angle': 85.0,
                    'right_shoulder_angle': 110.0,
                    'body_rotation': 15.0,
                    'left_punch_extension': 5.0,
                    'right_punch_extension': 45.0,
                },
                'description': 'Lead hand straight punch',
                'tips': [
                    "Keep your jab straight and snap it back quickly",
                    "Rotate your fist palm down on impact",
                    "Keep your right hand up to guard your chin",
                    "Step forward slightly with your lead foot"
                ],
                'key_points': ['speed', 'straight_line', 'quick_retraction']
            },
            
            'cross': {
                'angles': {
                    'left_elbow_angle': 90.0,
                    'right_elbow_angle': 170.0,
                    'left_shoulder_angle': 110.0,
                    'right_shoulder_angle': 80.0,
                    'body_rotation': 45.0,
                    'left_punch_extension': 45.0,
                    'right_punch_extension': 8.0 
                },
                'description': 'Rear hand straight punch',
                'tips': [
                    "Generate power from your rear foot and hip rotation",
                    "Keep your lead hand up for protection",
                    "Rotate your shoulders and hips together",
                    "Aim straight through the target"
                ],
                'key_points': ['power', 'hip_rotation', 'straight_trajectory']
            },

            'hook': {
                'angles': {
                    'left_elbow_angle': 90.0,
                    'right_elbow_angle': 85.0,
                    'left_shoulder_angle': 90.0,
                    'right_shoulder_angle': 120.0,
                    'body_rotation': 60.0,
                    'left_punch_extension': 90.0,
                    'right_punch_extension': 45.0 
                },
                'description': 'Lead hand circular punch',
                'tips': [
                    "Keep your elbow at 90 degrees throughout the punch",
                    "Pivot on your lead foot and rotate your body",
                    "Keep your hook parallel to the ground",
                    "Generate power from hip rotation, not just arm swing"
                ],
                'key_points': ['circular_motion', 'elbow_angle', 'body_pivot']
            },

            'uppercut': {
                'angles': {
                    'left_elbow_angle': 85.0,
                    'right_elbow_angle': 90.0,
                    'left_shoulder_angle': 130.0,
                    'right_shoulder_angle': 110.0,
                    'body_rotation': 25.0,
                    'left_punch_extension': 135.0,
                    'right_punch_extension': 45.0 
                },
                'description': 'Upward driving punch',
                'tips': [
                    "Drop your shoulder and drive upward with your legs",
                    "Keep your elbow close to your body during the drive",
                    "Generate power from bending and extending your knees",
                    "Aim to come up under your opponent's guard"
                ],
                'key_points': ['upward_drive', 'leg_power', 'close_elbow']
            }
        }
    
    def get_reference_pose(self, technique: str) -> Optional[Dict]:
        """
        Get reference pose data for a specific boxing technique.
        Args:
            technique: Name of the boxing technique
        Returns:
            Reference pose data or None if technique not found
        """
        return self.reference_data.get(technique.lower())
    
    def get_all_techniques(self) -> list:
        """Get a list of all available boxing techniques."""
        return list(self.reference_data.keys())
    
    def get_technique_description(self, technique: str) -> str:
        ref_data = self.get_reference_pose(technique)
        return ref_data['description'] if ref_data else "Unknown technique"
    
    def get_technique_tips(self, technique: str) -> list:
        ref_data = self.get_reference_pose(technique)
        return ref_data['tips'] if ref_data else []
    
    def compare_angles(self, user_angles: Dict, techniques: str) -> Dict:
        """
        Compare user angles against reference technique.
        Args:
            user_angles: Dictionary of user's measured angles
            techniques: Reference technique to compare against
        Returns:
            Comparison results with differences and scores
        """
        references = self.get_reference_pose(techniques)
        if not references:
            return {'error': f'Unknown Technique: {techniques}'}
        
        ref_angles = references['angles']
        comparison = {}

        for angle_name, ref_value in ref_angles.items():
            if angle_name in user_angles:
                user_value = user_angles[angle_name]
                difference = abs(user_value - ref_value)

                # Calculate score
                if difference <= 10:
                    score = 100 - difference
                elif difference <= 30:
                    score = 90 - (difference - 10) * 2
                else:
                    score = max(0, 50 - (difference - 30))

                comparison[angle_name] = {
                    'user_value': user_value,
                    'reference_value': ref_value,
                    'difference': difference,
                    'score': score,
                    'status': self._get_angle_status(difference)
                }
        return comparison
    
    def _get_angle_status(self, difference: float) -> str:
        """Get status description based on angle difference."""
        if difference <= 5:
            return 'excellent'
        elif difference <= 15:
            return 'good'
        elif difference <= 25:
            return 'fair'
        else:
            return 'needs_improvement'
        
    def get_optimal_range(self, technique: str, angle_name: str) -> Dict:
        """
        Get optimal range for a specific angle in a technique
        Args: 
            technique: Boxing technique name
            angle_name: Specific angle name
        Returns:
            Dictionary with min, max, and optimal values
        """
        reference = self.get_reference_pose(technique)
        if not reference or angle_name not in reference['angles']:
            return {'error': 'Invalid technique or angle name'}
        
        optimal_value = reference['angles'][angle_name]

        # Define acceptable ranges based on angle type
        if 'elbow' in angle_name:
            tolerance = 15
        elif 'shoulder' in angle_name:
            tolerance = 20
        elif 'extension' in angle_name:
            tolerance = 25
        else:
            tolerance = 10

        return {
            'optimal': optimal_value,
            'min_acceptable': max(0, optimal_value - tolerance),
            'max_acceptable': min(180, optimal_value + tolerance),
            'tolerance': tolerance
        }