import numpy as np
from src.RobotModels.ContinuumRobotModel import *

class ConstantCurvature(ContinuumRobotModel):
    """
    Continuum robot model with constant curvature segments using Clarke coordinates.
    
    The robot is modeled as a series of constant curvature arcs in 2D or 3D.
    Each segment has a fixed length and variable tendon pulls (cx, cy).
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """
        # Initialize the continuum robot.
        
        # Args:
        #     num_segments: Number of segments in the robot
        #     segment_length: Length of each segment
        #     max_tendon_pull: Maximum tendon pull for each segment
        #     min_tendon_pull: Minimum tendon pull for each segment
        #     tendon_offset: Distance of tendons from center axis
        #     debug_points_per_segment: Number of points to generate per segment for debugging
        """

        super().__init__(**kwargs)
        
    
    def forward_kinematics(self, clarke_coords: np.ndarray, tip_at_origin=True):
        """
        Compute forward kinematics for the continuum robot.
        
        Args:
            clarke_coords: Array of shape (num_segments * 2) with Clarke coordinates
            
        Returns:
            endpoints: Array of shape (num_segments, 3) with endpoints of each segment
            shape_points: Array of shape (sum of lengths/resolution, 3) with points along the shape of the robot
        """
        if len(clarke_coords) != self.num_segments * 2:
            raise ValueError(f"Expected {self.num_segments * 2} Clarke coordinates, got {len(clarke_coords)}")
        
        # Initialize arrays
        endpoints = np.zeros((self.num_segments, 3))
        shape_points = []
        
        # Calculate total length for base positioning
        total_length = sum(self.segment_lengths)
        
        # Start with base at (0,0,-total_length) so tip is at origin when straight
        current_transform = np.eye(4)
        if tip_at_origin:
            current_transform[2, 3] = -total_length  # Move base to -total_length in z
        
        for i in range(self.num_segments):
            # Extract Clarke coordinates for this segment
            cx = clarke_coords[i * 2]
            cy = clarke_coords[i * 2 + 1]
            
            # Get segment parameters
            length = self.segment_lengths[i]
            offset = self.tendon_offset[i]
            
            # Compute segment transformation
            segment_transform = seg_clark_to_task(length, offset, cx, cy)
            
            # Update current transformation
            current_transform = current_transform @ segment_transform
            
            # Store endpoint (translation part of transformation matrix)
            endpoints[i] = current_transform[:3, 3]
            
            # Generate shape points along this segment
            num_points = max(1, int(length / self.points_resolution))
            for j in range(num_points):
                # Scale the segment transformation by the fraction of the segment
                t = j / (num_points - 1) if num_points > 1 else 0
                
                # Create scaled transformation for intermediate point
                scaled_transform = seg_clark_to_task(length * t, offset, cx * t, cy * t)
                
                # Apply to the previous transformation
                point_transform = current_transform @ np.linalg.inv(segment_transform) @ scaled_transform
                
                # Extract point position
                point = point_transform[:3, 3]
                shape_points.append(point)
        
        return endpoints, np.array(shape_points), point_transform
    
    def forward_kinematics_from_base_SE3(self, clarke_coords: np.ndarray, base_position: np.ndarray, 
                                   base_orientation: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics from an arbitrary base pose and return the tips SE3.
        
        Args:
            clarke_coords: Array of shape (num_segments * 2) with Clarke coordinates
            base_position: Base position [x, y, z]
            base_orientation: Base orientation as 3x3 rotation matrix
            
        Returns:
            endpoints: Array of shape (num_segments, 3) with endpoints of each segment
            shape_points: Array of shape (sum of lengths/resolution, 3) with points along the shape
        """
        if len(clarke_coords) != self.num_segments * 2:
            raise ValueError(f"Expected {self.num_segments * 2} Clarke coordinates, got {len(clarke_coords)}")
        
        # Start with base transformation
        current_transform = np.eye(4)
        current_transform[:3, :3] = base_orientation
        current_transform[:3, 3] = base_position
        
        for i in range(self.num_segments):
            # Extract Clarke coordinates for this segment
            cx = clarke_coords[i * 2]
            cy = clarke_coords[i * 2 + 1]
            
            # Get segment parameters
            length = self.segment_lengths[i]
            offset = self.tendon_offset[i]
            
            # Compute segment transformation
            segment_transform = seg_clark_to_task(length, offset, cx, cy)
            
            # Update current transformation
            current_transform = current_transform @ segment_transform

            # Generate shape points along this segment
            num_points = max(1, int(length / self.points_resolution))
            for j in range(num_points):
                # Scale the segment transformation by the fraction of the segment
                t = j / (num_points - 1) if num_points > 1 else 0
                
                # Create scaled transformation for intermediate point
                scaled_transform = seg_clark_to_task(length * t, offset, cx * t, cy * t)
                
                # Apply to the previous transformation
                point_transform = current_transform @ np.linalg.inv(segment_transform) @ scaled_transform
        
        return point_transform
        
    
    def compute_base_pose_for_tip_pose(self, clarke_coords: np.ndarray, desired_tip_position: np.ndarray,
                                      desired_tip_orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the required base pose to achieve a desired tip pose with given Clarke coordinates.
        
        Args:
            clarke_coords: Array of shape (num_segments * 2) with Clarke coordinates
            desired_tip_position: Desired tip position [x, y, z]
            desired_tip_orientation: Desired tip orientation as 3x3 rotation matrix
            
        Returns:
            base_position: Required base position [x, y, z]
            base_orientation: Required base orientation as 3x3 rotation matrix
        """
        if len(clarke_coords) != self.num_segments * 2:
            raise ValueError(f"Expected {self.num_segments * 2} Clarke coordinates, got {len(clarke_coords)}")
        
        # Compute robot transformation from base to tip (without base pose)
        robot_transform = np.eye(4)
        
        for i in range(self.num_segments):
            cx = clarke_coords[i * 2]
            cy = clarke_coords[i * 2 + 1]
            length = self.segment_lengths[i]
            offset = self.tendon_offset[i]
            
            segment_transform = seg_clark_to_task(length, offset, cx, cy)
            robot_transform = robot_transform @ segment_transform
        
        # Desired tip transformation
        desired_tip_transform = np.eye(4)
        desired_tip_transform[:3, :3] = desired_tip_orientation
        desired_tip_transform[:3, 3] = desired_tip_position
        
        # Compute required base transformation
        # desired_tip = base_transform @ robot_transform
        # base_transform = desired_tip @ inv(robot_transform)
        robot_transform_inv = np.linalg.inv(robot_transform)
        required_base_transform = desired_tip_transform @ robot_transform_inv
        
        # Extract base position and orientation
        base_position = required_base_transform[:3, 3]
        base_orientation = required_base_transform[:3, :3]
        
        return base_position, base_orientation
    
    
    def get_full_transform(self, clarke_coords: np.ndarray) -> np.ndarray:
        """
        Get the full transformation matrix from base to tip.
        
        Args:
            clarke_coords: Array of shape (num_segments * 2) with Clarke coordinates
            
        Returns:
            transform: 4x4 transformation matrix from base to tip
        """
        if len(clarke_coords) != self.num_segments * 2:
            raise ValueError(f"Expected {self.num_segments * 2} Clarke coordinates, got {len(clarke_coords)}")
        
        # Calculate total length for base positioning
        total_length = sum(self.segment_lengths)
        
        # Start with base at (0,0,-total_length) so tip is at origin when straight
        current_transform = np.eye(4)
        current_transform[2, 3] = -total_length  # Move base to -total_length in z
        
        for i in range(self.num_segments):
            cx = clarke_coords[i * 2]
            cy = clarke_coords[i * 2 + 1]
            length = self.segment_lengths[i]
            offset = self.tendon_offset[i]
            
            segment_transform = seg_clark_to_task(length, offset, cx, cy)
            current_transform = current_transform @ segment_transform
        
        return current_transform
        
    


        
        