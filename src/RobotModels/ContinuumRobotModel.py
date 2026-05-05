import numpy as np
from enum import Enum


class ReferenceFrame(Enum):
    """Reference frame options for forward kinematics."""
    BASE = "base"      # Base frame (default)
    TIP = "tip"        # Tip frame
    WORLD = "world"    # World frame (with custom transform)


def seg_clark_to_task(length, offset, cx, cy):
    """
    Compute transformation matrix for a constant curvature segment using Clark coordinates.
    
    Args:
        length: Length of the segment
        offset: Tendon offset from center axis
        cx, cy: Clark coordinates (tendon pulls)
        
    Returns:
        4x4 transformation matrix from segment base to segment tip
    """
    theta = np.sqrt(cx**2 + cy**2) / offset
    if np.isclose(theta, 0.0):
        # Straight segment
        fk_mat = np.eye(4)
        fk_mat[2, 3] = length
    else:
        # Curved segment
        phi = np.atan2(cy, cx)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Rotation matrix components
        r11 = cos_phi * cos_phi * (cos_theta - 1) + 1
        r12 = sin_phi * cos_phi * (cos_theta - 1)
        r13 = cos_phi * sin_theta
        r21 = sin_phi * cos_phi * (cos_theta - 1)
        r22 = cos_phi * cos_phi * (1 - cos_theta) + cos_theta
        r23 = sin_phi * sin_theta
        r31 = -cos_phi * sin_theta
        r32 = -sin_phi * sin_theta
        r33 = cos_theta
        
        # Translation components
        tx = cos_phi * (1 - cos_theta) * length / theta
        ty = sin_phi * (1 - cos_theta) * length / theta
        tz = length * sin_theta / theta
        
        fk_mat = np.array([
            [r11, r12, r13, tx],
            [r21, r22, r23, ty],
            [r31, r32, r33, tz],
            [0, 0, 0, 1]
        ])
        
    return np.nan_to_num(fk_mat)


def quaternion_to_rotation_matrix(self, quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quaternion: Quaternion [w, x, y, z] (normalized)
        
    Returns:
        rotation_matrix: 3x3 rotation matrix
    """
    w, x, y, z = quaternion
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
def rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        quaternion: Quaternion [w, x, y, z] (normalized)
    """
    # Implementation of rotation matrix to quaternion conversion
    trace = np.trace(rotation_matrix)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s
    
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)  # Normalize



class ContinuumRobotModel:

    def __init__(
        self,
        num_segments: int = 3,
        segment_lengths: list[float] = [1.0, 1.0, 1.0],
        tendon_offset: list[float] = [0.1, 0.1, 0.1],
        points_resolution: float = 0.1
    ):
        """Constructor for continuum robot model class. This class contains the basic methods required
        to use a continuum robot such as forward kinematics.

        Args:
            num_segments (int, optional): How many segments compose your robot. Defaults to 3.
            segment_length (list[float], optional): Length of each segment. Defaults to [1.0, 1.0, 1.0].
            tendon_offset (list[float], optional): How far from center line each tendon is. Defaults to [0.1, 0.1, 0.1].
            points_resolution (float, optional): Resolution to describe the shape of the robot with discrete points.
                In the forward kinematics function, if the resolution is 0.1 then the robot shape will be returned as a discretized line with each point being
                0.1 distance from eachother. Defaults to 0.1.
        """
        self.num_segments = num_segments
        self.segment_lengths = segment_lengths
        self.tendon_offset = tendon_offset
        self.points_resolution = points_resolution

        self.num_discretized_points = round(sum(self.segment_lengths) / self.points_resolution)
        
        
        self.robot_length = sum(self.segment_lengths)

    def forward_kinematics(self, clarke_coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs forward kinematics of a continuum robot.

        Args:
            clarke_coordinates (np.ndarray): Desired configuration to perform FK on. 
                Example, [0, 0, -0.2, 0.2, 0, 0.5]

        Returns:
            tuple[np.ndarray, np.ndarray]: Endpoints and shape points of robot.
                Endpoints contain the center of the last link for each segment. It is self.num_seg
        """

        raise NotImplementedError("Forward Kinematics - not implemented. Implement in child class")

    def forward_kinematics_from_base(self, clarke_coords: np.ndarray, 
                                     base_position: np.ndarray, 
                                     base_orientation: np.ndarray,
                                     ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics from an arbitrary base pose.
        
        Args:
            clarke_coords: Array of shape (num_segments * 2) with Clark coordinates
            base_position: Base position [x, y, z]
            base_orientation: Base orientation as 3x3 rotation matrix
        Returns:
            endpoints: Array of shape (num_segments, 3) with endpoints of each segment
            shape_points: Array of shape (sum of lengths/resolution, 3) with points along the shape
        """
        endpoints, shapepoints, tip_SE3 = self.forward_kinematics(clarke_coords, tip_at_origin=False)

        current_transform = np.eye(4)
        current_transform[:3, :3] = base_orientation
        current_transform[:3, 3] = base_position

        
        # breakpoint()
        homogeneous_endpoints = np.hstack((endpoints, np.ones((self.num_segments, 1)))).T
        homogeneous_shapepoints = np.hstack((shapepoints, np.ones((self.num_discretized_points, 1)))).T
        
        transformed_endpoints = (current_transform @ homogeneous_endpoints)[:3].T
        transformed_shapepoints = (current_transform @ homogeneous_shapepoints)[:3].T
        transformed_tip = current_transform @ tip_SE3

        return transformed_endpoints, transformed_shapepoints, transformed_tip


    def get_tip_position(self, clarke_coords: np.ndarray) -> np.ndarray:
        """
        Get the tip position of the robot.
        
        Args:
            clarke_coords: Array of shape (num_segments * 2) with Clark coordinates
            
        Returns:
            tip_position: 3D position of the robot tip
        """

        endpoints, _, _ = self.forward_kinematics(clarke_coords)
        return endpoints[-1]     

    def get_tip_orientation(self, clarke_coords: np.ndarray) -> np.ndarray:
        """
        Get the tip orientation of the robot.
        
        Args:
            clark_coords: Array of shape (num_segments * 2) with Clarke coordinates
            
        Returns:
            tip_orientation: 3x3 rotation matrix of the robot tip
        """

        _, _, tip_SE3 = self.forward_kinematics(clarke_coords)
        return tip_SE3[:3, :3]
    

        
        

        

    def get_CR_parameters(self) -> dict:
        """
        Get the robot parameters as a dictionary.
        
        Returns:
            params: Dictionary with robot parameters
        """
        return {
            "num_segments": self.num_segments,
            "segment_length": self.segment_lengths,
            "tendon_offset": self.tendon_offset,
            "points_resolution": self.points_resolution
            # whatever else we want here
        }