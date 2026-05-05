import numpy as np



def create_SE3_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 SE3 transformation matrix from a rotation matrix and translation vector.
    
    Args:
        rotation (np.ndarray): A 3x3 rotation matrix.
        translation (np.ndarray): A 3-element translation vector.
        
    Returns:
        np.ndarray: A 4x4 SE3 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def apply_SO3_to_points(R: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 SO3 rotation matrix to an array of 3D points.
    
    Args:
        R (np.ndarray): A 3x3 rotation matrix.
        points (np.ndarray): An Nx3 array of 3D points.
        
    Returns:
        np.ndarray: An Nx3 array of rotated 3D points.
    """
    return (R @ points.T).T

def apply_SE3_to_vector(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 SE3 transformation matrix to a 3D vector.
    
    Args:
        T (np.ndarray): A 4x4 SE3 transformation matrix.
        v (np.ndarray): A 3-element vector.
        
    Returns:
        np.ndarray: The transformed 3-element vector.
    """
    v_homogeneous = np.ones(4)
    v_homogeneous[:3] = v
    v_transformed = T @ v_homogeneous
    return v_transformed[:3]


def apply_SE3_to_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 SE3 transformation matrix to an array of 3D points.
    
    Args:
        T (np.ndarray): A 4x4 SE3 transformation matrix.
        points (np.ndarray): An Nx3 array of 3D points.
        
    Returns:
        np.ndarray: An Nx3 array of transformed 3D points.
    """
    num_points = points.shape[0]
    points_homogeneous = np.hstack((points, np.ones((num_points, 1))))
    points_transformed = (T @ points_homogeneous.T).T
    return points_transformed[:, :3]

def change_base_frame(new_base, points):
    """
    Change the base frame of a set of points given a new base transformation.
    
    Args:
        new_base (np.ndarray): A 4x4 SE3 transformation matrix representing the new base frame.
        points (np.ndarray): An Nx3 array of 3D points in the original frame.
        
    Returns:
        np.ndarray: An Nx3 array of points in the new base frame.
    """
    # Assume old base of all frames is [0, 0, -3]
    old_base = np.eye(4)
    old_base[:3, 3] = np.array([0, 0, -3])
    old_base_inv = np.linalg.inv(old_base)
    transform = new_base @ old_base_inv
    transformed_points = apply_SE3_to_points(transform, points)
    return transformed_points