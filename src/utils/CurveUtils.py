import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, Dict


def compute_arc_length(points: np.ndarray) -> float:
    """
    Compute the arc length of a curve defined by points.
    
    Args:
        points: Nx3 array of points defining the curve
        
    Returns:
        arc_length: Total arc length of the curve
    """
    if len(points) < 2:
        return 0.0
    
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    return np.sum(lengths)

def resample_curve_by_arc_length(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resample a curve uniformly by arc length using perations.
    
    Args:
        points: Nx3 array of points defining the curve
        num_samples: Number of points in resampled curve
        
    Returns:
        resampled: num_samples x 3 array of resampled points
    """
    if len(points) < 2:
        return points
    
    # Compute cumulative arc length
    segments = points[1:] - points[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(lengths)])
    total_length = cumulative_length[-1]
    
    # Target arc lengths for resampling
    target_lengths = np.linspace(0, total_length, num_samples)
    
    # Find segment indices for all target lengths at once
    indices = np.searchsorted(cumulative_length, target_lengths)
    
    # Handle edge cases using masks
    # Clamp indices to valid range [1, len(points)-1]
    indices = np.clip(indices, 1, len(points) - 1)
    
    # Get segment starts and ends for all samples at once
    segment_starts = cumulative_length[indices - 1]  # Shape: (num_samples,)
    segment_ends = cumulative_length[indices]  # Shape: (num_samples,)
    
    # Compute interpolation parameters for all samples
    # Avoid division by zero
    segment_lengths = segment_ends - segment_starts
    # Add small epsilon to avoid division by zero
    segment_lengths = np.where(segment_lengths > 1e-10, segment_lengths, 1.0)
    t = (target_lengths - segment_starts) / segment_lengths  # Shape: (num_samples,)
    
    # Interpolate all points at once
    # points[indices-1] has shape (num_samples, 3)
    # points[indices] has shape (num_samples, 3)
    # t[:, np.newaxis] has shape (num_samples, 1) for broadcasting
    resampled = (1 - t[:, np.newaxis]) * points[indices - 1] + t[:, np.newaxis] * points[indices]
    
    # Handle exact endpoints (where target_length is 0 or total_length)
    resampled[0] = points[0]
    resampled[-1] = points[-1]
    
    return resampled



def align_curves(curve1: np.ndarray, curve2: np.ndarray, 
                fix_at: str = 'tip') -> tuple[np.ndarray, float, np.ndarray]:
    """
    Optimized alignment of two curves by fixing their tips (or bases) and rotating to minimize distance.
    
    Args:
        curve1: Nx3 array of points for reference curve
        curve2: Mx3 array of points for curve to be aligned
        fix_at: 'tip' to fix tips together, 'base' to fix bases
        
    Returns:
        aligned_curve2: Transformed curve2 aligned to curve1
        rotation_angle: Angle of rotation applied (in radians)
        rotation_axis: Axis of rotation
    """
    # Use indexing directly instead of creating intermediate variables
    if fix_at == 'tip':
        # Fix tips together (last points)
        fixed_idx = -1
        other_idx = 0
    else:  # fix_at == 'base'
        # Fix bases together (first points)
        fixed_idx = 0
        other_idx = -1
    
    # Compute translation and apply in one step
    # Step 1: Translate curve2 so fixed points coincide
    translation = curve1[fixed_idx] - curve2[fixed_idx]
    translated_curve2 = curve2 + translation
    
    # Step 2: Rotate curve2 about fixed point to align other points
    # Compute vectors and norms together
    vec1 = curve1[other_idx] - curve1[fixed_idx]
    vec2 = curve2[other_idx] + translation - curve1[fixed_idx]  # Use translated other point directly
    
    if np.linalg.norm(vec1) < 1e-6 or np.linalg.norm(vec2) < 1e-6:
        # One of the vectors is too small to define a rotation
        return translated_curve2, 0.0, np.array([0.0, 0.0, 1.0])
    
    
    # Compute norms once and reuse
    vec1_len = np.linalg.norm(vec1)
    vec2_len = np.linalg.norm(vec2)
    
    # Normalize vectors (in-place division to avoid temporary arrays)
    vec1_norm = vec1 / vec1_len
    vec2_norm = vec2 / vec2_len
    
    # Compute rotation axis and angle
    rotation_axis = np.cross(vec2_norm, vec1_norm)
    axis_norm = np.linalg.norm(rotation_axis)
    
    # Early return for aligned case
    if axis_norm < 1e-6:
        # Vectors are parallel or anti-parallel
        dot_product = np.dot(vec2_norm, vec1_norm)
        if dot_product < 0:
            # Anti-parallel: rotate 180 degrees about any perpendicular axis
            # Use ternary to avoid double cross product
            rotation_axis = (np.cross(vec1_norm, np.array([1.0, 0.0, 0.0])) 
                            if abs(vec1_norm[0]) < 0.9 
                            else np.cross(vec1_norm, np.array([0.0, 1.0, 0.0])))
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.pi
        else:
            # Already aligned - return early
            return translated_curve2, 0.0, np.array([0.0, 0.0, 1.0])
    else:
        # Normalize and compute angle in one block
        rotation_axis = rotation_axis / axis_norm
        # Use dot product already computed for normalized vectors
        rotation_angle = np.arccos(np.clip(np.dot(vec2_norm, vec1_norm), -1.0, 1.0))
    
    # Apply rotation with vectorized operations
    # Create rotation object
    rotation = Rotation.from_rotvec(rotation_angle * rotation_axis)
    
    # Combine centering, rotation, and recentering in fewer operations
    # Center around fixed point, rotate, and recenter in one expression
    fixed_point = curve1[fixed_idx]
    aligned_curve2 = rotation.apply(translated_curve2 - fixed_point) + fixed_point
    
    return aligned_curve2, rotation_angle, rotation_axis




def compute_shape_distance_fast(curve1: np.ndarray, curve2: np.ndarray, 
                            metric: str = 'average') -> float:
    """
    Optimized computation of average distance between two curves.
    
    Optimized for the 'average' metric case (which is always used).
    
    Args:
        curve1: Nx3 array of points
        curve2: Mx3 array of points
        metric: 'average' (other metrics not optimized)
        
    Returns:
        distance: Average shape distance
    """
    # FAST PATH: For same-sized curves, use direct point-to-point distance
    # if len(curve1) == len(curve2):
    #     # Direct point-to-point distance (much faster for equal-sized curves)
    #     print('eq size')
    #     return float(np.mean(np.linalg.norm(curve1 - curve2, axis=1)))
    
    # GENERAL CASE: Different sized curves
    # Use squared distances for efficiency, only sqrt at the end
    
    if len(curve1) * len(curve2) > 150:  # Use cdist for larger problems
        # Compute pairwise distances using optimized cdist
        pairwise_dists = cdist(curve1, curve2, metric='euclidean')
        
        # Get minimum distances for each point
        distances1to2 = np.min(pairwise_dists, axis=1)
        distances2to1 = np.min(pairwise_dists, axis=0)
        # print('big ')
    else:
        # For small problems, broadcasting with squared distances
        # Compute squared distances first (faster than norm)
        # print('small')
        diff = curve1[:, np.newaxis, :] - curve2[np.newaxis, :, :]
        dists_squared = np.sum(diff * diff, axis=2)
        
        # Take sqrt and find minimums
        distances1to2 = np.sqrt(np.min(dists_squared, axis=1))
        distances2to1 = np.sqrt(np.min(dists_squared, axis=0))
    
    # Return average of all min distances
    return float((np.mean(distances1to2) + np.mean(distances2to1)) / 2)


def match_robot_to_trajectory(robot_shape: np.ndarray, trajectory: np.ndarray,
                                trajectory_arc_length: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Match a robot shape to a trajectory by aligning similar-length portions.
    
    This is specifically designed for continuum robot path following where:
    - The robot shape extends from base to tip
    - The trajectory is a sequence of waypoints
    - We want to match the portion of robot from tip with similar arc length
    
    Args:
        robot_shape: Nx3 array of robot shape points (base to tip)
        trajectory: Mx3 array of trajectory waypoints
        trajectory_arc_length: If provided, use this arc length; otherwise compute
        
    Returns:
        aligned_robot_portion: Aligned portion of robot shape
        shape_distance: Distance between aligned shapes
    """
    
    # Compute trajectory arc length if not provided
    if trajectory_arc_length is None:
        trajectory_arc_length = compute_arc_length(trajectory)
    
    # Find portion of robot shape with similar arc length from tip
    robot_cumulative = [0]
    for i in range(1, len(robot_shape)):
        segment_length = np.linalg.norm(robot_shape[i] - robot_shape[i-1])
        robot_cumulative.append(robot_cumulative[-1] + segment_length)
    
    total_robot_length = robot_cumulative[-1]
    
    # Find start index for similar arc length from tip
    start_length = total_robot_length - trajectory_arc_length
    start_idx = max(0, np.searchsorted(robot_cumulative, start_length))
    
    robot_portion = robot_shape[start_idx:]
    
    # Resample both to same number of points for fair comparison
    num_points = min(len(trajectory), len(robot_portion), 50)
    trajectory_resampled = resample_curve_by_arc_length(trajectory, num_points)
    robot_resampled = resample_curve_by_arc_length(robot_portion, num_points)
    
    # Align robot portion to trajectory (fix tips)
    aligned_robot, _, _ = align_curves(
        trajectory_resampled, robot_resampled, fix_at='tip'
    )
    
    # Compute shape distance
    shape_distance = compute_shape_distance_fast(
        trajectory_resampled, aligned_robot, metric='average'
    )
    
    return aligned_robot, shape_distance