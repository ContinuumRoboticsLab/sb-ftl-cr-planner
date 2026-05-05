#!/usr/bin/env python3
"""
Optimized 3-Point Shape Matcher for Continuum Robot Path Following

This module implements an improved shape matching algorithm that uses 3 points
to resolve rotation ambiguity. The optimization selects the 3rd point as the
one furthest from the tip-base line for maximum constraint.

"""

import numpy as np

from scipy.spatial.transform import Rotation as R
import time
import src.utils.CurveUtils as CurveUtils
from typing import Tuple, Optional, Dict



class ShapeMatcher3PointOptimized:
    """Optimized 3-point shape matcher with intelligent 3rd point selection."""
    
    
    def match_robot_to_trajectory_3point_fast(self, robot_info: Dict, 
                                        trajectory: np.ndarray,
                                        cum_arc_length: np.ndarray = None,
                                        calc_align_robot: bool = True
                                        ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Match a robot shape to a trajectory using optimized 3-point alignment.
        
        This method extends the 2-point matching by intelligently selecting the 3rd point
        as the trajectory point furthest from the tip-base line, providing maximum
        constraint for resolving rotation ambiguity.
        
        Args:
            robot_shape: Nx3 array of robot shape points (base to tip)
            trajectory: Mx3 array of trajectory waypoints  
            trajectory_arc_length: If provided, use this arc length; otherwise compute
            calc_align_robot: If False, skip full robot alignment (saves time)
            
        Returns:
            aligned_robot: Aligned robot shape after 3-point matching (None if calc_align_robot=False)
            shape_distance: Distance between aligned shapes 
            transform_matrix: 4x4 transformation matrix applied
        """
        robot_shape = robot_info['shape_points']
        if cum_arc_length is None:
            cum_arc_length = [0]
            for i in range(1, len(robot_shape)):
                segment_length = np.linalg.norm(robot_shape[i] - robot_shape[i-1])
                cum_arc_length.append(cum_arc_length[-1] + segment_length)
        
        trajectory_arc_length = cum_arc_length[-1] if cum_arc_length is not None else None

        # Step 1: Do initial 2-point matching
        # Compute trajectory arc length if not provided


        robot_cumulative = robot_info.get('arc_length_cumulative', None)
        if robot_cumulative is None:
            raise Exception("robot_comulative must be provided for optimized 3-point matching.")
        
        total_robot_length = robot_cumulative[-1]
        
        # Find start index for similar arc length from tip
        start_length = total_robot_length - trajectory_arc_length
        start_idx = max(0, np.searchsorted(robot_cumulative, start_length))
        
        robot_portion = robot_shape[start_idx:]
        
        # Resample both to same number of points
        num_points = min(len(trajectory), len(robot_portion), 50)
        if num_points != len(trajectory):
            trajectory_resampled = CurveUtils.resample_curve_by_arc_length(trajectory, num_points)
        else:
            trajectory_resampled = trajectory
        robot_resampled = CurveUtils.resample_curve_by_arc_length(robot_portion, num_points)


        # Initial 2-point alignment (fix tips)
        # if len(trajectory) > 3:
        #     breakpoint()
        aligned_robot_2pt, rotation_angle_2pt, rotation_axis_2pt = CurveUtils.align_curves(
            trajectory_resampled, robot_resampled, fix_at='tip'
        )

        
        # Convert to rotation object
        rotation_axis_2pt_norm = np.linalg.norm(rotation_axis_2pt)
        if rotation_axis_2pt_norm > 1e-6:
            rotation_2pt = R.from_rotvec(rotation_angle_2pt * rotation_axis_2pt / rotation_axis_2pt_norm)
        else:
            rotation_2pt = R.from_matrix(np.eye(3))
        
        # Step 2: OPTIMIZED 3rd point selection - find point furthest from tip-base line
        tip_point = trajectory[-1]
        base_point = trajectory[0]
        line_direction = tip_point - base_point
        line_length = np.linalg.norm(line_direction)
        
        if line_length > 1e-6 and len(trajectory) > 2:
            line_direction = line_direction / line_length
            
            # Find point with maximum distance from tip-base line - VECTORIZED
            # Get all intermediate points (excluding endpoints)
            intermediate_points = trajectory[1:-1]  # Shape: (N-2, 3)
            
            # Vectors from base to all intermediate points
            base_to_points = intermediate_points - base_point  # Shape: (N-2, 3)
            
            # Project onto line direction (vectorized dot product)
            projection_lengths = base_to_points @ line_direction  # Shape: (N-2,)
            
            # Projection points for all intermediate points
            projection_points = base_point + np.outer(projection_lengths, line_direction)  # Shape: (N-2, 3)
            
            # Distances from points to their projections
            distances_from_line = np.linalg.norm(intermediate_points - projection_points, axis=1)  # Shape: (N-2,)
            
            # Find index of maximum distance
            max_idx_in_intermediate = np.argmax(distances_from_line)
            third_idx = max_idx_in_intermediate + 1  # Adjust for offset (we excluded first point)            
        else:
            # Degenerate case - use middle point
            third_idx = len(trajectory) // 2

        third_point_traj = trajectory[third_idx]
        third_point_arc_length = cum_arc_length[third_idx]
        
        # Find corresponding point on robot (after 2-point alignment) - VECTORIZED
        # Compute all segment lengths at once
        segments = aligned_robot_2pt[1:] - aligned_robot_2pt[:-1]
        segment_lengths = np.linalg.norm(segments, axis=1)
        aligned_arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
       
        # Map trajectory arc length to robot arc length
        if trajectory_arc_length > 1e-6:
            robot_third_arc_length = (third_point_arc_length / trajectory_arc_length) * aligned_arc_lengths[-1]
        else:
            robot_third_arc_length = aligned_arc_lengths[-1] / 2  # Use middle
        robot_third_idx = np.searchsorted(aligned_arc_lengths, robot_third_arc_length)
        robot_third_idx = min(max(1, robot_third_idx), len(aligned_robot_2pt) - 2)
        
        third_point_robot = aligned_robot_2pt[robot_third_idx]
        
        # Step 3: Compute rotation around tip-base axis to align 3rd points
        # Get tip and base points
        tip_point = trajectory[-1]  # Both should be at same position after 2-pt matching
        base_point_traj = trajectory[0]
        
        # Axis of rotation is along tip-base line
        axis = tip_point - base_point_traj
        axis_length = np.linalg.norm(axis)
        if axis_length > 1e-6:
            axis = axis / axis_length
            
            # Project 3rd points onto plane perpendicular to axis
            # Vector from base to 3rd point
            v_traj = third_point_traj - base_point_traj
            v_robot = third_point_robot - base_point_traj
            
            # Project onto perpendicular plane
            v_traj_perp = v_traj - np.dot(v_traj, axis) * axis
            v_robot_perp = v_robot - np.dot(v_robot, axis) * axis
            
            # Compute angle between projections
            len_traj = np.linalg.norm(v_traj_perp)
            len_robot = np.linalg.norm(v_robot_perp)
            
            if len_traj > 1e-6 and len_robot > 1e-6:
                v_traj_perp = v_traj_perp / len_traj
                v_robot_perp = v_robot_perp / len_robot
                
                # Compute rotation angle
                cos_angle = np.clip(np.dot(v_traj_perp, v_robot_perp), -1, 1)
                angle = np.arccos(cos_angle)
                
                # Determine rotation direction
                cross = np.cross(v_robot_perp, v_traj_perp)
                if np.dot(cross, axis) < 0:
                    angle = -angle
                
                # OPTIMIZATION: Skip rotation if angle is negligible (< 0.01 radians ≈ 0.57 degrees)
                if abs(angle) > 0.01:
                    # Apply rotation around axis
                    rotation_3pt = R.from_rotvec(angle * axis)
                    
                    # Rotate robot around tip point - VECTORIZED
                    # Translate all points relative to tip
                    points_rel = aligned_robot_2pt - tip_point  # Shape: (N, 3)
                    # Apply rotation to all points at once
                    points_rot = rotation_3pt.apply(points_rel)  # Shape: (N, 3)
                    # Translate back
                    aligned_robot_3pt = points_rot + tip_point  # Shape: (N, 3)
                else:
                    # Angle too small, skip rotation
                    aligned_robot_3pt = aligned_robot_2pt
                    rotation_3pt = R.from_matrix(np.eye(3))
            else:
                # No additional rotation needed
                aligned_robot_3pt = aligned_robot_2pt
                rotation_3pt = R.from_matrix(np.eye(3))
        else:
            # Degenerate case - no additional rotation
            aligned_robot_3pt = aligned_robot_2pt
            rotation_3pt = R.from_matrix(np.eye(3))
        
        
        # Compute final shape distance
        shape_distance = CurveUtils.compute_shape_distance_fast(
            trajectory_resampled,
            CurveUtils.resample_curve_by_arc_length(aligned_robot_3pt, num_points),
            metric='average'
        )
        
        
        # Create transformation matrix
        total_rotation = rotation_3pt * rotation_2pt
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = total_rotation.as_matrix()
        transform_matrix[:3, 3] = trajectory[-1] - robot_shape[-1]  # Translation to match tips
        
        # Return full aligned robot (not just portion)
        # Apply transformation to full robot shape
        aligned_full_robot = None
        if calc_align_robot:
            aligned_full_robot = np.zeros_like(robot_shape)
            for i in range(len(robot_shape)):
                if i >= start_idx:
                    # This portion was aligned
                    idx_in_portion = i - start_idx
                    if idx_in_portion < len(aligned_robot_3pt):
                        aligned_full_robot[i] = aligned_robot_3pt[idx_in_portion]
                    else:
                        # Extrapolate if needed
                        aligned_full_robot[i] = aligned_robot_3pt[-1]
                else:
                    # Apply same transformation to rest of robot
                    point_transformed = transform_matrix[:3, :3] @ robot_shape[i] + transform_matrix[:3, 3]
                    aligned_full_robot[i] = point_transformed
            
        return aligned_full_robot, shape_distance, transform_matrix
    

if __name__ == "__main__":
    # Test the optimized 3-point shape matcher
    print("Testing Optimized 3-Point Shape Matcher")
    
    # Create a test trajectory (curved path with significant deviation from straight line)
    t = np.linspace(0, np.pi, 20)
    trajectory = np.column_stack([
        np.sin(t),
        0.5 * np.cos(2*t),  # Larger amplitude for better testing
        t
    ])
    
    # Create a robot shape (similar curve but rotated)
    robot_shape = np.column_stack([
        np.sin(t),
        0.5 * np.sin(2*t),  # Different from trajectory
        t
    ])
    
    # Test optimized 3-point matching
    matcher = ShapeMatcher3PointOptimized()
    aligned_robot, distance, transform = matcher.match_robot_to_trajectory_3point(
        robot_shape, trajectory
    )
    
    print(f"Shape distance after optimized 3-point matching: {distance:.6f}")
    print(f"Transform matrix:\n{transform}")
    
    # Show optimization info
    opt_info = matcher.get_last_optimization_info()
    if opt_info:
        print(f"\nOptimization details:")
        print(f"  3rd point index: {opt_info['third_point_idx']}")
        print(f"  Distance from tip-base line: {opt_info['third_point_distance_from_line']:.6f}")
        print(f"  Trajectory points: {opt_info['num_trajectory_points']}")
        print(f"  Tip-base line length: {opt_info['line_length']:.6f}")