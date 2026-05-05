#!/usr/bin/env python3

import numpy as np
from typing import Dict, List
import time
from scipy.spatial.transform import Rotation as SciRot
from scipy.optimize import minimize

from src.MotionPlanners.GeneralMotionPlanner import *
from src.RobotModels.ConstantCurvatureModel import ConstantCurvature
from src.utils.ShapeConstraints import ConvexShape


class DirectOptimization(GeneralMotionPlanner):
    """Clean implementation of stable sampling-based path follower with 3-point matching."""
    
    def __init__(self, robot: ConstantCurvature,
                 **kwargs):

        kwargs['num_samples'] = 1
        super().__init__(robot,
                    **kwargs)

    def __repr__(self):
        return 'Direct Optimization'

    def get_pos_bounds(self, base_pos_center, dx=2.5, dy=2.5, dz=2.5):
        """
        Creates a square volume centered around base_pos_center. 

        Would do sphere but bounds needs to be a box for each variable to pass into optimizer.
        """

        assert base_pos_center.shape[0] == 3
        x_bound = [base_pos_center[0] - dx, base_pos_center[0] + dx]
        y_bound = [base_pos_center[1] - dy, base_pos_center[1] + dy]
        z_bound = [base_pos_center[2] - dz, base_pos_center[2] + dz]
        
        pos_bounds = np.stack((x_bound, y_bound, z_bound))

        return pos_bounds

    def get_ori_bounds_axis_angle(self):
        """Use axis-angle representation (3 params) instead of quaternion.
        Each component represents rotation around x, y, z in radians.
        """
        # Allow rotation up to +-pi radians around each axis
        return np.array([(-np.pi, np.pi) for _ in range(3)])

    def get_ori_bounds(self, curr_x, curr_y, curr_z, dx=45, dy=45, dz=45):
        """
        Returns box of joint angles bounds in euler angles
        """
         
        x_bound = [curr_x - dx, curr_x + dx]
        y_bound = [curr_y - dy, curr_y + dy]
        z_bound = [curr_z - dz, curr_z + dz]   

        ori_bounds = np.stack((x_bound, y_bound, z_bound))

        return ori_bounds     

    def get_clarke_bounds(self, curr_clarke_coords, dclarke=0.5):
        """
        Creates box around clarke coordinates

        For now just return a static box
        """
        seg1_d = 0.004 * 10
        seg2_d = 0.006 * 10
        seg3_d = 0.008 * 10
        
        seg1x_box = np.array([-seg1_d, seg1_d])
        seg1y_box = np.array([-seg1_d, seg1_d])

        seg2x_box = np.array([-seg2_d, seg2_d])
        seg2y_box = np.array([-seg2_d, seg2_d])
        
        seg3x_box = np.array([-seg3_d, seg3_d])
        seg3y_box = np.array([-seg3_d, seg3_d])

        clarke_bounds = np.stack((seg1x_box, seg1y_box, 
                                  seg2x_box, seg2y_box, 
                                  seg3x_box, seg3y_box,))

        return clarke_bounds


    def cost_function(self, optimization_vars, curr_waypoints, cum_waypoint_arc_length):


        # Extract optimization vars
        base_pos = optimization_vars[:3]
        base_rot_rotvec = optimization_vars[3:6]  # axis-angle representation (3 params)
        clarke_coords = optimization_vars[6:]

        # Convert axis-angle (rotvec) to rotation matrix
        base_rot_as_matrix = SciRot.from_rotvec(base_rot_rotvec).as_matrix()

        # Make sure current robot is TDCR and not Mujoco robot because in theory
        assert isinstance(self.robot, ConstantCurvature)

        _, shape_p, _ = self.robot.forward_kinematics_from_base(clarke_coords, base_pos, base_rot_as_matrix)


        # Calc deviation
        shape_dev = self.compute_shape_deviation_closest(
            shape_p,
            curr_waypoints,
            cum_waypoint_arc_length
        )

        # distance_cost = 
        tip_distance = np.linalg.norm(curr_waypoints[-1] - shape_p[-1])

        if shape_dev < 1:
            shape_dev = shape_dev ** (1/2)
        else:
            shape_dev = shape_dev ** 2
        return shape_dev + tip_distance / 2

    def compute_shape_deviation_closest(self, rob_shape_points, active_waypoints, wp_arc_length):
                    
        # Step 1: Compute arc length of active waypoints            
        active_arc_length = wp_arc_length

        # Step 1: Sample robot shape points from tip based on inserted length
        # Robot shape points go from base (index 0) to tip (index -1)
        robot_shape_points = rob_shape_points
        
        # Compute cumulative arc length from base to find the subset from tip
        robot_cumulative = [0]
        for i in range(1, len(robot_shape_points)):
            segment_length = np.linalg.norm(robot_shape_points[i] - robot_shape_points[i-1])
            robot_cumulative.append(robot_cumulative[-1] + segment_length)
        
        total_robot_length = robot_cumulative[-1]
        
        # Find the portion of robot inserted into path (from tip backwards)
        # We want the last 'active_arc_length' of the robot
        start_length = max(0, total_robot_length - active_arc_length)
        start_idx = max(0, np.searchsorted(robot_cumulative, start_length))
        
        # Get robot subset from start_idx to tip (the inserted portion)
        robot_subset = robot_shape_points[start_idx:]
        
        
        # Step 2: Resample both curves to the same number of points for fair comparison
        active_wps = active_waypoints
        num_sample_points = min(50, max(len(active_wps), len(robot_subset)))
        
        resampled_active_wps = CurveUtils.resample_curve_by_arc_length(active_wps, num_sample_points)
        resampled_robot_subset = CurveUtils.resample_curve_by_arc_length(robot_subset, num_sample_points)
        
        # Step 3 & 4: Compute closest point distances using vectorized operations
        # For each robot point, find minimum distance to any waypoint
        distances = np.min(
            np.linalg.norm(
                resampled_robot_subset[:, np.newaxis, :] - resampled_active_wps[np.newaxis, :, :],
                axis=2
            ),
            axis=1
        )
        
        avg_distance = np.mean(distances)

        return avg_distance 



    def follow_path(self, waypoints: np.ndarray, verbose: bool = True, base_constraint: ConvexShape = None ) -> List[Dict]:
        """
        Follow a path using stable sampling-based approach with 3-point matching.
        
        Args:
            waypoints: Array of waypoint positions to follow
            verbose: Whether to print verbose output
            
        Returns:
            List of history dictionaries for each step
        """
        
        history = []
        
        # Initialize robot state
        base_transform = np.eye(4)
        previous_base_pos = np.array([0, 0, -3]) 
        previous_base_quat = np.array([0, 0, 0, 1]) # Identity
        previous_clarke_coord = np.zeros(self.robot.num_segments * 2)
        
        # Track waypoint progress
        waypoint_index = 0
        cum_arc_length = []
        while waypoint_index < len(waypoints):
            step_start_time = time.time()
            
            # Get active waypoints (all waypoints up to current)
            active_waypoints = waypoints[:waypoint_index + 1]
            target_tip = waypoints[waypoint_index]
            

            # Active waypoints arc_length
            cum_arc_length.append(CurveUtils.compute_arc_length(active_waypoints))


            pos_bounds = self.get_pos_bounds(previous_base_pos)
            ori_bounds = self.get_ori_bounds_axis_angle()
            clarke_bounds = self.get_clarke_bounds(previous_clarke_coord)

            total_bounds = [tuple(x) for x in pos_bounds.tolist()
                            ] + [tuple(x) for x in ori_bounds.tolist()
                            ] + [tuple(x) for x in clarke_bounds.tolist()]
            # Convert previous quaternion to axis-angle for initial guess
            previous_base_rotvec = SciRot.from_quat(previous_base_quat).as_rotvec()
            initial_guess = np.append(np.append(previous_base_pos, previous_base_rotvec), previous_clarke_coord) # Make it the last config

            optim_options = {
                # 'maxiter': 20,
                # 'ftol': 1e-6,  # Function tolerance for convergence
                # 'gtol': 1e-5,  # Gradient tolerance
                # 'maxfun': 500  # Max function evaluations
            }
            #breakpoint()
            optim_result = minimize(
                self.cost_function,
                initial_guess, # last configuration
                args=(active_waypoints, cum_arc_length[-1]),
                method='L-BFGS-B',
                bounds=total_bounds,
                options=optim_options
            )

            if self.verbose:
                print(f'Waypoint: {waypoint_index} - Found result: {optim_result.success}, cost={optim_result.fun:.6f} ')
            
            # Extract optimized values
            best_base_pos = optim_result.x[:3]
            best_base_rotvec = optim_result.x[3:6]
            best_clarke_coords = optim_result.x[6:]
            
            # Convert axis-angle back to rotation matrix and quaternion for tracking
            best_base_rot = SciRot.from_rotvec(best_base_rotvec).as_matrix()
            best_base_quat = SciRot.from_rotvec(best_base_rotvec).as_quat()
            
            best_score = optim_result.fun
     
            # Update base transform
            base_transform = np.eye(4)
            base_transform[:3, :3] = best_base_rot
            base_transform[:3, 3] = best_base_pos
            
            # Compute final shape
            endpoints, shape_points, _ = self.robot.forward_kinematics_from_base(
                best_clarke_coords, best_base_pos, best_base_rot
            )

            if self.verbose:
                print(f".   Best Clarke: {np.round(best_clarke_coords, 4)}")

            
            actual_tip = endpoints[-1]
            tip_error = np.linalg.norm(actual_tip - target_tip)

            # Compute max deviation from waypoints
            max_deviation, waypoint_deviations = self.compute_deviation_metric(active_waypoints, shape_points)
            
            waypoint_index += 1
                
            
            # Record step in history
            step_time = time.time() - step_start_time
            history_entry = {
                'step': len(history) + 1,
                'clark_coords': best_clarke_coords.copy(),
                'base_transform': base_transform.copy(),
                'base_position': base_transform[:3, 3].copy(),
                'base_orientation': base_transform[:3, :3].copy(),
                'endpoints': endpoints.copy(),
                'shape_points': shape_points.copy(),
                'target_tip': target_tip.copy(),
                'actual_tip': actual_tip.copy(),
                'active_waypoints': active_waypoints.copy(),
                'tip_error': tip_error,
                'shape_proximity': max_deviation,  # For compatibility
                'max_deviation': max_deviation,
                'waypoint_deviations': waypoint_deviations,
                'sampling_score': best_score,
                'fallback_used': False,
                'computation_time': step_time,
            }
            
            history.append(history_entry)
            
            # Update previous base position and rotation
            previous_base_pos = best_base_pos.copy()
            previous_base_quat = best_base_quat.copy()
            previous_clarke_coord = best_clarke_coords.copy()
            
            if verbose:
                print(f"\nStep {len(history)}: Target waypoint {waypoint_index}/{len(waypoints)}")
                print(f"  Total step time: {step_time:.4f}s")


        return history
    


