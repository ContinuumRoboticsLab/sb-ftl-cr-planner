import numpy as np
from typing import Dict, List
import time

from src.RobotModels import ContinuumRobotModel
from src.utils.ShapeConstraints import ConvexShape
from src.utils.parralel_utils import parallel_find_min_shape_eval#, _generate_single_shape
from src.MotionPlanners.GeneralMotionPlanner import GeneralMotionPlanner
import src.utils.CurveUtils as CurveUtils


class SamplingNoCluster(GeneralMotionPlanner):
    
    def __init__(self, robot: ContinuumRobotModel, 
                 **kwargs):
        """
        Initialize the stable sampling path follower.
        
        Args:
            robot: TDCR robot model
            num_samples: Number of random shapes to pre-compute
            activation_radius: Distance threshold for waypoint completion
            base_stability_weight: Weight for base movement penalty (0-1)
            max_base_movement: Maximum allowed base movement per step
            use_parallel: If True, use parallel processing
            batch_size: Batch size for parallel processing
            use_optimization_fallback: If True, use optimization when sampling fails
            optimization_params: Parameters for optimization fallback
            fast_mode: If True, use faster settings
        """
        super().__init__(robot, 
                         **kwargs)

    def __repr__(self):
        return "No Cluster_Linear Sampling"

    
    
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
        clark_coords = np.zeros(self.robot.num_segments * 2)
        base_transform = np.eye(4)
        previous_base_pos = None
        previous_base_rot = None
        
        # Track waypoint progress
        waypoint_index = 0
        cum_arc_length = []
        
        while waypoint_index < len(waypoints):
            step_start_time = time.time()
            
            # Get active waypoints (all waypoints up to current)
            active_waypoints = waypoints[:waypoint_index + 1]
            target_tip = waypoints[waypoint_index]
            
            if verbose:
                print(f"\nStep {len(history) + 1}: Target waypoint {waypoint_index + 1}/{len(waypoints)}")
                print(f"  Target position: {target_tip}")

            # Evaluate all shapes with base stability
            best_score = float('inf')
            best_shape = None
            best_base_pos = None
            best_base_rot = None
            
            # Choose shape evaluation function

            shape_eval_function = self.evaluate_shape
            cum_arc_length.append(CurveUtils.compute_arc_length(active_waypoints))


            if False: #self.use_parallel:
                other_args = (active_waypoints, target_tip, previous_base_pos, previous_base_rot, waypoint_index, cum_arc_length)
                ret_val = parallel_find_min_shape_eval(shape_eval_function, self.shape_library, other_args, self.num_threads)
                best_score, best_shape, best_base_pos, best_base_rot = (
                    ret_val['min_value'], ret_val['best_shape'], ret_val['best_base_pos'], ret_val['best_base_rot']
                )
            else:
                for shape_info in self.shape_library:
                    score, base_pos, base_rot = shape_eval_function(
                        shape_info, active_waypoints, target_tip, previous_base_pos, 
                        previous_base_rot, waypoint_index, cum_arc_length
                    )
                    
                    if score < best_score:
                        best_score = score
                        best_shape = shape_info
                        best_base_pos = base_pos
                        best_base_rot = base_rot
                    
            # Handle case where no valid shape is found
            if best_shape is None:
                # No fallback, skip to next waypoint
                if verbose:
                    print("  WARNING: No valid shape found ...")
                waypoint_index += 1
                continue
            else:
                clark_coords = best_shape['clark_coords'].copy()
                fallback_used = False
                selected_shape_info = best_shape
            
            # Update base transform
            base_transform = np.eye(4)
            base_transform[:3, :3] = best_base_rot
            base_transform[:3, 3] = best_base_pos
            
            # Compute final shape
            endpoints, shape_points, tip_SE3 = self.robot.forward_kinematics_from_base(
                clark_coords, best_base_pos, best_base_rot
            )
            actual_tip = endpoints[-1]
            tip_error = np.linalg.norm(actual_tip - target_tip)
            
            # Compute max deviation from waypoints
            max_deviation, waypoint_deviations = self.compute_deviation_metric(active_waypoints, shape_points)
            
            
            # Check if we've reached the target
            if tip_error < self.activation_radius:
                waypoint_index += 1
            
            # Record step in history
            step_time = time.time() - step_start_time
            history_entry = {
                'step': len(history) + 1,
                'clark_coords': clark_coords.copy(),
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
                'selected_shape_info': selected_shape_info,
                'computation_time': step_time,
            }
            
            history.append(history_entry)
            
            # Update previous base position
            previous_base_pos = best_base_pos.copy()
            previous_base_rot = best_base_rot.copy()

            
            if verbose:
                print(f"  Total step time: {step_time:.2f}s")


        # Paper §III-B-2: fix up the first two waypoints by reusing w3's shape.
        history = self._fixup_initial_waypoints(history, waypoints)
        return history
