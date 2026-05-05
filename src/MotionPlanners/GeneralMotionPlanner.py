
import numpy as np
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool
import json, os, sys, time, warnings


from scipy.spatial.transform import Rotation

# from src.robot_model import TDCR
from src.RobotModels.ContinuumRobotModel import ContinuumRobotModel
from src.MotionPlanners.ShapeMatching import ShapeMatcher3PointOptimized
from src.utils.ShapeConstraints import ConvexShape

import src.utils.CurveUtils as CurveUtils

from src.utils.parralel_utils import _generate_single_shape_uniform, _generate_single_shape_normal, get_global_pool


class GeneralMotionPlanner:
    """
    Parent class for all motion planners
    """
    
    def __init__(self, robot: ContinuumRobotModel, 
                 num_samples: int = 5000,
                 activation_radius: float = 0.3,
                 base_stability_weight: float = 0.1,
                 base_stability_weight_rot: float = 0.1,
                 max_base_movement: float = 0.5,
                 batch_size: int = 500,
                 base_constraint: ConvexShape = None,
                 num_threads: int = -1,
                 custom_shape_lib_path: str = None,
                 random_seed: int = None,
                 verbose: bool =False,
                 parallel_lib_gen: bool = True,
                 **kwargs
                 ):
        """
        Initialize the stable sampling path follower.
        
        Args:
            robot: TDCR robot model
            num_samples: Number of random shapes to pre-compute
            activation_radius: Distance threshold for waypoint completion
            base_stability_weight: Weight for base movement penalty (0-1)
            max_base_movement: Maximum allowed base movement per step
            batch_size: Batch size for parallel processing
            optimization_params: Parameters for optimization fallback
            fast_mode: If True, use faster settings
        """
        self.robot = robot
        self.num_samples = num_samples
        self.activation_radius = activation_radius
        self.random_seed = random_seed

        self.base_stability_weight = base_stability_weight
        self.base_stability_weight_rot = base_stability_weight_rot

        self.max_base_movement = max_base_movement

        self.num_threads = num_threads
        self.batch_size = batch_size

        self.base_volume_constraint = base_constraint

        self.custom_shape_lib_path = custom_shape_lib_path

        self.verbose = verbose
        # multiprocessing isn't available under Emscripten/Pyodide; fall back to
        # serial generation transparently. No effect on native runs.
        if sys.platform == "emscripten" and parallel_lib_gen:
            parallel_lib_gen = False
        self.parallel_lib_gen = parallel_lib_gen

        if self.num_threads <= 0:
            import os
            self.num_threads = os.cpu_count()
        
        # Shape matcher for 3-point alignment
        self.shape_matcher = ShapeMatcher3PointOptimized()
        

        # Pre-compute shape library
        if self.custom_shape_lib_path is not None:
            import json
            print(f"Loading custom shape library from {self.custom_shape_lib_path}...")
            self.shape_library = self.load_custom_shape_library(self.custom_shape_lib_path)
            print(f"Custom shape library loaded with {len(self.shape_library)} shapes")
        else:
            self.shape_library = self._generate_shape_library(use_parallel=self.parallel_lib_gen)
            #print(f"Shape library generated with {len(self.shape_library)} shapes")
        
        # Check for duplicates based on clarke_coords
        self._check_for_duplicates()


    def load_custom_shape_library(self, shape_lib_path: str):
        """Load a custom shape library from a JSON file."""
        with open(shape_lib_path, 'r') as f:
            shape_lib = json.load(f)
        
        metadata = shape_lib.get("meta", {})
        print(f'Custom shape library metadata: {metadata}')
        
        # Ideally read metadata['num_tendons'] to pass into interpolator, now hardcoded to 3
        # for radial symmetry
        shape_library = shape_lib.get("shapes", [])
        
        # Convert lists into numpy arrays
        for shape_info in shape_library:
            # breakpoint()
            shape_info['clark_coords'] = np.array(shape_info['clark_coords'])
            shape_info['shape_points'] = np.array(shape_info['shape_points'])
            shape_info['endpoints'] = np.array(shape_info['endpoints'])
            shape_info['tip_position'] = np.array(shape_info['tip_position'])


            if shape_info.get('arc_length_cumulative') is not None:
                shape_info['arc_length_cumulative'] = np.array(shape_info['arc_length_cumulative'])
            else:
                raise ValueError("Custom shape library is missing 'arc_length_cumulative' data.")
                
        if len(shape_library) == 0:
            raise ValueError("Custom shape library is empty or invalid.")
        return shape_library
    
    def _check_for_duplicates(self):
        """Check for duplicate shapes based on clark_coords and report statistics."""
        seen_coords = {}
        duplicates = []
        
        for idx, shape_info in enumerate(self.shape_library):
            # Convert clark_coords to tuple for hashing (round to avoid floating point issues)
            coords_tuple = tuple(np.round(shape_info['clark_coords'], decimals=10))
            
            if coords_tuple in seen_coords:
                duplicates.append({
                    'index': idx,
                    'duplicate_of': seen_coords[coords_tuple],
                    'clark_coords': shape_info['clark_coords']
                })
            
            # Always track this coordinate (whether it's a duplicate or not)
            # Only store the first occurrence
            if coords_tuple not in seen_coords:
                seen_coords[coords_tuple] = idx
        
        if duplicates:
            print(f"WARNING: Found {len(duplicates)} duplicate shapes in library (out of {len(self.shape_library)} total)")
            print(f"  First few duplicates:")
            for dup in duplicates[:5]:
                print(f"    Shape {dup['index']} is duplicate of shape {dup['duplicate_of']}")
        else:
            pass
            #print(f"No duplicates found in shape library ({len(self.shape_library)} unique shapes)")
    
    
    def _generate_shape_library(self, use_parallel=True, shape_lib_config=None) -> List[Dict]:
        """
        Generate shape library with diverse configurations using rotational symmetry.
        
        Returns:
            List of shape information dictionaries
        """
        # Use Gaussian distribution exploiting rotational symmetry
        
        if shape_lib_config is None:
            shape_lib_config = {
                # "normal_distribution": {
                #     "std_dev": 0.4,
                #     "num_samples": self.num_samples // 2,
                #     "function": _generate_single_shape_normal,
                #     "param_key": "std_dev"
                # },
                "uniform_distribution": {
                    "bounds": 0.35, # Uniform between -0.2 and 0.2
                    "num_samples": self.num_samples,
                    "function": _generate_single_shape_uniform,
                    "param_key": "bounds"
                }
            }
        

        shape_library = []
        

        if use_parallel:

            # Get global pool to avoid over-spawning processes when multiple instances exist
            pool = get_global_pool(self.num_threads)
                    
            # Generate shapes for each distribution type
            for dist_name, config in shape_lib_config.items():
                # Create args list with appropriate parameter
                param_value = config[config["param_key"]]
                # Each thread gets a unique seed: base_seed + thread_idx
                base_seed = self.random_seed if self.random_seed is not None else 0
                args_list = [
                    (i, self.robot.num_segments, param_value, self.robot, self.shape_matcher, base_seed + i)
                    for i in range(config["num_samples"])
                ]
                
                dist_label = dist_name.replace('_', ' ').title()            
                # Use pool.imap for progress updates and better resource management
                for completed_count, result in enumerate(pool.imap(config["function"], args_list), 1):
                    shape_library.append(result)
                    if completed_count % 100 == 0 or completed_count == len(args_list):
                        progress = completed_count / len(args_list) * 100
                        if self.verbose:
                            print(f"\r{dist_label}: {completed_count}/{len(args_list)} ({progress:.1f}%)", end="", flush=True)

                if self.verbose:
                    print()  # New line after progress        
            # # Add some nearly-straight configurations for better coverage
            # # Seed this section for reproducibility
            # if self.random_seed is not None:
            #     np.random.seed(self.random_seed + self.num_samples + 1000)
            
            straight_count = 0# max(10, self.num_samples // 50)
            for i in range(straight_count):
                clark_coords = np.zeros(self.robot.num_segments * 2)
                # Small perturbations for nearly straight shapes
                clark_coords[:] = np.random.normal(0, 0.01, self.robot.num_segments * 2)
                clark_coords = np.clip(clark_coords, -0.02, 0.02)
                
                endpoints, shape_points, tip_SE3 = self.robot.forward_kinematics(clark_coords)
                
                robot_cumulative = [0]
                for i in range(1, len(shape_points)):
                    segment_length = np.linalg.norm(shape_points[i] - shape_points[i-1])
                    robot_cumulative.append(robot_cumulative[-1] + segment_length)
                
                shape_info = {
                    'clark_coords': clark_coords,
                    'shape_points': shape_points,
                    'endpoints': endpoints,
                    'tip_position': endpoints[-1],
                    'arc_length': CurveUtils.compute_arc_length(shape_points),
                    'arc_length_cumulative': robot_cumulative
                }
                shape_library.append(shape_info)
            # print(f"Shape library:{len(shape_library)} generation (parallel) took {time.time() - time0:.3f}s")
        
        else:
            for dist_name, config in shape_lib_config.items():

                param_value = config[config["param_key"]]
                # Each thread gets a unique seed: base_seed + thread_idx
                base_seed = self.random_seed if self.random_seed is not None else 0
                args_list = [
                    (i, self.robot.num_segments, param_value, self.robot, self.shape_matcher, base_seed + i)
                    for i in range(config["num_samples"])
                ]
                
                dist_label = dist_name.replace('_', ' ').title()            
                # Use pool.imap for progress updates and better resource management
                gen_func = config['function']
                for si, s in enumerate(args_list):
                    shape_library.append(gen_func(s))
                    if si % 100 == 0 or si == len(args_list):    
                        if self.verbose:
                            print(f"\r{dist_label}: {completed_count}/{len(args_list)} ({progress:.1f}%)", end="", flush=True)

                if self.verbose:
                            print()  # New line after progress        
        
        return shape_library
    
    
    def save_curr_shape_library(self, save_path: str):
        """Save the current raw shape library to disk to avoid regenerating it every time
        or for consistency sake 

        Args:
            save_path (str): Desired filename path to save too
        """


        meta_dict = {
           # "num_shapes": len(self.shape_library),
            "num_tendons": 3, #self.robot.num_tendons,
            "num_segments": self.robot.num_segments
        }
        shape_dict = []
        for shape_info in self.shape_library:
            shape_cumu = shape_info.get('arc_length_cumulative', None)
            # check if numpy array
            if shape_cumu is not None and isinstance(shape_cumu, np.ndarray):
                shape_info['arc_length_cumulative'] = shape_cumu.tolist()
                
            shape_entry = {
                'clark_coords': shape_info['clark_coords'].tolist(),
                'shape_points': shape_info['shape_points'].tolist(),
                'endpoints': shape_info['endpoints'].tolist(),
                'tip_position': shape_info['tip_position'].tolist(),
                'arc_length': shape_info['arc_length'],
                'arc_length_cumulative': shape_cumu
            }
            shape_dict.append(shape_entry)
        
        full_shape_lib = {
            "meta": meta_dict,
            "shapes": shape_dict
        }
        
        if os.path.exists(save_path):
            # Add warning
            warnings.warn(f"Overwriting existing shape library at {save_path}")
            
        with open(save_path, 'w') as f:
            json.dump(full_shape_lib, f, indent=4)
        print(f"Saved current shape library with {len(self.shape_library)} shapes to {save_path}")

    
    
    
    def evaluate_shape_3point(self, shape_info: Dict, active_waypoints: np.ndarray, cum_arc_length: Optional[np.ndarray]):
        """
        Evaluate how well a shape matches the active waypoints using 3-point matching.
        
        Args:
            shape_info: Dictionary containing robot shape information
            active_waypoints: Array of active waypoint positions
            
        Returns:
            Score indicating shape-waypoint match quality (lower is better)
        """
        if len(active_waypoints) == 1:
            # Simple tip distance for single waypoint
            transform = np.eye(4)
            transform[:3, :3] = np.array([[0, -1, 0],
                                          [1,  0, 0],
                                          [0,  0, 1]])
            return np.linalg.norm(shape_info['tip_position'] - active_waypoints[0]), np.eye(4)
        
        elif len(active_waypoints) == 2:
            # Use tip and one other point for 2 waypoints
            robot_curve = shape_info['shape_points']
            trajectory = active_waypoints
            
            # Match robot to trajectory
            _, avg_distance = CurveUtils.match_robot_to_trajectory(
                robot_curve, trajectory
            )
            # Create 90-degree rotation about z-axis
            transform = np.eye(4)
            transform[:3, :3] = np.array([[0, -1, 0],
                                          [1,  0, 0],
                                          [0,  0, 1]])
            return avg_distance, np.eye(4)
#         Shape Deviation (% of robot length):
#   Mean:    2.4558%
#   Max:     4.7000%
#   Min:     0.8941%
# ============================================================

        else:
            # Use optimized 3-point matching for 3+ waypoints
            robot_curve = shape_info['shape_points']
            
            # Get matching result
            _, avg_distance, transform_matrix = self.shape_matcher.match_robot_to_trajectory_3point_fast(
                shape_info, active_waypoints, cum_arc_length,
                calc_align_robot=False
            )

            return avg_distance, transform_matrix
    
    
    def evaluate_base_volume_constraint(self) -> float:
        # Two options
        if self.base_volume_constraint is not None:
            base_pos_in_constraint_volume = self.base_volume_constraint.contains_point(base_pos)
            if not base_pos_in_constraint_volume:
                # heavy penalty if base position is outside the convex shape
                return 500.0  # Arbitrary large penalty
        return 0
    
    def evaluate_base_movement_constraint(self, curr_base_pos: np.ndarray, previous_base_pos: np.ndarray, shape_score: float) -> float:

        # Add base stability penalty if applicable
        # Skip penalty for first two waypoints (startup phase)
        if previous_base_pos is not None and self.base_stability_weight > 0:
            base_movement = np.linalg.norm(curr_base_pos - previous_base_pos)
            
            # Exponential penalty for excessive movement
            if base_movement > self.max_base_movement:
                base_penalty = self.base_stability_weight * np.exp(base_movement - self.max_base_movement)
            else:
                base_penalty = self.base_stability_weight * (base_movement / self.max_base_movement)
            
            return shape_score * (1 - self.base_stability_weight) + base_penalty * shape_score
        else:
            return shape_score   
     
    def evaluate_base_rot_constraint(self, curr_base_rot, previous_base_rot, shape_score: float) -> float:
        if previous_base_rot is not None and self.base_stability_weight_rot > 0:
            rot_penalty = self.calc_normal_rot_penalty(previous_base_rot, curr_base_rot)
            return rot_penalty * shape_score 
        return 0

    def evaluate_shape(self, shape_info: Dict, 
                                            active_waypoints: np.ndarray,
                                            target_tip: np.ndarray,
                                            previous_base_pos: Optional[np.ndarray],
                                            previous_base_rot: Optional[np.ndarray],
                                            waypoint_index: int,
                                            cum_arc_length: np.ndarray
                                            ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate shape with base stability consideration and base volume consideration, where the base SE3
        must lie within the convex shape specified..
        
        Args:
            shape_info: Dictionary containing robot shape information
            active_waypoints: Array of active waypoint positions
            target_tip: Target tip position
            previous_base_pos: Previous base position (if available)
            previous_base_rot: Previous base rotation (if available)
            waypoint_index: Current waypoint index (0-based)
            
        Returns:
            Tuple of (total_score, base_position, base_rotation)
        """
        # Get shape matching score
        shape_score, transform_matrix = self.evaluate_shape_3point(shape_info, 
                                                                    active_waypoints, 
                                                                    cum_arc_length)
        
        # Compute base pose for this shape
        if transform_matrix is not None:

            # Use 3-point alignment
            base_rot = transform_matrix[:3, :3]
            base_pos = transform_matrix[:3, 3]
            
            # Adjust base position for robot base
            robot_base_offset = shape_info['shape_points'][0] - shape_info['shape_points'][-1]
            base_pos = target_tip + base_rot @ robot_base_offset

        else:
            raise Exception('Could not calculate Evaluation Transformation Matrix')
        
        total_score = 0
        total_score += self.evaluate_base_volume_constraint()
        total_score += self.evaluate_base_movement_constraint(base_pos, previous_base_pos, shape_score)
        total_score += self.evaluate_base_rot_constraint(base_rot, previous_base_rot, shape_score)
        

        return total_score, base_pos, base_rot 
    
    def calc_normal_rot_penalty(self, previous_base_rot: np.ndarray, current_base_rot: np.ndarray) -> float:
        # Compute geodesic distance between rotations on SO(3)
        # Using the Frobenius norm of the rotation difference (related to geodesic distance)
        # Alternatively, use the angle-axis representation
        R_prev = Rotation.from_matrix(previous_base_rot)
        R_curr = Rotation.from_matrix(current_base_rot)
        
        # Compute relative rotation
        R_diff = R_curr * R_prev.inv()
        
        # Get rotation angle (geodesic distance on SO(3))
        rotation_angle = R_diff.magnitude()  # in radians
        
        # Define a maximum acceptable rotation angle (e.g., 30 degrees = 0.524 radians)
        max_rotation_angle = np.pi / 6  # 30 degrees
        
        # Apply significantly large penalty if rotations are far apart
        # Using exponential penalty similar to position penalty
        if rotation_angle > max_rotation_angle:
            rotation_penalty = self.base_stability_weight_rot * 10.0 * np.exp(rotation_angle - max_rotation_angle)
        else:
            rotation_penalty = self.base_stability_weight_rot * 10.0 * (rotation_angle / max_rotation_angle)
        
        return rotation_penalty
    
    def calc_z_angle_rot_penalty(self, previous_base_rot: np.ndarray, current_base_rot: np.ndarray) -> float:
        """
        Calculate rotation penalty based on XY-plane alignment between consecutive frames.
        Measures how much the X and Y axes have rotated relative to each other.
        
        Args:
            previous_base_rot: Previous rotation matrix (3x3)
            current_base_rot: Current rotation matrix (3x3)
            
        Returns:
            Penalty value based on XY-axis misalignment
        """
        # Extract X and Y axis vectors from both rotation matrices
        # Rotation matrix columns are the basis vectors: [X, Y, Z]
        prev_x_axis = previous_base_rot[:, 0]  # X-axis of previous frame
        prev_y_axis = previous_base_rot[:, 1]  # Y-axis of previous frame
        
        curr_x_axis = current_base_rot[:, 0]  # X-axis of current frame
        curr_y_axis = current_base_rot[:, 1]  # Y-axis of current frame
        
        # Calculate angular difference for X-axis alignment
        # Using dot product: cos(theta) = v1 · v2 / (|v1| * |v2|)
        # Since rotation matrices have unit column vectors, we can skip normalization
        cos_x_angle = np.clip(np.dot(prev_x_axis, curr_x_axis), -1.0, 1.0)
        x_angle = np.arccos(cos_x_angle)  # in radians
        
        # Calculate angular difference for Y-axis alignment
        cos_y_angle = np.clip(np.dot(prev_y_axis, curr_y_axis), -1.0, 1.0)
        y_angle = np.arccos(cos_y_angle)  # in radians
        
        # Combined misalignment: average of X and Y axis angles
        # This captures how much the XY-plane has rotated
        avg_xy_misalignment = (x_angle + y_angle) / 2.0
        
        # Define maximum acceptable XY misalignment (e.g., 30 degrees)
        max_xy_angle = np.pi / 6  # 30 degrees
        
        # Apply exponential penalty for large misalignments
        if avg_xy_misalignment > max_xy_angle:
            rotation_penalty = self.base_stability_weight_rot * 10.0 * np.exp(avg_xy_misalignment - max_xy_angle)
        else:
            rotation_penalty = self.base_stability_weight_rot * 10.0 * (avg_xy_misalignment / max_xy_angle)
        
        return rotation_penalty
    
    
    def compute_deviation_metric(self, active_waypoints: np.ndarray, shape_points: np.ndarray) -> Tuple[float, List[float]]:
        
        max_deviation = 0.0
        waypoint_deviations = []
        for waypoint in active_waypoints:
            distances = np.linalg.norm(shape_points - waypoint, axis=1)
            min_distance = np.min(distances)
            waypoint_deviations.append(min_distance)
            max_deviation = max(max_deviation, min_distance)
        
        return max_deviation, waypoint_deviations

    
    
    def _fixup_initial_waypoints(self, history: List[Dict], waypoints: np.ndarray) -> List[Dict]:

        if len(history) < 3 or len(waypoints) < 3:
            return history

        third = history[2]
        third_clark = np.asarray(third["clark_coords"]).copy()
        third_rot   = np.asarray(third["base_orientation"]).copy()
        shape_local = np.asarray(third["selected_shape_info"]["shape_points"])
        # Same offset formula as evaluate_shape: tip_world = base_pos + base_rot @ tip_local
        # so base_pos = target_tip + base_rot @ (shape_base - shape_tip).
        robot_base_offset = shape_local[0] - shape_local[-1]

        for i in (0, 1):
            target = np.asarray(waypoints[i])
            new_base_pos = target + third_rot @ robot_base_offset

            endpoints, shape_points, _ = self.robot.forward_kinematics_from_base(
                third_clark, new_base_pos, third_rot
            )

            new_base_transform = np.eye(4)
            new_base_transform[:3, :3] = third_rot
            new_base_transform[:3, 3]  = new_base_pos

            h = history[i]
            h["clark_coords"]        = third_clark.copy()
            h["base_position"]       = new_base_pos.copy()
            h["base_orientation"]    = third_rot.copy()
            h["base_transform"]      = new_base_transform
            h["shape_points"]        = shape_points
            h["endpoints"]           = endpoints
            h["actual_tip"]          = endpoints[-1].copy()
            h["tip_error"]           = float(np.linalg.norm(endpoints[-1] - target))
            h["selected_shape_info"] = third["selected_shape_info"]

        return history

    def follow_path(self, waypoints: np.ndarray, verbose: bool = True, base_constraint: ConvexShape = None ) -> List[Dict]:
       raise NotImplementedError("This method should be implemented in subclasses.")
    


