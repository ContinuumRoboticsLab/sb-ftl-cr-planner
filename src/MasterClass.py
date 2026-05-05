
from src.MotionPlanners.OptimizationBased.DirectOptimization import DirectOptimization
from src.MotionPlanners.SamplingBased.SamplingNoCluster import SamplingNoCluster
from src.MotionPlanners.SamplingBased.ThresholdCluster import ThresholdCluster

from src.utils import ExtraUtils as extraUtils
from src.RobotModels.ConstantCurvatureModel import ConstantCurvature
from src.RobotModels.ContinuumRobotModel import ContinuumRobotModel
import src.utils.CurveUtils as CurveUtils

import json
import numpy as np
from scipy.spatial.transform import Rotation
# NOTE: matplotlib is imported lazily inside the methods that use it
# (graph_alignment_step, compute_shape_deviation_closest with plot_deviation=True).
# Native behaviour is unchanged; this lets the module import on Emscripten/Pyodide
# without pulling matplotlib at module load.
from scipy.spatial.transform import Rotation as SciRot

class GeneralPathFollower():

    def __init__(self, robot: ContinuumRobotModel, 
                #  opt_params=None, fast_mode=None
                 ):
        self.robot = robot
        # self.opt_params = opt_params
        # self.fast_mode = fast_mode
        self.curr_follower = self.get_all_sampling_methods()[0]

    def __repr__(self):
        return "Master MP Manager Class"

    def change_Robot(self, new_Robot):
        self.robot = new_Robot

    def get_sampling_no_cluster(self, *args, **kwargs):
        self.curr_follower = SamplingNoCluster(self.robot, *args, **kwargs)
        return self.curr_follower

    def get_sampling_threshold_cluster(self, *args, **kwargs):
        self.curr_follower = ThresholdCluster(self.robot, *args, **kwargs)
        return self.curr_follower


    def get_direct_optimization_planner(self, *args, **kwargs):
        self.curr_follower = DirectOptimization(self.robot, *args, **kwargs)
        return self.curr_follower



    def get_all_sampling_methods(self):
        return [
            self.get_direct_optimization_planner,
            self.get_sampling_threshold_cluster,
            self.get_sampling_no_cluster,
        ]

    def get_all_sampling_classes(self):
        return [
            DirectOptimization,
            ThresholdCluster,
            SamplingNoCluster,
        ]

    def get_sampling_methods_by_name(self, method_names:  list):
        """
        Get motion planner getter functions by their string names.

        Args:
            method_names: List of string names matching the __repr__ of motion planners

        Returns:
            List of getter functions that return motion planner instances
        """
        # Mapping of string names to getter functions
        name_to_getter = {
            "Direct Optimization": self.get_direct_optimization_planner,
            "Threshold Cluster": self.get_sampling_threshold_cluster,
            "No Cluster/Linear Sampling": self.get_sampling_no_cluster,
        }
        
        ret_methods = []
        for method_name in method_names:
            if method_name in name_to_getter:
                ret_methods.append(name_to_getter[method_name])
            else:
                print(f"Motion Planning method: '{method_name}' not found.")
                print(f"Available methods: {list(name_to_getter.keys())}")
        
        return ret_methods



    def graph_alignment_step(self, old_SE3, old_rob_curve, new_SE3, new_rob_curve, active_waypoints):
        """
        Visualize the alignment step showing SE3 frames and robot curves.
        
        Args:
            old_SE3: Original SE3 transform (4x4 matrix)
            old_rob_curve: Original robot curve points (Nx3 array)
            new_SE3: New SE3 transform (4x4 matrix)
            new_rob_curve: New robot curve points (Nx3 array)
            active_waypoints: Active waypoints to display (Nx3 array)
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16, 7))
        
        # Left plot: SE3 coordinate frames
        ax1 = fig.add_subplot(121, projection='3d')
        
        axis_length = 0.5
        
        # Plot old SE3 frame (lighter colors with alpha)
        origin_old = old_SE3[:3, 3]
        
        # X-axis (red) - use quiver for arrow
        ax1.quiver(origin_old[0], origin_old[1], origin_old[2], 
                  old_SE3[0, 0], old_SE3[1, 0], old_SE3[2, 0],
                  length=axis_length, color='red', alpha=0.5, arrow_length_ratio=0.3, linewidth=3)
        ax1.text(origin_old[0] + axis_length * old_SE3[0, 0], 
                origin_old[1] + axis_length * old_SE3[1, 0], 
                origin_old[2] + axis_length * old_SE3[2, 0], 
                'X', color='red', fontsize=12, fontweight='bold')
        
        # Y-axis (green)
        ax1.quiver(origin_old[0], origin_old[1], origin_old[2], 
                  old_SE3[0, 1], old_SE3[1, 1], old_SE3[2, 1],
                  length=axis_length, color='green', alpha=0.5, arrow_length_ratio=0.3, linewidth=3)
        ax1.text(origin_old[0] + axis_length * old_SE3[0, 1], 
                origin_old[1] + axis_length * old_SE3[1, 1], 
                origin_old[2] + axis_length * old_SE3[2, 1], 
                'Y', color='green', fontsize=12, fontweight='bold')
        
        # Z-axis (blue)
        ax1.quiver(origin_old[0], origin_old[1], origin_old[2], 
                  old_SE3[0, 2], old_SE3[1, 2], old_SE3[2, 2],
                  length=axis_length, color='blue', alpha=0.5, arrow_length_ratio=0.3, linewidth=3)
        ax1.text(origin_old[0] + axis_length * old_SE3[0, 2], 
                origin_old[1] + axis_length * old_SE3[1, 2], 
                origin_old[2] + axis_length * old_SE3[2, 2], 
                'Z', color='blue', fontsize=12, fontweight='bold')
        
        # Plot new SE3 frame (brighter colors)
        origin_new = new_SE3[:3, 3]
        
        # X-axis (red) - use quiver for arrow
        ax1.quiver(origin_new[0], origin_new[1], origin_new[2], 
                  new_SE3[0, 0], new_SE3[1, 0], new_SE3[2, 0],
                  length=axis_length, color='red', alpha=1.0, arrow_length_ratio=0.3, linewidth=3)
        ax1.text(origin_new[0] + axis_length * new_SE3[0, 0], 
                origin_new[1] + axis_length * new_SE3[1, 0], 
                origin_new[2] + axis_length * new_SE3[2, 0], 
                'X', color='red', fontsize=12, fontweight='bold')
        
        # Y-axis (green)
        ax1.quiver(origin_new[0], origin_new[1], origin_new[2], 
                  new_SE3[0, 1], new_SE3[1, 1], new_SE3[2, 1],
                  length=axis_length, color='green', alpha=1.0, arrow_length_ratio=0.3, linewidth=3)
        ax1.text(origin_new[0] + axis_length * new_SE3[0, 1], 
                origin_new[1] + axis_length * new_SE3[1, 1], 
                origin_new[2] + axis_length * new_SE3[2, 1], 
                'Y', color='green', fontsize=12, fontweight='bold')
        
        # Z-axis (blue)
        ax1.quiver(origin_new[0], origin_new[1], origin_new[2], 
                  new_SE3[0, 2], new_SE3[1, 2], new_SE3[2, 2],
                  length=axis_length, color='blue', alpha=1.0, arrow_length_ratio=0.3, linewidth=3)
        ax1.text(origin_new[0] + axis_length * new_SE3[0, 2], 
                origin_new[1] + axis_length * new_SE3[1, 2], 
                origin_new[2] + axis_length * new_SE3[2, 2], 
                'Z', color='blue', fontsize=12, fontweight='bold')
        
        # Mark origins
        ax1.scatter(*origin_old, c='cyan', s=100, marker='o', label='Old Origin')
        ax1.scatter(*origin_new, c='orange', s=100, marker='o', label='New Origin')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('SE3 Coordinate Frames')
        ax1.legend()
        ax1.set_box_aspect([1,1,1])
        
        # Right plot: Robot curves and waypoints
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot active waypoints
        if active_waypoints is not None:
            ax2.plot(active_waypoints[:, 0], active_waypoints[:, 1], active_waypoints[:, 2], 
                    'k--', linewidth=1, alpha=0.5, label='Path')
            ax2.scatter(active_waypoints[:, 0], active_waypoints[:, 1], active_waypoints[:, 2], 
                       c='gray', s=20, alpha=0.5)
        
        # Plot old robot curve (without SE3 transform)
        ax2.plot(old_rob_curve[:, 0], old_rob_curve[:, 1], old_rob_curve[:, 2], 
                'b-', linewidth=2, alpha=0.6, label='Old Robot Curve')
        ax2.scatter(old_rob_curve[0, 0], old_rob_curve[0, 1], old_rob_curve[0, 2], 
                   c='cyan', s=60, marker='o')
        
        # Plot new robot curve (without SE3 transform)
        ax2.plot(new_rob_curve[:, 0], new_rob_curve[:, 1], new_rob_curve[:, 2], 
                'r-', linewidth=2, alpha=0.8, label='New Robot Curve')
        ax2.scatter(new_rob_curve[0, 0], new_rob_curve[0, 1], new_rob_curve[0, 2], 
                   c='orange', s=60, marker='o')
        
        # Add base SE3 frames to right plot
        base_axis_length = 0.3
        
        # Old SE3 frame (lighter)
        ax2.quiver(origin_old[0], origin_old[1], origin_old[2], 
                  old_SE3[0, 0], old_SE3[1, 0], old_SE3[2, 0],
                  length=base_axis_length, color='red', alpha=0.5, arrow_length_ratio=0.3, linewidth=2)
        ax2.text(origin_old[0] + base_axis_length * old_SE3[0, 0] * 1.2, 
                origin_old[1] + base_axis_length * old_SE3[1, 0] * 1.2, 
                origin_old[2] + base_axis_length * old_SE3[2, 0] * 1.2, 
                'X', color='red', fontsize=10, alpha=0.7)
        
        ax2.quiver(origin_old[0], origin_old[1], origin_old[2], 
                  old_SE3[0, 1], old_SE3[1, 1], old_SE3[2, 1],
                  length=base_axis_length, color='green', alpha=0.5, arrow_length_ratio=0.3, linewidth=2)
        ax2.text(origin_old[0] + base_axis_length * old_SE3[0, 1] * 1.2, 
                origin_old[1] + base_axis_length * old_SE3[1, 1] * 1.2, 
                origin_old[2] + base_axis_length * old_SE3[2, 1] * 1.2, 
                'Y', color='green', fontsize=10, alpha=0.7)
        
        ax2.quiver(origin_old[0], origin_old[1], origin_old[2], 
                  old_SE3[0, 2], old_SE3[1, 2], old_SE3[2, 2],
                  length=base_axis_length, color='blue', alpha=0.5, arrow_length_ratio=0.3, linewidth=2)
        ax2.text(origin_old[0] + base_axis_length * old_SE3[0, 2] * 1.2, 
                origin_old[1] + base_axis_length * old_SE3[1, 2] * 1.2, 
                origin_old[2] + base_axis_length * old_SE3[2, 2] * 1.2, 
                'Z', color='blue', fontsize=10, alpha=0.7)
        
        # New SE3 frame (brighter)
        ax2.quiver(origin_new[0], origin_new[1], origin_new[2], 
                  new_SE3[0, 0], new_SE3[1, 0], new_SE3[2, 0],
                  length=base_axis_length, color='red', alpha=1.0, arrow_length_ratio=0.3, linewidth=2)
        ax2.text(origin_new[0] + base_axis_length * new_SE3[0, 0] * 1.2, 
                origin_new[1] + base_axis_length * new_SE3[1, 0] * 1.2, 
                origin_new[2] + base_axis_length * new_SE3[2, 0] * 1.2, 
                'X', color='red', fontsize=10, fontweight='bold')
        
        ax2.quiver(origin_new[0], origin_new[1], origin_new[2], 
                  new_SE3[0, 1], new_SE3[1, 1], new_SE3[2, 1],
                  length=base_axis_length, color='green', alpha=1.0, arrow_length_ratio=0.3, linewidth=2)
        ax2.text(origin_new[0] + base_axis_length * new_SE3[0, 1] * 1.2, 
                origin_new[1] + base_axis_length * new_SE3[1, 1] * 1.2, 
                origin_new[2] + base_axis_length * new_SE3[2, 1] * 1.2, 
                'Y', color='green', fontsize=10, fontweight='bold')
        
        ax2.quiver(origin_new[0], origin_new[1], origin_new[2], 
                  new_SE3[0, 2], new_SE3[1, 2], new_SE3[2, 2],
                  length=base_axis_length, color='blue', alpha=1.0, arrow_length_ratio=0.3, linewidth=2)
        ax2.text(origin_new[0] + base_axis_length * new_SE3[0, 2] * 1.2, 
                origin_new[1] + base_axis_length * new_SE3[1, 2] * 1.2, 
                origin_new[2] + base_axis_length * new_SE3[2, 2] * 1.2, 
                'Z', color='blue', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Robot Curves (Local Frame)')
        ax2.legend()
        ax2.set_box_aspect([1,1,1])
        
        # Set equal axis limits for both plots
        all_points = [old_rob_curve, new_rob_curve]
        if active_waypoints is not None:
            all_points.append(active_waypoints)
        
        # Add axis endpoints for proper scaling
        x_end_old = origin_old + axis_length * old_SE3[:3, 0]
        y_end_old = origin_old + axis_length * old_SE3[:3, 1]
        z_end_old = origin_old + axis_length * old_SE3[:3, 2]
        x_end_new = origin_new + axis_length * new_SE3[:3, 0]
        y_end_new = origin_new + axis_length * new_SE3[:3, 1]
        z_end_new = origin_new + axis_length * new_SE3[:3, 2]
        all_points.append(np.array([origin_old, origin_new, x_end_old, y_end_old, z_end_old, x_end_new, y_end_new, z_end_new]))
        
        all_points_combined = np.vstack(all_points)
        x_min, x_max = all_points_combined[:, 0].min(), all_points_combined[:, 0].max()
        y_min, y_max = all_points_combined[:, 1].min(), all_points_combined[:, 1].max()
        z_min, z_max = all_points_combined[:, 2].min(), all_points_combined[:, 2].max()
        
        # Calculate the maximum range across all axes
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        
        # Add padding and set equal ranges
        padding = max_range * 0.1
        half_range = (max_range + padding) / 2
        
        for ax in [ax1, ax2]:
            ax.set_xlim([x_center - half_range, x_center + half_range])
            ax.set_ylim([y_center - half_range, y_center + half_range])
            ax.set_zlim([z_center - half_range, z_center + half_range])
        
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

    def save_history_to_file(self, history, filename: str, waypoints=None, metad=None):
        """
        Save path following history to a file.
        
        Args:
            history: Path following history
            filename: Output filename
        """
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
            
        save_json = {'metad': metad,'robot': self.robot.get_CR_parameters(), 'history': history, 'waypoints': waypoints}

        # save as json
        with open(filename, 'w+') as f:
            json.dump(save_json, f, indent=4, cls=NumpyEncoder)
        print(f"History saved to {filename}")
    
    def compute_shape_deviation_closest(self, history, waypoints, plot_deviation=False, num_waypoints=None):
        """
        Compute shape deviation by comparing robot shape to active waypoints at each step.
        
        Process:
        1. For each step, find the portion of robot inserted into the path based on arc length 
           of active waypoints, selecting robot points from tip backwards
        2. Resample both the active waypoints curve and robot subset to same number of points 
        3. For each point on the robot subset, find the closest point on the active waypoint 
           curve and compute the distance
        4. Average the distances to get the shape deviation for each history step, 
           expressed as a percentage of robot length
        5. Compute overall shape deviation for entire history as average of per-step deviations
        
        Args:
            history: List of path following history steps
            waypoints: Target waypoints (not used in current implementation)
            plot_deviation: If True, display a plot showing deviation at each step
            
        Returns:
            overall_deviation: Average shape deviation across all steps (as percentage of robot length)
        """
        
        
        ### FIX BECAUSE INTERP WAYPOINTS NEEDS TO BE UPDATED
        
        
        shape_m = self.curr_follower.shape_matcher
        robot_length = self.robot.robot_length
        
        for step_hist in history:
            active_wps = step_hist['active_waypoints']
            if len(active_wps) < 2:
                step_hist['shape_deviation'] = 0.0
                continue
            
            # Step 1: Compute arc length of active waypoints            
            active_arc_length = CurveUtils.compute_arc_length(active_wps)
            
            # Step 1: Sample robot shape points from tip based on inserted length
            # Robot shape points go from base (index 0) to tip (index -1)
            robot_shape_points = step_hist['shape_points']
            
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
            
            # Handle edge case where robot subset is too small
            if len(robot_subset) < 2:
                step_hist['shape_deviation'] = 0.0
                continue
            
            # Step 2: Resample both curves to the same number of points for fair comparison
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
            step_hist['shape_deviation'] = avg_distance / robot_length  # as percentage of robot length

        # Step 5: Overall shape deviation for entire history
        deviations_percent = [step['shape_deviation'] * 100 for step in history]
        overall_deviation = np.mean(deviations_percent)
        max_deviation = np.max(deviations_percent)
        min_deviation = np.min(deviations_percent)
        
        # Optional visualization of deviation over time
        if plot_deviation:
            import matplotlib.pyplot as plt
            
            steps = [step['step'] for step in history if 'step' in step]
            
            plt.figure(figsize=(12, 6))
            plt.plot(steps if steps else range(len(deviations_percent)), deviations_percent, 'b-', linewidth=2, label='Shape Deviation')
            plt.axhline(y=overall_deviation, color='r', linestyle='--', linewidth=1.5, 
                       label=f'Average: {overall_deviation:.3f}%')
            plt.axhline(y=max_deviation, color='orange', linestyle=':', linewidth=1.5, 
                       label=f'Max: {max_deviation:.3f}%')
            plt.axhline(y=min_deviation, color='green', linestyle=':', linewidth=1.5, 
                       label=f'Min: {min_deviation:.3f}%')
            plt.xlabel('Step Number', fontsize=12)
            plt.ylabel('Shape Deviation (% of robot length)', fontsize=12)
            plt.title('Shape Deviation Over Path Following History', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            if num_waypoints is not None:
                # Draw a vertical dashed line at every num_waypoints interval, so if num_waypoints=24 then 

                interval_size = len(history) / num_waypoints
                plt.vlines(np.arange(interval_size, len(history), interval_size), ymin=0, ymax=max(deviations_percent), )        
                
                
                           
            plt.show()
        
        return {
            'mean': overall_deviation,
            'max': max_deviation,
            'min': min_deviation
        }
    
    def compute_tip_deviation(self, history, waypoints):   
        """
        Compute the deviation of the actual tip positions from the target waypoints
         as a percentage of robot length
        
        Args:
            history: Path following history
            waypoints: Target waypoints
        Returns:
            dict: Dictionary containing 'mean', 'max', 'min' deviations as percentages
        """
        deviations = []
        robot_length = sum(self.robot.segment_lengths)
        
        for insert_step in history:
            actual_tip = insert_step['actual_tip']
            closest_wp_idx = np.argmin(np.linalg.norm(waypoints - actual_tip, axis=1))
            insert_step = waypoints[closest_wp_idx]
            
            deviation = np.linalg.norm(actual_tip - insert_step)
            
            deviations.append(100 * deviation / robot_length)  # Convert to percentage
        
        return {
            'mean': np.mean(deviations),
            'max': np.max(deviations),
            'min': np.min(deviations)
        }  
        
                
     
    def slerp_between_rotations(self, R1: np.ndarray, R2: np.ndarray, num_steps: int = 10) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R, Slerp
        # Convert to Rotation objects
        r1 = R.from_matrix(R1)
        r2 = R.from_matrix(R2)
        
        # Create SLERP interpolator
        key_times = [0, 1]
        key_rots = R.concatenate([r1, r2])
        slerp = Slerp(key_times, key_rots)
        
        # Interpolation points
        t_vals = np.linspace(0, 1, num_steps)
        interp_rots = slerp(t_vals)
        
        # Return as array of 3x3 matrices
        return interp_rots.as_matrix()   

    def interpolate_se3_decoupled(self, T1, T2, n):
        """
        SLERP for rotation, linear for translation.
        Interpolate n poses between two SE(3) transformations.

        T1, T2: 4x4 transformation matrices
        n: number of intermediate poses
        """
        from scipy.spatial.transform import Slerp

        # Extract rotations and translations
        R1 = Rotation.from_matrix(T1[:3, :3])
        R2 = Rotation.from_matrix(T2[:3, :3])
        t1 = T1[:3, 3]
        t2 = T2[:3, 3]

        # Generate interpolation parameters (exclude endpoints)
        alphas = np.linspace(0, 1, n + 2)[1:-1]

        # Use scipy's Slerp for proper spherical linear interpolation
        # This handles quaternion sign ambiguity correctly (takes shortest path)
        key_rots = Rotation.concatenate([R1, R2])
        slerp = Slerp([0, 1], key_rots)

        poses = []
        for alpha in alphas:
            # SLERP for rotation (proper shortest-path interpolation)
            R_interp = slerp(alpha)

            # Linear interpolation for translation
            t_interp = (1 - alpha) * t1 + alpha * t2

            # Construct SE(3) matrix
            T_interp = np.eye(4)
            T_interp[:3, :3] = R_interp.as_matrix()
            T_interp[:3, 3] = t_interp
            poses.append(T_interp)

        return poses

    # def interpolate_mp_rot3(self, history, steps_per_waypoint=10, enable_optimization=True, similarity_threshold=0.01):
    #     """
    #     Interpolate motion plan using rotated cluster strategy.

    #     Args:
    #         history: List of history dictionaries from follow_path
    #         steps_per_waypoint: Number of interpolation steps per waypoint
    #         enable_optimization: Whether to apply rotation optimization
    #         similarity_threshold: Maximum allowed shape similarity (default 0.005).
    #                               Shapes with similarity > threshold are excluded.
    #     """
    #     hist_copy = history.copy()

    #     # Optimize each step by rotating base to reference angle and finding matching shape in rotated cluster
    #     optimized_history = []

    #     # Reference X-axis direction from first step
    #     reference_x = history[0]['base_orientation'][:, 0]

    #     for step_num_idx, step in enumerate(hist_copy):
    #         step_copy = step.copy()

    #         shape_lib_idx = step.get('shape_lib_idx', None)
    #         if shape_lib_idx is None:
    #             # No shape index recorded; keep this step unchanged
    #             optimized_history.append(step_copy)
    #             continue

    #         # Identify the rotated cluster for this shape
    #         rotated_cluster_id = self.curr_follower.shape_to_rotated_cluster.get(shape_lib_idx, None)
    #         if rotated_cluster_id is None:
    #             optimized_history.append(step_copy)
    #             continue

    #         # Find base pos by getting relative to last base_pos
    #         if step_num_idx == 0:
    #             last_base_pos = step['base_position']
    #         else:
    #             last_base_pos = hist_copy[step_num_idx - 1]['base_position']

    #         base_pos = step['base_position']  # - last_base_pos
    #         base_ori = step['base_orientation']

    #         # Get current un-transformed shape
    #         current_shape = self.curr_follower.shape_library[shape_lib_idx]

    #         # Step 1: Calculate rotation angle to align base orientation X-axis with reference_x
    #         # (Same logic as interpolate_mp)
    #         x_axis = base_ori[:, 0]
    #         y_axis = base_ori[:, 1]

    #         A = np.dot(reference_x, x_axis)
    #         B = np.dot(reference_x, y_axis)
    #         rotation_angle = np.arctan2(-B, A)

    #         # Step 2: Create the rotated base orientation
    #         Rz_ref = np.array([
    #             [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
    #             [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
    #             [0, 0, 1]
    #         ])
    #         new_base_ori = base_ori @ Rz_ref.T

    #         # Step 3: Apply the rotation to the current shape in local frame
    #         # When base rotates by Rz_ref.T, the shape in local frame needs to rotate by Rz_ref
    #         # to maintain the same world-frame appearance
    #         rotated_shape_points = np.array([Rz_ref @ pt for pt in current_shape['shape_points']])

    #         # Step 4: Find the shape in the rotated cluster that best matches the rotated shape
    #         best_shape_idx = shape_lib_idx
    #         best_shape = current_shape
    #         best_similarity = float('inf')

    #         cluster = self.curr_follower.rotated_clusters[rotated_cluster_id]
    #         if len(cluster) == 0:
    #             print(f"  Step {step_num_idx+1}: SKIPPED (empty cluster {rotated_cluster_id})")
    #             optimized_history.append(step_copy)
    #             continue

    #         for candidate_idx in cluster:
    #             cand = self.curr_follower.shape_library[candidate_idx]
    #             # Compute similarity between rotated current shape and candidate
    #             similarity = self.curr_follower.curve_similarity(
    #                 rotated_shape_points, cand['shape_points']
    #             )

    #             # Find the best (lowest) similarity
    #             if similarity < best_similarity:
    #                 best_similarity = similarity
    #                 best_shape_idx = candidate_idx
    #                 best_shape = cand

    #         # If best shape doesn't pass threshold, keep original without rotation compensation
    #         if best_similarity > similarity_threshold:
    #             print(f"  Step {step_num_idx+1}: SKIPPED (best similarity {best_similarity:.6f} "
    #                   f"> threshold {similarity_threshold}, cluster size={len(cluster)})")
    #             optimized_history.append(step_copy)
    #             continue

    #         # Step 5: Re-run 3-point matching with the best shape to get correct base transform
    #         active_waypoints = step.get('active_waypoints', None)
    #         if active_waypoints is not None and len(active_waypoints) > 2:
    #             # Use 3-point matching to get the correct transformation for best_shape
    #             aligned_robot, _, transform_matrix = \
    #                 self.curr_follower.shape_matcher.match_robot_to_trajectory_3point_fast(
    #                     best_shape['shape_points'], active_waypoints
    #                 )
    #             final_base_ori = transform_matrix[:3, :3]
    #             # Compute base position: tip is at target, base is offset from tip
    #             target_tip = step['target_tip']
    #             robot_base_offset = best_shape['shape_points'][0] - best_shape['shape_points'][-1]
    #             final_base_pos = target_tip + final_base_ori @ robot_base_offset
    #         else:
    #             # Fallback to the rotation-based approach if not enough waypoints
    #             final_base_ori = new_base_ori
    #             final_base_pos = base_pos

    #         # Step 6: Build T_base from 3-point matching result
    #         T_base = np.eye(4)
    #         T_base[:3, :3] = final_base_ori
    #         T_base[:3, 3] = final_base_pos

    #         print(f"  Step {step_num_idx+1}: Rotation={np.degrees(rotation_angle):.2f}°, "
    #               f"Best shape={best_shape_idx}, Similarity={best_similarity:.6f}")

    #         endpoints_world = extraUtils.change_base_frame(T_base, best_shape['endpoints'])
    #         shapepoints_world = extraUtils.change_base_frame(T_base, best_shape['shape_points'])
            
    #         # Visualize current shape and all shapes in rotated cluster
    #         cluster_shapes = [self.curr_follower.shape_library[idx] for idx in self.curr_follower.rotated_clusters[rotated_cluster_id]]
            
    #         # Find indices in cluster
    #         best_idx_in_cluster = None
    #         original_idx_in_cluster = None
    #         for i, idx in enumerate(self.curr_follower.rotated_clusters[rotated_cluster_id]):
    #             if idx == best_shape_idx:
    #                 best_idx_in_cluster = i
    #             if idx == shape_lib_idx:
    #                 original_idx_in_cluster = i
        
    #         # Construct SE3 transforms for original and optimized shapes
    #         T_base_orig = np.eye(4)
    #         T_base_orig[:3, :3] = step['base_orientation']
    #         T_base_orig[:3, 3] = step['base_position']

    #         T_base_opt = np.eye(4)
    #         T_base_opt[:3, :3] = final_base_ori
    #         T_base_opt[:3, 3] = final_base_pos

    #         if step_num_idx > 2 and step_num_idx % 6 == 0:
    #             self.visualize_rotated_cluster(
    #                 rotated_cluster_shapes=cluster_shapes,
    #                 step_num=step_num_idx + 1,
    #                 original_shape_lib_idx=shape_lib_idx,
    #                 optimized_shape_lib_idx=best_shape_idx,
    #                 original_SE3=T_base_orig,
    #                 optimized_SE3=T_base_opt,
    #                 best_shape_idx=best_idx_in_cluster,
    #                 original_shape_idx=original_idx_in_cluster,
    #                 path_waypoints=step.get('active_waypoints', None)
    #             )

    #         if enable_optimization:
    #             # Update step copy with the new shape and 3-point matched transformation
    #             step_copy['clark_coords'] = best_shape['clark_coords'].copy()
    #             step_copy['base_position'] = final_base_pos.copy()
    #             step_copy['base_orientation'] = final_base_ori.copy()
    #             step_copy['base_transform'][:3, :3] = final_base_ori
    #             step_copy['base_transform'][:3, 3] = final_base_pos
    #             step_copy['endpoints'] = endpoints_world
    #             step_copy['shape_points'] = shapepoints_world
    #             step_copy['actual_tip'] = endpoints_world[-1]
    #             step_copy['shape_lib_idx'] = best_shape_idx

    #         optimized_history.append(step_copy)

    #     # Now interpolate with optimized history
    #     interp_history = []
    #     interp_waypoints = []
    #     global_step_counter = 1  # Start step numbering from 1

    #     for i in range(len(optimized_history) - 1):
    #         print(f'Interpolating segment {i+1}/{len(optimized_history)-1}')
    #         is_final_waypoint = (i == len(optimized_history) - 2)
    #         interp_segment = self.interpolate_single_step(optimized_history, steps_per_waypoint, i, is_final_waypoint, global_step_counter)
    #         interp_history.extend(interp_segment[0])
    #         interp_waypoints.extend(interp_segment[1])
    #         global_step_counter += len(interp_segment[0])  # Update counter by number of steps added

    #     return interp_history, interp_waypoints
    
    def interpolate_optimization(self, history, steps_per_waypoint=10, enable_optimization=True, verbose=True):
        """
        Interpolate between history steps by optimizing at each interpolated waypoint.
        
        At each interpolation point, performs the same optimization as follow_path:
        optimizes for base position, base rotation (axis-angle), and clarke coordinates
        to minimize shape deviation and tip distance to the interpolated waypoint.
        
        This method only works with DirectOptimization motion planners.
        
        Args:
            history: List of history dictionaries from follow_path
            steps_per_waypoint: Number of interpolation steps per waypoint
            enable_optimization: Whether to perform optimization (if False, returns empty history)
            verbose: Whether to print verbose output
            
        Returns:
            Tuple of (interpolated_history, interpolated_waypoints)
        """
        from scipy.optimize import minimize
        
        assert isinstance(self.curr_follower, DirectOptimization), \
            "interpolate_optimization must only be used with DirectOptimization motion planners"
        
        if not enable_optimization:
            return [], []
        
        interp_history = []
        interp_waypoints = []
        global_step_counter = 1
        
        # Interpolate between consecutive history steps
        for hist_idx in range(len(history) - 1):
            step_1 = history[hist_idx]
            step_2 = history[hist_idx + 1]
            
            # Get waypoint targets from history
            w1 = step_1['active_waypoints'][-1]
            w2 = step_2['active_waypoints'][-1]
            
            if verbose:
                print(f'Interpolating segment {hist_idx + 1}/{len(history) - 1}')
            
            # Generate interpolated waypoints between step_1 and step_2
            for interp_idx in range(steps_per_waypoint):
                alpha = interp_idx / steps_per_waypoint
                
                # Interpolate target waypoint
                target_wp = (1 - alpha) * w1 + alpha * w2
                interp_waypoints.append(target_wp)
                
                # Build active waypoints for optimization (all waypoints up to current interpolation)
                # Use waypoints from step_1 since that's the full path traversed so far
                active_waypoints = step_1['active_waypoints'].copy()
                
                # Calculate arc length of active waypoints
                wp_arc_length = CurveUtils.compute_arc_length(active_waypoints)
                
                # Initial guess from previous step in interpolation or from history
                if interp_idx == 0:
                    # Use step_1 as initial guess
                    previous_base_pos = step_1['base_position'].copy()
                    previous_base_quat = SciRot.from_matrix(step_1['base_orientation']).as_quat()
                    previous_clarke_coord = step_1['clark_coords'].copy()
                else:
                    # Use last interpolated step as initial guess
                    previous_base_pos = interp_history[-1]['base_position'].copy()
                    previous_base_quat = SciRot.from_matrix(interp_history[-1]['base_orientation']).as_quat()
                    previous_clarke_coord = interp_history[-1]['clark_coords'].copy()
                
                # Get optimization bounds
                pos_bounds = self.curr_follower.get_pos_bounds(previous_base_pos)
                ori_bounds = self.curr_follower.get_ori_bounds_axis_angle()
                clarke_bounds = self.curr_follower.get_clarke_bounds(previous_clarke_coord)
                
                total_bounds = [tuple(x) for x in pos_bounds.tolist()
                                ] + [tuple(x) for x in ori_bounds.tolist()
                                ] + [tuple(x) for x in clarke_bounds.tolist()]
                
                # Convert quaternion to axis-angle for initial guess
                previous_base_rotvec = SciRot.from_quat(previous_base_quat).as_rotvec()
                initial_guess = np.append(np.append(previous_base_pos, previous_base_rotvec), previous_clarke_coord)
                
                # Perform optimization
                optim_result = minimize(
                    self.curr_follower.cost_function,
                    initial_guess,
                    args=(active_waypoints, wp_arc_length),
                    method='L-BFGS-B',
                    bounds=total_bounds,
                    options={}
                )
                
                if verbose:
                    print(f'  Interp point {interp_idx + 1}/{steps_per_waypoint}: cost={optim_result.fun:.6f}')
                
                # Extract optimized values
                best_base_pos = optim_result.x[:3]
                best_base_rotvec = optim_result.x[3:6]
                best_clarke_coords = optim_result.x[6:]
                
                # Convert axis-angle back to rotation matrix and quaternion
                best_base_rot = SciRot.from_rotvec(best_base_rotvec).as_matrix()
                best_base_quat = SciRot.from_rotvec(best_base_rotvec).as_quat()
                
                # Compute final shape
                endpoints, shape_points, tip_SE3 = self.robot.forward_kinematics_from_base(
                    best_clarke_coords, best_base_pos, best_base_rot
                )
                
                # Create base transform
                base_transform = np.eye(4)
                base_transform[:3, :3] = best_base_rot
                base_transform[:3, 3] = best_base_pos
                
                # Compute tip error
                actual_tip = endpoints[-1]
                tip_error = np.linalg.norm(actual_tip - target_wp)
                
                # Record step in history
                history_entry = {
                    'step': global_step_counter,
                    'clark_coords': best_clarke_coords.copy(),
                    'base_transform': base_transform.copy(),
                    'base_position': best_base_pos.copy(),
                    'base_orientation': best_base_rot.copy(),
                    'endpoints': endpoints.copy(),
                    'shape_points': shape_points.copy(),
                    'target_tip': target_wp.copy(),
                    'actual_tip': actual_tip.copy(),
                    'active_waypoints': active_waypoints.copy(),
                    'tip_error': tip_error,
                    'sampling_score': optim_result.fun,
                    'fallback_used': False,
                    'computation_time': 0,  # Not tracked in this version
                }
                
                interp_history.append(history_entry)
                global_step_counter += 1
        
        # Add final waypoint from history
        if len(history) > 0:
            final_step = history[-1]
            interp_history.append({
                'step': global_step_counter,
                'clark_coords': final_step['clark_coords'].copy(),
                'base_transform': final_step['base_transform'].copy(),
                'base_position': final_step['base_position'].copy(),
                'base_orientation': final_step['base_orientation'].copy(),
                'endpoints': final_step['endpoints'].copy(),
                'shape_points': final_step['shape_points'].copy(),
                'target_tip': final_step['target_tip'].copy(),
                'actual_tip': final_step['actual_tip'].copy(),
                'active_waypoints': final_step['active_waypoints'].copy(),
                'tip_error': final_step['tip_error'],
                'sampling_score': final_step['sampling_score'],
                'fallback_used': final_step['fallback_used'],
                'computation_time': 0,
            })
            interp_waypoints.append(final_step['active_waypoints'][-1])
        
        if verbose:
            print(f'Interpolation complete: {len(interp_history)} total steps')
        
        return interp_history, interp_waypoints


     
    def interpolate_mp(self, history, steps_per_waypoint=10, enable_optimization=True, verbose=True):
        

        assert not isinstance(self.curr_follower, DirectOptimization), "Optimization based motion planners may not use this interpolation method"
        
        hist_copy = history.copy()

        ## Rotate base poses to align X axis with reference X axis
        optimized_history = []

        # Reference X-axis direction from first step
        reference_x = history[0]['base_orientation'][:, 0]
    
        for step in hist_copy:
            step_copy = step.copy()

            # Get current base orientation
            base_ori = step['base_orientation']
            x_axis = base_ori[:, 0]
            y_axis = base_ori[:, 1]

            # Strategy: Find rotation angle θ to apply to Clark coords such that
            # the resulting base orientation has its X-axis as close to reference_x as possible.
            #
            # When we rotate Clark coords by θ:
            #   new_base_orientation = base_orientation @ Rz(-θ)
            #   new_x_axis = base_orientation @ Rz(-θ) @ [1,0,0]
            #              = base_orientation @ [cos(θ), -sin(θ), 0]
            #              = cos(θ) * x_axis - sin(θ) * y_axis
            #
            # We want to maximize: reference_x · new_x_axis
            #                    = reference_x · (cos(θ) * x_axis - sin(θ) * y_axis)
            #                    = cos(θ) * (reference_x · x_axis) - sin(θ) * (reference_x · y_axis)
            #                    = A*cos(θ) - B*sin(θ)
            #
            # where A = reference_x · x_axis, B = reference_x · y_axis
            #
            # To find maximum, take derivative and set to zero:
            #   d/dθ [A*cos(θ) - B*sin(θ)] = -A*sin(θ) - B*cos(θ) = 0
            #   tan(θ) = -B/A
            #   θ = atan2(-B, A)

            A = np.dot(reference_x, x_axis)
            B = np.dot(reference_x, y_axis)

            rotation_angle = np.arctan2(-B, A)
            
            # Rotate Clark coordinates by rotation_angle around z
            clark_coords = np.array(step['clark_coords'])
            num_segments = len(clark_coords) // 2

            rotated_clark = clark_coords.copy()
            for seg_idx in range(num_segments):
                cx = clark_coords[seg_idx * 2]
                cy = clark_coords[seg_idx * 2 + 1]

                # Rotate (cx, cy) by rotation_angle
                rotated_clark[seg_idx * 2] = (cx * np.cos(rotation_angle) -
                                              cy * np.sin(rotation_angle))
                rotated_clark[seg_idx * 2 + 1] = (cx * np.sin(rotation_angle) +
                                                   cy * np.cos(rotation_angle))

            # Create rotation matrix around z-axis
            Rz = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1]
            ])

            # Apply rotation to base orientation
            new_base_ori = base_ori @ Rz.T
            # Recompute with rotated orientation
            # endpoints, shapepoints = self.robot.forward_kinematics_from_base(
            #     rotated_clark, step['base_position'], new_base_ori
            # )
            
            # Rotate shape points around the base_ori z-axis by rotation angle
            # Then rotate the base_ori by rotation and update shape points accordingly
            
            # In effect its a rotation then negative of that rotation so back to start
            shapepoints = step_copy['shape_points']
            endpoints = step_copy['endpoints']
            
            # new shapepoints should be exact same as before
            assert np.allclose(shapepoints, step['shape_points']), "Shapepoints changed after rotation!"

            # Update step
            if enable_optimization:
                step_copy['clark_coords'] = rotated_clark
                step_copy['base_position'] = step['base_position']
                step_copy['base_orientation'] = new_base_ori
                step_copy['base_transform'][:3, :3] = new_base_ori
                step_copy['base_transform'][:3, 3] = step['base_position']
                step_copy['endpoints'] = endpoints
                step_copy['shape_points'] = shapepoints
                step_copy['actual_tip'] = endpoints[-1]

            optimized_history.append(step_copy)

        # Now interpolate with optimized history
        interp_history = []
        interp_waypoints = []
        global_step_counter = 1  # Start step numbering from 1

        for i in range(len(optimized_history) - 1):
            if verbose:
                print(f'Interpolating segment {i+1}/{len(optimized_history)-1}')
            is_final_waypoint = (i == len(optimized_history) - 2)
            interp_segment = self.interpolate_single_step(optimized_history, steps_per_waypoint, i, is_final_waypoint, global_step_counter)
            interp_history.extend(interp_segment[0])
            interp_waypoints.extend(interp_segment[1])
            global_step_counter += len(interp_segment[0])  # Update counter by number of steps added

        return interp_history, interp_waypoints
        

    def interpolate_single_step(self, history, steps_per_waypoint, hist_index, is_final_waypoint=False, start_step_num=1):
        """
        INTERPOLATION FORMAT if steps_per_waypoint=10 then the first point will be waypoint p1
        then 9 points will follow and p2 will not be included. So in total 10 steps should be included
        """
        
        interp_history = []
        interpolated_waypoints = []
        
        n = steps_per_waypoint
        
        p1, p2 = history[hist_index], history[hist_index + 1]
        c1, c2 = p1['clark_coords'], p2['clark_coords']
        w1, w2 = p1['active_waypoints'][-1], p2['active_waypoints'][-1]
        
        
        # reconstruct tip pose
        _, _, tip_se3_1 = self.robot.forward_kinematics_from_base(c1, p1['base_position'], p1['base_orientation'])
        _, _, tip_se3_2 = self.robot.forward_kinematics_from_base(c2, p2['base_position'], p2['base_orientation'])
        
       
        # interpolate list of n tip poses
        tip_se3_interp = self.interpolate_se3_decoupled(tip_se3_1, tip_se3_2, n)
        tip_se3_interp.append(tip_se3_2)
        
        for i in range(n + 1):

            # Linear interpolate clark and tip position between waypoints
            alpha = i / n
            wp_interp = (1 - alpha) * w1 + alpha * w2
            interpolated_waypoints.append(wp_interp)

            interp_c = (1 - alpha) * c1 + alpha * c2
            interp_t = tip_se3_interp[i]
            interp_t[:3, 3] = wp_interp
            
            # Find the tip SE3 at the interpolated clark coords
            _, _, tip_SE3 = self.robot.forward_kinematics_from_base(interp_c, np.zeros(3,), np.eye(3))

            # Update base pos from tip pos error
            base_transform = interp_t @ np.linalg.inv(tip_SE3)
            base_pos = base_transform[:3, 3]
            base_rot = base_transform[:3, :3]

            # For complete history info in the dict
            endpoints, shapepoints, _ = self.robot.forward_kinematics_from_base(interp_c, base_pos, base_rot)
          
            interp_transform = np.eye(4)
            interp_transform[:3, :3] = base_rot
            interp_transform[:3, 3] = base_pos
            
            interp_history.append({
                'step': start_step_num + i,  # Sequential step numbering
                'clark_coords': interp_c.copy(),
                'base_transform': interp_transform.copy(),
                'base_position': base_pos.copy(),
                'base_orientation': base_rot.copy(),
                'endpoints': endpoints.copy(),
                'shape_points': shapepoints.copy(),
                'target_tip': interp_t[:3,3].copy(),
                'actual_tip': endpoints[-1].copy(),
                'active_waypoints': history[hist_index]['active_waypoints'].copy(),
                'tip_error': -1, # not implemented yet
                'last_wp_SE3': p1['base_transform'].copy(),
                'next_wp_SE3': p2['base_transform'].copy(),
            })

        if is_final_waypoint:
            return interp_history, interpolated_waypoints
        return interp_history[:-1], interpolated_waypoints[:-1]
        
