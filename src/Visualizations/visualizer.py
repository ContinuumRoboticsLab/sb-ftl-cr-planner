"""
Visualizer for Continuum Robot Path Following

This module provides visualization tools for the path following process,
showing robot shapes, waypoints, and optimization progress.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
# from src.robot_model import TDCR
from src.RobotModels.ContinuumRobotModel import ContinuumRobotModel


# If running this on linux
# import matplotlib; matplotlib.use('TkAgg')

class PathVisualizer:
    """Visualizer for continuum robot path following."""
    
    def __init__(self, robot: ContinuumRobotModel, use_2d: bool = False):
        """
        Initialize visualizer.
        
        Args:
            robot: TDCR robot model
            use_2d: Whether to use 2D visualization for XZ plane paths
        """
        self.robot = robot
        self.fig = None
        self.ax = None
        self.fixed_limits = None  # Store fixed axis limits
        self.use_2d = use_2d  # New flag for 2D mode
    
    def _compute_fixed_limits(self, waypoints: np.ndarray, history: List[dict] = None):
        """
        Compute a single cubic bounding box covering all waypoints, robot
        shapes, and base positions, so X/Y/Z share the same scale and the
        scene is not skewed.
        """
        all_points = [waypoints]

        if history:
            for step_info in history:
                shape_points = step_info['shape_points']
                base_position = step_info.get('base_position', np.array([0.0, 0.0, -self.robot.robot_length]))
                all_points.extend([shape_points, base_position.reshape(1, 3)])

        all_points = np.vstack(all_points)

        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        center = 0.5 * (min_vals + max_vals)
        half_range = 0.5 * float(np.max(max_vals - min_vals))
        # Pad so robot/markers don't sit on the box edge
        half_range *= 1.15

        if self.use_2d:
            self.fixed_limits = {
                'xlim': (center[0] - half_range, center[0] + half_range),
                'zlim': (center[2] - half_range, center[2] + half_range),
            }
        else:
            self.fixed_limits = {
                'xlim': (center[0] - half_range, center[0] + half_range),
                'ylim': (center[1] - half_range, center[1] + half_range),
                'zlim': (center[2] - half_range, center[2] + half_range),
            }

    def _set_fixed_view_limits(self):
        """Re-apply per-frame state that ax.clear() wipes: limits and aspect.
        View angle is intentionally NOT touched here so user pans/rotations
        persist across frames.
        """
        if not self.fixed_limits:
            return

        self.ax.set_xlim(self.fixed_limits['xlim'])
        if self.use_2d:
            self.ax.set_ylim(self.fixed_limits['zlim'])  # Y axis shows Z values in 2D
            self.ax.set_aspect('equal', adjustable='box')
        else:
            self.ax.set_ylim(self.fixed_limits['ylim'])
            self.ax.set_zlim(self.fixed_limits['zlim'])
            try:
                self.ax.set_box_aspect((1, 1, 1))
            except AttributeError:
                pass

    def _init_view_angle(self):
        """Set the initial viewing angle once, at animation start only."""
        if self.use_2d:
            return
        self.ax.view_init(elev=25, azim=-60)
    
    
    def plot_history(self, history: List[dict], waypoints: np.ndarray, 
                    save_dir: str = None, show_animation: bool = True, 
                    base_constraint: Optional[object] = None,
                    animation_interval: int = 500):
        """
        Plot the entire path following history.
        
        Args:
            history: List of step information from PathFollower
            waypoints: All waypoints in the path
            save_dir: Directory to save step plots (optional)
            show_animation: Whether to show animation
        """
        # Compute fixed limits based on all data
        self._compute_fixed_limits(waypoints, history)
        
        if show_animation:
            self._create_animation(history, waypoints, base_constraint, animation_interval, save_dir)
        else:
            pass
    def _create_animation(self, history: List[dict], waypoints: np.ndarray, 
                          base_constraint: Optional[object] = None, 
                          animation_interval: int = 500, save_dir: str=None):
        """Create an animation of the path following process."""

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._init_view_angle()

        # Remove all axis elements for 3D plots
        # self.ax.axis('off')  # Turn off the axis on the 3D axes object
        # self.ax.grid(False)
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # self.ax.set_zticks([])
        # # Hide panes
        # self.ax.xaxis.pane.fill = False
        # self.ax.yaxis.pane.fill = False
        # self.ax.zaxis.pane.fill = False
        # # Hide grid lines
        # self.ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
        # self.ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
        # self.ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
        # # Make panes invisible
        # self.ax.xaxis.pane.set_edgecolor('none')
        # self.ax.yaxis.pane.set_edgecolor('none')
        # self.ax.zaxis.pane.set_edgecolor('none')
        # # Make axes invisible
        # self.ax.xaxis.line.set_visible(False)
        # self.ax.yaxis.line.set_visible(False)
        # self.ax.zaxis.line.set_visible(False)
        # # Hide tick labels
        # self.ax.set_xticklabels([])
        # self.ax.set_yticklabels([])
        # self.ax.set_zticklabels([])
        # # Make the axis labels invisible
        # self.ax.set_xlabel('')
        # self.ax.set_ylabel('')
        # self.ax.set_zlabel('')

        def animate(frame):
            # Preserve any view angle the user set by dragging — ax.clear()
            # would otherwise reset it to matplotlib's default each frame.
            elev, azim = self.ax.elev, self.ax.azim
            self.ax.clear()
            self.ax.view_init(elev=elev, azim=azim)

            step_info = history[frame]
            shape_points = step_info['shape_points']
            endpoints = step_info['endpoints']
            target_tip = step_info['target_tip']
            actual_tip = step_info['actual_tip']
            active_waypoints = step_info['active_waypoints']
            base_position = step_info.get('base_position', np.array([0.0, 0.0, -self.robot.robot_length]))
            
            # Plot robot shape
            self.ax.plot(shape_points[:, 0], shape_points[:, 1], shape_points[:, 2], 
                        color='blue', linewidth=3, alpha=0.8)
            
            # Plot segment endpoints
            self.ax.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
                           color='red', s=100, zorder=5)
            
            # Plot base point
            self.ax.scatter([base_position[0]], [base_position[1]], [base_position[2]], 
                           color='green', s=150, zorder=5)

            # Plot all waypoints
            self.ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                           color='gray', s=50, alpha=0.3)
            
            # Plot active waypoints
            if len(active_waypoints) > 0:
                self.ax.scatter(active_waypoints[:, 0], active_waypoints[:, 1], active_waypoints[:, 2], 
                               color='orange', s=50, alpha=0.5, zorder=5)
            
            # Plot target and actual tip positions
            self.ax.scatter([target_tip[0]], [target_tip[1]], [target_tip[2]], 
                           color='purple', s=150, marker='*', zorder=5)
            self.ax.scatter([actual_tip[0]], [actual_tip[1]], [actual_tip[2]], 
                           color='cyan', s=150, marker='s', zorder=5)
            
            if step_info.get('last_wp_SE3') is not None and step_info.get('next_wp_SE3') is not None:
                last_wp_SE3 = step_info['last_wp_SE3']
                next_wp_SE3 = step_info['next_wp_SE3']
                self.ax.quiver(*last_wp_SE3[:3, 3], *last_wp_SE3[:3, 0],  length=0.4, color='red', )
                self.ax.quiver(*last_wp_SE3[:3, 3], *last_wp_SE3[:3, 1],  length=0.4, color='green', )
                self.ax.quiver(*last_wp_SE3[:3, 3], *last_wp_SE3[:3, 2],  length=0.4, color='blue', )
                self.ax.text(*(last_wp_SE3[:3, 3] + np.array([0, 0, 0.05])), 'Last WP', color='black')  
                
                self.ax.quiver(*next_wp_SE3[:3, 3], *next_wp_SE3[:3, 0],  length=0.4, color='red')
                self.ax.quiver(*next_wp_SE3[:3, 3], *next_wp_SE3[:3, 1],  length=0.4, color='green')
                self.ax.quiver(*next_wp_SE3[:3, 3], *next_wp_SE3[:3, 2],  length=0.4, color='blue')
                self.ax.text(*(next_wp_SE3[:3, 3] + np.array([0, 0, 0.05])), 'Next WP', color='black')  
            
            

            if base_constraint is not None:
                # Plot the base constraint (e.g., cone)
                X, Y, Z = base_constraint.get_matplotlib_mesh()
                self.ax.plot_surface(X, Y, Z, color='skyblue', alpha=0.7, shade=False, linewidth=0)
            
            
            
            # Set labels and title
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            
            if base_constraint is None:
                self.ax.set_title(f"Path Following Animation - Step {frame}\n"
                                #f"Tip Error: {step_info['tip_error']:.4f}"
                                )
            elif base_constraint is not None:
                self.ax.set_title(f"Path Following Animation - Step {frame}\n"
                                #f"Tip Error: {step_info['tip_error']:.4f}\n"
                                f"Base position in constraint volume: {base_constraint.contains_point(base_position)}")
                                    
            
            # Set fixed view limits
            self._set_fixed_view_limits()
        
        # Create animation
        anim = FuncAnimation(self.fig, animate, frames=len(history), 
                           interval=animation_interval, repeat=True)
        

        # Save as gif BEFORE showing (showing closes the figure)
        if save_dir:
            gif_outputname = save_dir
        else:
            gif_outputname = "path_following_animation.gif"
        print(f"Saving animation to {gif_outputname}...")
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=10)
        #anim.save(gif_outputname, writer=writer)
        # print(f"Animation saved successfully!")
        
        plt.show()
        return anim
    
    def plot_metrics(self, history: List[dict]):
        """
        Plot optimization metrics over time.
        
        Args:
            history: List of step information from PathFollower
        """
        steps = [step_info['step'] for step_info in history]
        tip_errors = [step_info['tip_error'] for step_info in history]
        shape_proximities = [step_info['shape_proximity'] for step_info in history]
        num_active = [step_info['num_active_waypoints'] for step_info in history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Tip error over time
        ax1.plot(steps, tip_errors, 'b-o')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Tip Error')
        ax1.set_title('Tip Position Error')
        ax1.grid(True)
        
        # Shape proximity over time
        ax2.plot(steps, shape_proximities, 'r-o')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Shape Proximity')
        ax2.set_title('Average Distance to Active Waypoints')
        ax2.grid(True)
        
        # Number of active waypoints
        ax3.plot(steps, num_active, 'g-o')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Number of Active Waypoints')
        ax3.set_title('Active Waypoints')
        ax3.grid(True)
        
        # Clark coordinates over time
        clark_data = np.array([step_info['clark_coords'] for step_info in history])
        for i in range(clark_data.shape[1]):
            ax4.plot(steps, clark_data[:, i], label=f'Clark {i+1}')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Clark Coordinate Value')
        ax4.set_title('Clark Coordinates Over Time')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_plot(self, history: List[dict], waypoints: np.ndarray):
        """
        Create a summary plot showing the entire path following process.
        
        Args:
            history: List of step information from PathFollower
            waypoints: All waypoints in the path
        """
        # Compute fixed limits if not already computed
        if self.fixed_limits is None:
            self._compute_fixed_limits(waypoints, history)
        
        fig = plt.figure(figsize=(15, 10))
        
        # Main 3D plot
        ax_main = fig.add_subplot(221, projection='3d')
        
        # Plot all waypoints
        ax_main.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                       color='gray', s=50, alpha=0.5, label='Waypoints')
        
        # Plot robot shapes at key steps
        key_steps = [0, len(history)//2, len(history)-1]
        colors = ['blue', 'green', 'red']
        
        for i, step_idx in enumerate(key_steps):
            if step_idx < len(history):
                step_info = history[step_idx]
                shape_points = step_info['shape_points']
                ax_main.plot(shape_points[:, 0], shape_points[:, 1], shape_points[:, 2], 
                           color=colors[i], linewidth=2, alpha=0.7, 
                           label=f'Step {step_idx}')
        
        ax_main.set_xlabel('X')
        ax_main.set_ylabel('Y')
        ax_main.set_zlabel('Z')
        ax_main.set_title('Path Following Summary')
        ax_main.legend()
        
        # Set fixed view limits and angle for main plot
        if self.fixed_limits:
            ax_main.set_xlim(self.fixed_limits['xlim'])
            ax_main.set_ylim(self.fixed_limits['ylim'])
            ax_main.set_zlim(self.fixed_limits['zlim'])
            try:
                ax_main.set_box_aspect((1, 1, 1))
            except AttributeError:
                pass
        ax_main.view_init(elev=25, azim=-60)
        
        # Metrics plots
        steps = [step_info['step'] for step_info in history]
        tip_errors = [step_info['tip_error'] for step_info in history]
        
        ax1 = fig.add_subplot(222)
        ax1.plot(steps, tip_errors, 'b-o')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Tip Error')
        ax1.set_title('Tip Position Error')
        ax1.grid(True)
        
        # Final robot shape
        ax2 = fig.add_subplot(223, projection='3d')
        final_step = history[-1]
        shape_points = final_step['shape_points']
        endpoints = final_step['endpoints']
        base_position = final_step.get('base_position', np.array([0.0, 0.0, -sum(self.robot.segment_length)]))
        
        ax2.plot(shape_points[:, 0], shape_points[:, 1], shape_points[:, 2], 
                color='blue', linewidth=3, alpha=0.8, label='Final Robot Shape')
        ax2.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
                   color='red', s=100, zorder=5, label='Segment Endpoints')
        ax2.scatter([base_position[0]], [base_position[1]], [base_position[2]], 
                   color='green', s=150, zorder=5, label='Base')
        ax2.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                   color='gray', s=50, alpha=0.5, label='Waypoints')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Final Robot Configuration')
        ax2.legend()
        
        # Set fixed view limits and angle for final robot shape plot
        if self.fixed_limits:
            ax2.set_xlim(self.fixed_limits['xlim'])
            ax2.set_ylim(self.fixed_limits['ylim'])
            ax2.set_zlim(self.fixed_limits['zlim'])
            try:
                ax2.set_box_aspect((1, 1, 1))
            except AttributeError:
                pass
        ax2.view_init(elev=25, azim=-60)
        
        # Clark coordinates evolution
        ax3 = fig.add_subplot(224)
        clark_data = np.array([step_info['clark_coords'] for step_info in history])
        for i in range(clark_data.shape[1]):
            ax3.plot(steps, clark_data[:, i], label=f'Clark {i+1}')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Clark Coordinate Value')
        ax3.set_title('Clark Coordinates Evolution')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

