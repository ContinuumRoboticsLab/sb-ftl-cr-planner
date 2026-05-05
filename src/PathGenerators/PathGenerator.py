"""
Task Generator for Continuum Robot

This module generates target paths for the continuum robot by sampling
waypoints from real robot shapes or predefined trajectories.
"""

import numpy as np
from typing import List, Tuple, Optional
from src.RobotModels.ContinuumRobotModel import ContinuumRobotModel

from scipy.spatial.transform import Rotation as R


class TaskGenerator:
    """Generates target paths for continuum robot tasks."""

    def __init__(self, robot: ContinuumRobotModel):
        """
        Initialize task generator with a robot model.

        Args:
            robot: TDCR robot model to use for generating paths
        """
        self.robot = robot

    def sample_from_robot_shape(self, clark_coords: np.ndarray, num_waypoints: int = 10) -> np.ndarray:
        """
        Sample waypoints from a robot shape defined by Clark coordinates.

        Args:
            clark_coords: Clark coordinates defining the robot shape
            num_waypoints: Number of waypoints to sample

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions
        """
        # Get the robot shape
        _, shape_points, _ = self.robot.forward_kinematics(clark_coords)

        # Sample evenly spaced waypoints along the shape
        indices = np.linspace(0, len(shape_points) - 1,
                              num_waypoints, dtype=int)
        waypoints = shape_points[indices]

        return waypoints

    def generate_straight_path(self, start_pos: np.ndarray, end_pos: np.ndarray,
                               num_waypoints: int = 10) -> np.ndarray:
        """
        Generate a straight line path between two points.

        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            num_waypoints: Number of waypoints along the path

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions
        """
        t = np.linspace(0, 1, num_waypoints)
        waypoints = np.array([(1 - ti) * start_pos + ti * end_pos for ti in t])
        return waypoints

    def generate_curved_path(self, start_pos: np.ndarray, end_pos: np.ndarray,
                             control_point: np.ndarray, num_waypoints: int = 10) -> np.ndarray:
        """
        Generate a curved path using quadratic Bezier curve.

        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            control_point: Control point for the curve [x, y, z]
            num_waypoints: Number of waypoints along the path

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions
        """
        t = np.linspace(0, 1, num_waypoints)
        waypoints = []

        for ti in t:
            # Quadratic Bezier curve
            point = (1 - ti)**2 * start_pos + 2 * (1 - ti) * \
                ti * control_point + ti**2 * end_pos
            waypoints.append(point)

        return np.array(waypoints)

    def generate_cubic_bezier_path(self, start_pos: np.ndarray, end_pos: np.ndarray, p1: np.ndarray, p2: np.ndarray, num_waypoints: int = 10) -> np.ndarray:
        """
        Generate a cubic Bezier curve path.

        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            p1: First control point [x, y, z]
            p2: Second control point [x, y, z]
            num_waypoints: Number of waypoints along the path

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions
        """
        t = np.linspace(0, 1, num_waypoints)
        waypoints = []

        for ti in t:
            omt = 1.0 - ti
            # Cubic Bezier: (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
            point = (omt**3) * start_pos + 3 * (omt**2) * ti * p1 + 3 * omt * (ti**2) * p2 + (ti**3) * end_pos
            waypoints.append(point)


        return np.array(waypoints)
    
    def generate_c_shape_path(self, start_pos: np.ndarray, end_pos: np.ndarray,
                              radial_angle: float = 0,
                              num_waypoints: int = 12) -> np.ndarray:
        """Generate a C shape path from start to end. The path is a circular arc (semi-circle).
        The radial angle is the angle in degrees about the axis from start to end which the path will
        go by.

        Args:
            start_pos (np.ndarray): start point
            end_pos (np.ndarray): end point
            radial_angle (float, optional): radial angle in degrees for the arc path to follow. Defaults to 0.
            num_waypoints (int, optional): number of waypoints. Defaults to 12.

        Returns:
            np.ndarray: Array of shape (num_waypoints, 3) with waypoint positions
        """
        chord = end_pos - start_pos
        L = np.linalg.norm(chord)
        if L < 1e-6:
            return np.tile(start_pos, (num_waypoints, 1))

        center = (start_pos + end_pos) / 2.0
        radius = L / 2.0

        # Unit vector from center to start
        v1 = (start_pos - center) / radius

        # Find a vector perpendicular to the chord to define the "outward" direction of the C
        if abs(chord[2]) < 0.9 * L:
            v_perp = np.cross(chord, [0, 0, 1])
        else:
            v_perp = np.cross(chord, [1, 0, 0])
        
        v_perp = v_perp / np.linalg.norm(v_perp)

        # Rotate v_perp around the chord axis by radial_angle
        rotation = R.from_rotvec(np.radians(radial_angle) * (chord / L))
        v2 = rotation.apply(v_perp)

        # Generate points along the semi-circle
        theta = np.linspace(0, np.pi, num_waypoints)
        waypoints = []
        for t in theta:
            point = center + radius * np.cos(t) * v1 + radius * np.sin(t) * v2
            waypoints.append(point)

        
        return np.array(waypoints)[len(waypoints) // 4:]

    def generate_s_shape_path(self, start_pos: np.ndarray, end_pos: np.ndarray,
                              control_scale: int = 0.5,
                              num_waypoints: int = 12) -> np.ndarray:
        """
        Generate an S-shaped path using cubic Bezier curves.

        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            num_waypoints: Number of waypoints along the path

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions
        """
        # Create S-shape with two control points
        mid_height = (start_pos[2] + end_pos[2]) / 2
        mid_x = (start_pos[0] + end_pos[0]) / 2
        control1 = np.array(
            [mid_x / 2, start_pos[1], mid_height + control_scale])
        control2 = np.array(
            [end_pos[0] - (mid_x / 2), end_pos[1], mid_height - control_scale])

        t = np.linspace(0, 1, num_waypoints)
        waypoints = []

        for ti in t:
            # Cubic Bezier curve
            point = (1 - ti)**3 * start_pos + \
                3 * (1 - ti)**2 * ti * control1 + \
                3 * (1 - ti) * ti**2 * control2 + \
                ti**3 * end_pos
            waypoints.append(point)

        return np.array(waypoints)



    def generate_circular_arc_path(self, center: np.ndarray, radius: float, angle_start: float, angle_end: float, num_waypoints=12) -> np.ndarray:
        
        """
        Generate a circular arc path in the XY plane.

        Args:
            center: Center of the circle [x, y, z]
            radius: Radius of the circle
            angle_start: Starting angle in radians
            angle_end: Ending angle in radians
        """
        
        angles = np.linspace(angle_start, angle_end, num=num_waypoints)
        waypoints = []
        for angle in angles[::-1]:
            x = center[0] + radius * np.cos(angle)
            y = center[2] 
            z = center[1] + radius * np.sin(angle)
            waypoints.append([x, y, z])
        return np.array(waypoints)

    def generate_z_shape_path(self, start_pos: np.ndarray, end_pos: np.ndarray,
                              num_waypoints: int = 10) -> np.ndarray:
        """
        Generate a Z-shaped path with three straight segments.

        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            num_waypoints: Number of waypoints along the path

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions
        """
        # Create Z-shape with three segments
        mid1 = np.array([end_pos[0], start_pos[1], start_pos[2]])
        mid2 = np.array([start_pos[0], end_pos[1], end_pos[2]])

        # Distribute waypoints across three segments
        n1 = num_waypoints // 3
        n2 = num_waypoints // 3
        n3 = num_waypoints - n1 - n2

        waypoints = []

        # First segment: start to mid1
        t1 = np.linspace(0, 1, n1)
        for ti in t1:
            point = (1 - ti) * start_pos + ti * mid1
            waypoints.append(point)

        # Second segment: mid1 to mid2
        t2 = np.linspace(0, 1, n2)
        for ti in t2:
            point = (1 - ti) * mid1 + ti * mid2
            waypoints.append(point)

        # Third segment: mid2 to end
        t3 = np.linspace(0, 1, n3)
        for ti in t3:
            point = (1 - ti) * mid2 + ti * end_pos
            waypoints.append(point)

        return np.array(waypoints)

    def generate_spiral_path(self, center: np.ndarray, radius: float, height: float,
                             num_turns: int = 2, num_waypoints: int = 20) -> np.ndarray:
        """
        Generate a spiral path.

        Args:
            center: Center of the spiral [x, y, z]
            radius: Radius of the spiral
            height: Total height of the spiral
            num_turns: Number of complete turns
            num_waypoints: Number of waypoints along the path

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions
        """
        t = np.linspace(0, 2 * np.pi * num_turns, num_waypoints)
        z = np.linspace(0, height, num_waypoints)

        waypoints = []
        for i, (ti, zi) in enumerate(zip(t, z)):
            x = center[0] + radius * np.cos(ti)
            y = center[1] + radius * np.sin(ti)
            z = center[2] + zi
            waypoints.append([x, y, z])


        return np.array(waypoints)
    
    def apply_transformation_to_path(self, waypoints: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        
        """
        Euler angles in xyz and translation
        """
        
        assert translation.shape in [(3, 1), (3,)] 
        assert rotation.shape in [(3, 1), (3,)] 
        
        from src.utils import extraUtils
        
        SE3_mat = extraUtils.create_SE3_matrix(
            rotation=R.from_euler('xyz', rotation.tolist(), degrees=True).as_matrix(),
            translation=translation.flatten()
        )
        transformed_waypoints = extraUtils.apply_SE3_to_points(SE3_mat, waypoints)
        return transformed_waypoints        
    
    
    def generate_task_library(self, task_config: dict, seed: int=42 ):
        """
       
        """
        
        # np.random.seed(seed)
        
        curve_lib = []
        for curve_set, set_config in task_config.items():
            curve_type = set_config.get('type')
            num_curves = set_config.get('count', 1)
            num_waypoints = set_config.get('num_waypoints', 10)

            start_point = set_config.get('start', None)
            end_point = set_config.get('end', None)



            if curve_type is None:
                raise Exception(f'Curve set: {curve_set} -  No curve type provided')
            if curve_type != 'Robot_curve':
                if start_point is None:
                    raise Exception(f'Curve set: {curve_set} - No start point provided in')
                if end_point is None:
                    raise Exception(f'Curve set: {curve_set} - No end point provided in')
    

            if curve_type == 'straight_path':
                def curve_gen_func():
                    start = np.random.normal(start_point.get('mean'), start_point.get('variance'))
                    end = np.random.normal(end_point.get('mean'), end_point.get('variance'))
                    return self.generate_straight_path(start, end, num_waypoints)
            
            elif curve_type == 'S_curve':
                def curve_gen_func():
                    # start = np.random.normal(start_point.get('mean'), start_point.get('variance'))
                    # end = np.random.normal(end_point.get('mean'), end_point.get('variance'))

                    start_mean, start_bound = np.array(start_point.get('mean')), np.array(start_point.get('variance'))
                    end_mean, end_bound = np.array(end_point.get('mean')), np.array(end_point.get('variance'))

                    start = np.random.uniform(start_mean - start_bound, start_mean + start_bound)
                    end = np.random.uniform(end_mean - end_bound, end_mean + end_bound)


                    return self.generate_s_shape_path(start, end, num_waypoints=num_waypoints)
            elif curve_type == 'Robot_curve':
                clarke_config = set_config.get('clarke_coord_sample')
                assert clarke_config is not None

                assert len(clarke_config.get('mean')) == self.robot.num_segments * 2
                assert len(clarke_config.get('variance')) == self.robot.num_segments * 2

                def curve_gen_func():
                    cm = np.array(clarke_config.get('mean'))
                    cv = np.array(clarke_config.get('variance'))

                    #clarke_coord_sample = np.random.normal(clarke_config.get('mean'), clarke_config.get('variance'))
                    #clarke_coord_sample = np.clip(clarke_coord_sample, (-np.array(clarke_config.get('variance'))).tolist(), clarke_config.get('variance'))
                    clarke_coord_sample = np.random.uniform(cm-cv, cm+cv)
                    return self.sample_from_robot_shape(clarke_coord_sample, num_waypoints=num_waypoints)                
            
            elif curve_type == 'C_curve':
                def curve_gen_func():
                    sm, sv = np.array(start_point.get('mean')), np.array(start_point.get('variance'))
                    em, ev = np.array(end_point.get('mean')), np.array(end_point.get('variance'))

                    start = np.random.uniform(sm-sv, sm+sv)
                    end = np.random.uniform(em-ev, em+ev)


                    # start = np.random.normal(start_point.get('mean'), start_point.get('variance'))
                    # end = np.random.normal(end_point.get('mean'), end_point.get('variance'))
                    radial_angle_config = set_config.get('radial_angle')
                    if radial_angle_config is None:
                        raise Exception(f'Curve set: {curve_set} - must provide radial angle')
                    radial_angle = np.random.normal(radial_angle_config.get('mean'), radial_angle_config.get('variance'))
                    return self.generate_c_shape_path(start, end, radial_angle=radial_angle, num_waypoints=num_waypoints)
            else:
                raise Exception(f'Curve set: {curve_set} - Invalid curve type')


            for i in range(num_curves):
                curve_data = {
                    "meta": {
                        "curve_type": curve_type
                    },
                    "data": curve_gen_func()
                }
                curve_lib.append(curve_data)

        return curve_lib


    
    def generate_xz_spline_path(self, control_points: np.ndarray, num_waypoints: int = 15) -> np.ndarray:
        """
        Generate a spline path in the XZ plane using B-spline interpolation.

        Args:
            control_points: Array of shape (n, 2) with control points in XZ coordinates
                          First point should be [0, 0] (origin)
                          Points should be ordered from start to end
            num_waypoints: Number of waypoints to generate along the spline

        Returns:
            waypoints: Array of shape (num_waypoints, 3) with waypoint positions [x, y, z]
                      where y=0 for all points (XZ plane)
        """
        from scipy.interpolate import splprep, splev

        # Ensure control points are in the right format
        if control_points.shape[1] != 2:
            raise ValueError("Control points must be 2D (XZ coordinates)")

        # Add origin as first point if not already there
        if not np.allclose(control_points[0], [0, 0]):
            control_points = np.vstack([[0, 0], control_points])

        # Create B-spline
        tck, u = splprep([control_points[:, 0], control_points[:, 1]],
                         s=0, k=min(3, len(control_points)-1))

        # Generate points along the spline
        u_new = np.linspace(0, 1, num_waypoints)
        x_coords, z_coords = splev(u_new, tck)

        # Create 3D waypoints (y=0 for XZ plane)
        waypoints = np.column_stack(
            [x_coords, np.zeros_like(x_coords), z_coords])

        return waypoints




def create_demo_target_paths():
    """Create example target paths for demonstration."""
    # Create a robot model
    robot = TDCR(
        num_segments=3,
        segment_length=[1.0, 1.0, 1.0],
        tendon_offset=[0.1, 0.1, 0.1],
        points_resolution=0.05
    )

    # Create task generator
    generator = TaskGenerator(robot)

    # Generate different target paths
    target_paths = {}

    # Path 1: Straight line
    target_paths["straight"] = generator.generate_straight_path(
        np.array([0, 0, 0]), np.array([0, 0, 2.5]), num_waypoints=8
    )

    # Path 2: Curved path
    target_paths["curved"] = generator.generate_curved_path(
        np.array([0, 0, 0]), np.array([1, 1, 2]),
        np.array([0.5, 0.5, 1]), num_waypoints=10
    )

    # Path 3: S-shape path
    target_paths["s_shape"] = generator.generate_s_shape_path(
        np.array([0, 0, 0]), np.array([1, 0, 2.5]), num_waypoints=12
    )

    # Path 4: C-shape path
    target_paths["c_shape"] = generator.generate_c_shape_path(
        np.array([0, 0, 0]), np.array([1, 0, 2.5]), num_waypoints=10
    )

    # Path 5: Z-shape path
    target_paths["z_shape"] = generator.generate_z_shape_path(
        np.array([0, 0, 0]), np.array([1, 0, 2.5]), num_waypoints=10
    )

    # Path 6: Spiral path
    target_paths["spiral"] = generator.generate_spiral_path(
        np.array([0, 0, 0]), radius=0.5, height=2, num_turns=1, num_waypoints=15
    )

    # Path 7: Sampled from robot shape
    target_paths["from_shape"] = generator.create_target_path_from_shape(
        "curved", num_waypoints=12
    )

    return target_paths


if __name__ == "__main__":
    # Test the task generator
    target_paths = create_demo_target_paths()

    for name, path in target_paths.items():
        print(f"{name} path: {len(path)} waypoints")
        print(f"  Start: {path[0]}")
        print(f"  End: {path[-1]}")
        print()
