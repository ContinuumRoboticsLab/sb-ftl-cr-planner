"""
Single-curve demo: generate a target path, plan an FTL motion for it, and visualize.

Examples:

  # Default: PCC robot, threshold-clustered planner, Bezier curve, 5k library
  python run_example.py

  # Linear search baseline, S-curve target, 20k library
  python run_example.py --planner "No Cluster/Linear Sampling" --curve s --num-samples 20000

  # Use an external shape library (skips library generation)
  python run_example.py --shape-lib-path /path/to/library.json

  # Skip animation, just print metrics
  python run_example.py --no-animate

  # Save the dense (interpolated) motion plan to JSON
  python run_example.py --save-history my_plan.json
"""

import argparse
import time

import numpy as np

from src.MasterClass import GeneralPathFollower
from src.PathGenerators.PathGenerator import TaskGenerator
from src.RobotModels.ConstantCurvatureModel import ConstantCurvature
from src.Visualizations.visualizer import PathVisualizer


CURVES = {
    "bezier":       "Quadratic Bezier through a control point.",
    "s":            "Cubic Bezier S shape between two endpoints.",
    "c":            "C-shaped semi-circular arc between two endpoints.",
    "robot":        "Path sampled from the robot's own forward model at a given Clarke configuration.",
    "cubic_bezier": "Cubic Bezier with user-provided control points (--p1 / --p2)",
}

PLANNERS = [
    "No Cluster/Linear Sampling",
    "Threshold Cluster",
    "Direct Optimization",
]


def _build_curve(generator: TaskGenerator, name: str, num_waypoints: int,
                 p1=None, p2=None) -> np.ndarray:
    if name == "bezier":
        return generator.generate_curved_path(
            np.array([0, 0, 0]),
            np.array([2, 0, 1.3]),
            np.array([0, 0, 1.6]),
            num_waypoints=num_waypoints,
        )
    if name == "s":
        return generator.generate_s_shape_path(
            np.array([0, 0, 0]), np.array([-2.3, 0, 0.8]), num_waypoints=num_waypoints
        )
    if name == "c":
        return generator.generate_c_shape_path(
            np.array([0, 0, 0]), np.array([-2.3, 0, 0.8]), num_waypoints=num_waypoints
        )
    if name == "robot":
        return generator.sample_from_robot_shape(
            np.array([0, 0.1, 0, 0.2, -0.3, -0.1]), num_waypoints=num_waypoints
        )
    if name == "cubic_bezier":
        # Same fixed start/end as the website's S-shape editor; control
        # points come from --p1 / --p2 (each takes three numbers: X Y Z).
        if p1 is None or p2 is None:
            raise ValueError("cubic_bezier requires --p1 and --p2 (each: three numbers, X Y Z).")
        return generator.generate_cubic_bezier_path(
            np.array([0, 0, 0]),
            np.array([-2.3, 0, 0.8]),
            np.array(p1, dtype=float),
            np.array(p2, dtype=float),
            num_waypoints=num_waypoints,
        )
    raise ValueError(f"Unknown curve: {name}. Choices: {list(CURVES)}")


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"  {label:<5} mean: {metrics['mean']:>8.4f}%   max: {metrics['max']:>8.4f}%   "
          f"min: {metrics['min']:>8.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--planner", default="Threshold Cluster", choices=PLANNERS,
                        help="Motion planner to use.")
    parser.add_argument("--num-samples", type=int, default=10_000,
                        help="Shape library size (ignored if --shape-lib-path is set).")
    parser.add_argument("--shape-lib-path", default=None,
                        help="Path to a pre-generated shape library JSON. If set, the library "
                             "is loaded instead of generated.")
    parser.add_argument("--curve", default="bezier", choices=list(CURVES),
                        help="Target curve to follow.")
    parser.add_argument("--p1", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                        help="Cubic-Bezier first control point (only used with --curve cubic_bezier). "
                             "Three numbers: X Y Z.")
    parser.add_argument("--p2", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                        help="Cubic-Bezier second control point (only used with --curve cubic_bezier). "
                             "Three numbers: X Y Z.")
    parser.add_argument("--num-waypoints", type=int, default=10,
                        help="Number of waypoints on the target curve.")
    parser.add_argument("--interpolation-steps", type=int, default=40,
                        help="Number of interpolation steps between consecutive waypoints.")
    parser.add_argument("--base-stability-weight", type=float, default=0.3)
    parser.add_argument("--base-stability-weight-rot", type=float, default=0.3)
    parser.add_argument("--similarity-threshold", type=float, default=2.5,
                        help="γ — similarity threshold for the Threshold-Cluster planner ")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (for reproducible library generation).")
    parser.add_argument("--no-animate", action="store_true",
                        help="Skip the matplotlib animation, only print metrics.")
    parser.add_argument("--save-history", default=None,
                        help="Path to save the dense (interpolated) motion plan as JSON. "
                             "Includes per-step Clarke coords, base SE(3), shape points, the "
                             "target waypoints, robot parameters, and the deviation metrics.")
    parser.add_argument("--save-sparse-history", default=None,
                        help="Path to also save the sparse pre-interpolation plan (one entry "
                             "per waypoint) as JSON.")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    robot = ConstantCurvature(
        num_segments=3,
        segment_lengths=[1, 1, 1],
        tendon_offset=[0.2, 0.2, 0.2],
        points_resolution=0.05,
    )

    generator = TaskGenerator(robot)
    waypoints = _build_curve(generator, args.curve, args.num_waypoints, p1=args.p1, p2=args.p2)

    general_follower = GeneralPathFollower(robot)
    sampler = general_follower.get_sampling_methods_by_name([args.planner])[0]

    sampler_kwargs = dict(
        base_stability_weight=args.base_stability_weight,
        base_stability_weight_rot=args.base_stability_weight_rot,
        random_seed=args.seed,
    )
    if args.shape_lib_path is not None:
        sampler_kwargs["num_samples"] = 1
        sampler_kwargs["custom_shape_lib_path"] = args.shape_lib_path
    else:
        sampler_kwargs["num_samples"] = args.num_samples

    # γ — only meaningful for Threshold Cluster; negative means "auto-estimate".
    if args.planner == "Threshold Cluster" and args.similarity_threshold >= 0:
        sampler_kwargs["similarity_threshold"] = args.similarity_threshold

    t0 = time.time()
    follower = sampler(**sampler_kwargs)
    print(f"Sampler creation took {time.time() - t0:.3f}s")

    t0 = time.time()
    history = follower.follow_path(waypoints)
    print(f"Path following took {time.time() - t0:.3f}s")

    t0 = time.time()
    interp_history, interp_waypoints = general_follower.interpolate_mp(
        history, steps_per_waypoint=args.interpolation_steps, enable_optimization=True
    )
    print(f"Interpolation took {time.time() - t0:.3f}s for {len(interp_history)} total steps")

    skip_first_n = 2 * args.interpolation_steps
    tip_dev = general_follower.compute_tip_deviation(
        interp_history[skip_first_n:], interp_waypoints[skip_first_n:]
    )
    shape_dev = general_follower.compute_shape_deviation_closest(
        interp_history[skip_first_n:],
        interp_waypoints[skip_first_n:],
        plot_deviation=False,
        num_waypoints=args.num_waypoints,
    )

    print("\n" + "=" * 60)
    print("INTERPOLATION DEVIATION METRICS (% of robot length)")
    print("=" * 60)
    _print_metrics("Tip", tip_dev)
    _print_metrics("Shape", shape_dev)
    print("=" * 60 + "\n")

    if args.save_history is not None:
        general_follower.save_history_to_file(
            interp_history,
            args.save_history,
            waypoints=interp_waypoints,
            metad={
                "planner": args.planner,
                "curve": args.curve,
                "num_samples": args.num_samples,
                "shape_lib_path": args.shape_lib_path,
                "num_waypoints": args.num_waypoints,
                "interpolation_steps_per_waypoint": args.interpolation_steps,
                "tip_deviation": tip_dev,
                "shape_deviation": shape_dev,
            },
        )

    if args.save_sparse_history is not None:
        general_follower.save_history_to_file(
            history,
            args.save_sparse_history,
            waypoints=waypoints,
            metad={
                "planner": args.planner,
                "curve": args.curve,
                "num_samples": args.num_samples,
                "shape_lib_path": args.shape_lib_path,
                "num_waypoints": args.num_waypoints,
                "note": "sparse pre-interpolation plan (one entry per waypoint)",
            },
        )

    if args.no_animate:
        return

    visualizer = PathVisualizer(robot)
    visualizer.plot_history(history, waypoints, show_animation=True)
    visualizer.plot_history(interp_history, waypoints, show_animation=True, animation_interval=0.01)


if __name__ == "__main__":
    main()
