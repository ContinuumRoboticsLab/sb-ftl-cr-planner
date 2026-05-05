import numpy as np
import time
from src.RobotModels import ContinuumRobotModel
from src.utils.ShapeConstraints import ConvexShape
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict
from src.MotionPlanners.GeneralMotionPlanner import GeneralMotionPlanner
import src.utils.CurveUtils as CurveUtils


class ThresholdCluster(GeneralMotionPlanner):
    """Sampling-based path follower with thresholded similarity clustering.

    Clusters are formed by connecting shapes whose flattened-shape distance
    is below a fixed threshold, producing connected components that are
    independent of index ordering and library size.`similarity_threshold`
    controls grouping strictness.
    """
    
    def __init__(self, robot: ContinuumRobotModel, 
                 similarity_threshold: float | None = None,
                 **kwargs):
        super().__init__(robot, **kwargs)

        # Determine threshold: use provided value or estimate from k-th neighbor distances
        if similarity_threshold is None:
            X = np.array([s["shape_points"].flatten() for s in self.shape_library])
            n = len(X)
            if n > 1:
                # Target a small local neighborhood to avoid giant components
                k_target = 8
                k = max(1, min(k_target, n - 1))
                nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
                dists, _ = nn.kneighbors(X)
                # Use a lower percentile of the k-th neighbor distance to keep neighborhoods tight
                kth = dists[:, k]
                self.similarity_threshold = float(np.percentile(kth, 90))
            else:
                self.similarity_threshold = 0.0
        else:
            self.similarity_threshold = float(similarity_threshold)

        self.clusters = self.build_threshold_clusters(self.shape_library, threshold=self.similarity_threshold)
        # Compute single global stat: per-node mean intra-cluster distance across all clusters
        global_node_means = []
        for c in self.clusters:
            m = len(c)
            if m < 2:
                continue  # skip singletons
            pts = [self.shape_library[i]['shape_points'] for i in c]
            D = np.zeros((m, m))
            for a in range(m):
                pa = pts[a]
                for b in range(a + 1, m):
                    pb = pts[b]
                    d = np.mean(np.linalg.norm(pa - pb, axis=1))
                    D[a, b] = D[b, a] = d
            per_node_means = D.sum(axis=1) / (m - 1)
            global_node_means.append(per_node_means)
        if len(global_node_means) > 0:
            concat_means = np.concatenate(global_node_means)
            g_avg = float(np.mean(concat_means))
            g_min = float(np.min(concat_means))
            g_max = float(np.max(concat_means))
            print(f"[ClusterStats] global avg_node_mean_dist={g_avg:.4f} min_node_mean_dist={g_min:.4f} max_node_mean_dist={g_max:.4f} n_nodes={concat_means.size} n_clusters={len(self.clusters)}")

        self.cluster_centers = [self.get_central_node_for_cluster(c) for c in self.clusters]

    def __repr__(self):
        return "Threshold Cluster"


    def build_threshold_clusters(self, shape_library, threshold: float) -> List[List[int]]:
        """
        Build clusters as connected components using a fixed similarity threshold.

        We compute neighbors within `threshold` via radius search, then BFS to
        form connected components. This avoids order-dependent greedy grouping
        and keeps cluster membership stable as the library grows.
        """
        X = np.array([s["shape_points"].flatten() for s in shape_library])
        n = len(X)
        if n == 0:
            return []

        # Use radius neighbors to get local balls; avoid transitive chaining
        nn = NearestNeighbors(metric="euclidean").fit(X)
        adj_indices = nn.radius_neighbors(X, radius=threshold, return_distance=False)

        visited = np.zeros(n, dtype=bool)
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            # Local ball around seed i (no transitive expansion)
            local = [idx for idx in adj_indices[i] if idx != i]
            comp = [i] + local
            for v in comp:
                visited[v] = True

            clusters.append(comp)

        try:
            sizes = [len(c) for c in clusters]
            #print(f"[ThresholdClustering] threshold={threshold:.4f}, clusters={len(clusters)}, avg_size={np.mean(sizes):.1f}, max_size={np.max(sizes)}, min_size={np.min(sizes)}")
        except Exception:
            pass

        return clusters

    def get_central_node_for_cluster(self, cluster: List[int]) -> int:
        """Return index of most central node in a cluster by mean intra-cluster distance."""
        shape_points = [self.shape_library[i]['shape_points'] for i in cluster]
        m = len(cluster)
        if m == 1:
            return cluster[0]
        D = np.zeros((m, m))
        for a in range(m):
            for b in range(a + 1, m):
                d = np.mean(np.linalg.norm(shape_points[a] - shape_points[b], axis=1))
                D[a, b] = D[b, a] = d
        return cluster[np.argmin(D.mean(axis=0))]

    def follow_path(self, waypoints: np.ndarray, verbose: bool = True, base_constraint: ConvexShape = None ) -> List[Dict]:
        """
        Follow a path using 3-point matching with threshold-clustered shape library.
        """
        history = []

        clark_coords = np.zeros(self.robot.num_segments * 2)
        base_transform = np.eye(4)
        previous_base_pos = None
        previous_base_rot = None

        waypoint_index = 0
        cum_arc_length = []
        while waypoint_index < len(waypoints):
            step_start_time = time.time()

            active_waypoints = waypoints[:waypoint_index + 1]
            target_tip = waypoints[waypoint_index]

            if verbose:
                print(f"\nStep {len(history) + 1}: Target waypoint {waypoint_index + 1}/{len(waypoints)}")
                if waypoint_index >= 2 and previous_base_pos is not None:
                    print(f"  Base stability weight: {self.base_stability_weight} (active)")
                elif waypoint_index < 2:
                    print(f"  Base stability weight: {self.base_stability_weight} (disabled for startup)")

            best_score = float('inf')
            best_shape = None
            best_base_pos = None
            best_base_rot = None

            shape_eval_function = self.evaluate_shape

            cum_arc_length.append(CurveUtils.compute_arc_length(active_waypoints))

            best_cluster_id = None
            best_cluster_score = float('inf')
            # First pass: evaluate cluster centers
            for c_id, clust_cent in enumerate(self.cluster_centers):
                shape_info = self.shape_library[clust_cent]
                score, base_pos, base_rot = shape_eval_function(
                    shape_info, active_waypoints, target_tip, previous_base_pos,
                    previous_base_rot, waypoint_index, cum_arc_length
                )
                if score < best_cluster_score:
                    best_cluster_score = score
                    best_cluster_id = c_id
                    if score < best_score:
                        best_score = score
                        best_shape = shape_info
                        best_base_pos = base_pos
                        best_base_rot = base_rot

            # Second pass: search within chosen cluster
            if best_cluster_id is not None:
                for node_id in self.clusters[best_cluster_id]:
                    shape_info = self.shape_library[node_id]
                    score, base_pos, base_rot = shape_eval_function(
                        shape_info, active_waypoints, target_tip, previous_base_pos,
                        previous_base_rot, waypoint_index, cum_arc_length
                    )
                    if score < best_score:
                        best_score = score
                        best_shape = shape_info
                        best_base_pos = base_pos
                        best_base_rot = base_rot

            if best_shape is None:
                raise RuntimeError("No valid shape found during sampling.")

            clark_coords = best_shape['clark_coords'].copy()
            selected_shape_info = best_shape

            base_transform = np.eye(4)
            base_transform[:3, :3] = best_base_rot
            base_transform[:3, 3] = best_base_pos

            endpoints, shape_points, tip_SE3 = self.robot.forward_kinematics_from_base(
                clark_coords, best_base_pos, best_base_rot
            )
            actual_tip = endpoints[-1]
            tip_error = np.linalg.norm(actual_tip - target_tip)

            max_deviation, waypoint_deviations = self.compute_deviation_metric(active_waypoints, shape_points)

            #if tip_error < self.activation_radius:
            waypoint_index += 1
            #else:
            #    print(f'Tip error: {tip_error}')

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
                'shape_proximity': max_deviation,
                'max_deviation': max_deviation,
                'waypoint_deviations': waypoint_deviations,
                'sampling_score': best_score,
                'selected_shape_info': selected_shape_info,
                'computation_time': step_time,
            }

            history.append(history_entry)

            previous_base_pos = best_base_pos.copy()
            previous_base_rot = best_base_rot.copy()

            if verbose:
                print(f"\nStep {len(history)}: Target waypoint {waypoint_index}/{len(waypoints)}")
                print(f"  Total step time: {step_time:.4f}s")

        history = self._fixup_initial_waypoints(history, waypoints)
        return history
