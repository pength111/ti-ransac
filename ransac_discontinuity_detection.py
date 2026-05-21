"""
TI-RANSAC: Topology-Aware Instance Segmentation for Semi-Automatic
Rock Discontinuity Annotation from 3D Point Clouds.

Core mechanisms:
1. Randomized Hough Transform (RHT) normal estimation — voting replaces PCA to
   preserve sharp edges without smoothing artifacts
2. Surface variation (curvature) constraint — filters out high-curvature
   non-planar features, reducing false plane generation at the source
3. Curvature-guided seed point selection — only locally flat points participate
   in random sampling
4. Dual geometric constraint inlier filtering — spatial distance + normal
   vector consistency
5. Euclidean connectivity topology verification (DBSCAN) — eliminates
   "infinite plane" and "false plane" topological artifacts
6. PCA refitting + RMSE evaluation — globally optimal analytical solution
"""

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import math


class TIRANSAC:
    """
    TI-RANSAC: Topology-Aware Instance Segmentation RANSAC.

    Key extensions over standard RANSAC:
    1. Randomized Hough Transform (RHT) normal estimation
    2. Curvature-guided seed point selection
    3. Euclidean connectivity topology verification (DBSCAN)
    """

    def __init__(self,
                 voxel_size=0.005,
                 distance_threshold=None,
                 angle_threshold=25.0,
                 min_points=50,
                 pt=0.99,
                 k=3,
                 connectivity_factor=1.5,
                 k2=1.5,
                 rht_neighbors=30,
                 rht_iterations=50,
                 curvature_k=20,
                 n_jobs=-1):
        """
        Initialize TI-RANSAC parameters.

        Args:
            voxel_size: voxel downsampling edge length (m), default 5 mm
            distance_threshold: point-to-plane distance threshold (m),
                auto-computed if None
            angle_threshold: normal angle threshold (degrees), default 25
            min_points: minimum points per plane instance
            pt: probability threshold for early termination
            k: minimum points to fit a plane (typically 3)
            connectivity_factor: spatial connectivity relaxation factor k1,
                default 1.5
            k2: DBSCAN connectivity relaxation factor, default 1.5
                (dbscan_eps = k2 * distance_threshold)
            rht_neighbors: RHT neighborhood size
            rht_iterations: RHT iterations per point
            curvature_k: neighborhood size for curvature estimation
            n_jobs: number of parallel workers, -1 = all CPU cores
        """
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.angle_threshold = np.radians(angle_threshold)
        self.min_points = min_points
        self.pt = pt
        self.k = k
        self.connectivity_factor = connectivity_factor
        self.k2 = k2
        self.rht_neighbors = rht_neighbors
        self.rht_iterations = rht_iterations
        self.curvature_k = curvature_k
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()

        # Adaptive distance threshold: Tgap = k1 * voxel_size^(1/3)
        if self.distance_threshold is None:
            self.distance_threshold = (
                self.connectivity_factor * (self.voxel_size ** (1/3))
            )
        self.tgap = self.distance_threshold
        # DBSCAN connectivity parameter: dbscan_eps = k2 * distance_threshold
        self.dbscan_eps = self.k2 * self.distance_threshold

        print(f"TI-RANSAC Initialized:")
        print(f"  Voxel size: {self.voxel_size*1000:.1f}mm")
        print(f"  Distance threshold (Tgap): {self.distance_threshold*1000:.1f}mm")
        print(f"  Angle threshold: {angle_threshold:.1f}deg")
        print(f"  Connectivity factors: "
              f"k1={self.connectivity_factor}, k2={self.k2}")
        print(f"  DBSCAN eps: {self.dbscan_eps*1000:.1f}mm")
        print(f"  Worker threads: {self.n_jobs}")

    # ------------------------------------------------------------------
    # 1. Voxel Grid Downsampling
    # ------------------------------------------------------------------

    def voxel_downsample(self, points):
        """
        Voxel grid downsampling to uniform spatial resolution L_voxel.

        Computes the centroid of points within each occupied voxel:
            p_voxel = (1/N) * Σ p_i

        Uses structured arrays + np.unique + bincount for O(N log N)
        centroid computation, avoiding the Python object overhead of
        np.split.

        Args:
            points: Nx3 point cloud array

        Returns:
            downsampled: Mx3 downsampled point cloud (M << N)
        """
        start_t = time.time()
        n_orig = len(points)

        voxel_idx = np.floor(points / self.voxel_size).astype(np.int32)

        voxel_dtype = np.dtype([
            ('x', np.int32), ('y', np.int32), ('z', np.int32)
        ])
        voxel_keys = np.empty(n_orig, dtype=voxel_dtype)
        voxel_keys['x'] = voxel_idx[:, 0]
        voxel_keys['y'] = voxel_idx[:, 1]
        voxel_keys['z'] = voxel_idx[:, 2]

        _, inverse, counts = np.unique(
            voxel_keys, return_inverse=True, return_counts=True
        )

        sum_x = np.bincount(inverse, weights=points[:, 0])
        sum_y = np.bincount(inverse, weights=points[:, 1])
        sum_z = np.bincount(inverse, weights=points[:, 2])

        n_voxels = len(counts)
        downsampled = np.empty((n_voxels, 3), dtype=np.float64)
        downsampled[:, 0] = sum_x / counts
        downsampled[:, 1] = sum_y / counts
        downsampled[:, 2] = sum_z / counts

        elapsed = time.time() - start_t
        print(f"Voxel downsampling: {n_orig} -> {n_voxels} points "
              f"({elapsed:.2f}s, ratio {n_voxels/n_orig*100:.1f}%)")
        return downsampled

    # ------------------------------------------------------------------
    # 2. RHT Normal Estimation
    # ------------------------------------------------------------------

    def _compute_normals_rht_batch(self, point_indices, points, k_neighbors,
                                    n_iters, kdtree):
        """
        Vectorized batch RHT normal computation (used by worker threads).

        Replaces the per-point Python loop over n_iters with a single
        numpy vectorized triplet sampling, cross-product, and accumulator
        voting pass.
        """
        batch_normals = np.zeros((len(point_indices), 3))
        n_theta = 18
        n_phi = 36
        theta_bin_size = 5.0
        phi_bin_size = 10.0

        for batch_idx, pt_idx in enumerate(point_indices):
            point = points[pt_idx]
            _, indices = kdtree.query([point], k=k_neighbors)
            neighbors = points[indices[0]]
            n_neighbors = len(neighbors)

            if n_neighbors < 3:
                continue

            # Vectorized: sample n_iters triplets in one shot
            triplet_idx = np.random.randint(0, n_neighbors, size=(n_iters, 3))
            dup = (triplet_idx[:, 0] == triplet_idx[:, 1]) | \
                  (triplet_idx[:, 0] == triplet_idx[:, 2]) | \
                  (triplet_idx[:, 1] == triplet_idx[:, 2])
            if np.any(dup):
                triplet_idx[dup] = np.random.randint(
                    0, n_neighbors, size=(np.sum(dup), 3)
                )

            p = neighbors[triplet_idx]          # (n_iters, 3, 3)
            v1 = p[:, 1] - p[:, 0]
            v2 = p[:, 2] - p[:, 0]
            normals = np.cross(v1, v2)           # (n_iters, 3)

            norms = np.linalg.norm(normals, axis=1)
            valid = norms > 1e-12
            normals[valid] /= norms[valid, np.newaxis]

            # Orient to upper hemisphere
            flip = normals[:, 2] < 0
            normals[flip] = -normals[flip]

            # Spherical coordinate conversion (vectorized)
            nz = np.clip(normals[:, 2], 0.0, 1.0)
            theta = np.degrees(np.arccos(nz))
            phi = np.degrees(np.arctan2(normals[:, 1], normals[:, 0])) % 360.0

            t_bins = np.clip(
                (theta / theta_bin_size).astype(int), 0, n_theta - 1
            )
            p_bins = np.clip(
                (phi / phi_bin_size).astype(int), 0, n_phi - 1
            )

            accumulator = np.zeros((n_theta, n_phi))
            np.add.at(accumulator, (t_bins[valid], p_bins[valid]), 1)

            max_idx = np.argmax(accumulator)
            t_peak, p_peak = np.unravel_index(max_idx, accumulator.shape)
            theta_peak = (t_peak + 0.5) * theta_bin_size
            phi_peak = (p_peak + 0.5) * phi_bin_size

            phi_rad = math.radians(phi_peak)
            theta_rad = math.radians(theta_peak)
            batch_normals[batch_idx] = [
                math.sin(theta_rad) * math.cos(phi_rad),
                math.sin(theta_rad) * math.sin(phi_rad),
                math.cos(theta_rad)
            ]

        return point_indices, batch_normals

    def compute_normals(self, points, k_neighbors=None, rht_iterations=None):
        """
        Compute point cloud normals using Randomized Hough Transform (RHT).

        Args:
            points: Nx3 point cloud array
            k_neighbors: RHT neighborhood size (default: self.rht_neighbors)
            rht_iterations: RHT iterations per point
                (default: self.rht_iterations)

        Returns:
            normals: Nx3 normal vector array
        """
        k_neighbors = k_neighbors or self.rht_neighbors
        rht_iterations = rht_iterations or self.rht_iterations

        print(f"Computing RHT normals for {len(points)} points "
              f"(neighbors={k_neighbors}, iters={rht_iterations}, "
              f"workers={self.n_jobs})...")
        start_t = time.time()

        kdtree = KDTree(points)
        normals = np.zeros_like(points)

        indices = np.arange(len(points))
        chunk_size = max(500, len(points) // (self.n_jobs * 4))
        chunks = [
            indices[i:i + chunk_size]
            for i in range(0, len(indices), chunk_size)
        ]

        processed = 0
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(
                    self._compute_normals_rht_batch,
                    chunk, points, k_neighbors, rht_iterations, kdtree
                )
                for chunk in chunks
            ]
            for future in as_completed(futures):
                idx_chunk, data_chunk = future.result()
                normals[idx_chunk] = data_chunk
                processed += len(idx_chunk)
                if processed % 10000 == 0 or processed == len(points):
                    print(f"  Processed {processed}/{len(points)} points",
                          end='\r')

        print(f"\nRHT normals computed in {time.time() - start_t:.2f}s")
        return normals

    # ------------------------------------------------------------------
    # 3. Surface Variation (Curvature)
    # ------------------------------------------------------------------

    def compute_curvature(self, points, k_neighbors=None):
        """
        Compute surface variation (curvature) for each point.

        Based on eigenvalues of the local covariance matrix:
            σ_λ = λ₀ / (λ₀ + λ₁ + λ₂)
        where λ₀ ≤ λ₁ ≤ λ₂ are the eigenvalues.

        Args:
            points: Nx3 point cloud array
            k_neighbors: number of neighboring points

        Returns:
            curvature: length-N curvature array
        """
        k_neighbors = k_neighbors or self.curvature_k
        print(f"Computing surface variation for {len(points)} points "
              f"(neighbors={k_neighbors})...")
        start_t = time.time()

        kdtree = KDTree(points)
        curvature = np.zeros(len(points))

        for i in range(len(points)):
            _, indices = kdtree.query([points[i]], k=k_neighbors)
            neighbors = points[indices[0]]

            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov = np.dot(centered.T, centered) / k_neighbors

            eigenvalues, _ = np.linalg.eigh(cov)
            eigenvalues = np.sort(eigenvalues)

            total = np.sum(eigenvalues)
            if total > 1e-12:
                curvature[i] = eigenvalues[0] / total

        elapsed = time.time() - start_t
        print(f"Surface variation computed in {elapsed:.2f}s")
        return curvature

    # ------------------------------------------------------------------
    # 4. Plane Fitting & Distance
    # ------------------------------------------------------------------

    def fit_plane(self, points):
        """Fit plane via SVD: Ax + By + Cz + D = 0."""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        d = -np.dot(normal, centroid)
        return np.append(normal, d)

    def point_to_plane_distance(self, points, plane_params):
        """Compute orthogonal point-to-plane distance."""
        A, B, C, D = plane_params
        denom = np.sqrt(A*A + B*B + C*C)
        if denom < 1e-12:
            return np.full(len(points), 1e12)
        return np.abs(A * points[:, 0] + B * points[:, 1] +
                      C * points[:, 2] + D) / denom

    # ------------------------------------------------------------------
    # 5. Octree Spatial Local Sampling & Curvature-Guided Seed Selection
    # ------------------------------------------------------------------

    def build_octree(self, points, max_depth=8):
        """Build an octree for spatial local sampling."""
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        octree = {
            'bounds': (min_bounds, max_bounds),
            'points': list(range(len(points))),
            'depth': 0
        }
        self._subdivide_octree(octree, points, max_depth)
        return octree

    def _subdivide_octree(self, node, points, max_depth):
        """Recursively subdivide an octree node."""
        if node['depth'] >= max_depth or len(node['points']) < 8:
            return
        min_bounds, max_bounds = node['bounds']
        center = (min_bounds + max_bounds) / 2
        node['children'] = []
        for i in range(8):
            child_min, child_max = min_bounds.copy(), max_bounds.copy()
            if i & 1:
                child_min[0] = center[0]
            else:
                child_max[0] = center[0]
            if i & 2:
                child_min[1] = center[1]
            else:
                child_max[1] = center[1]
            if i & 4:
                child_min[2] = center[2]
            else:
                child_max[2] = center[2]
            p_indices = np.array(node['points'])
            pts = points[p_indices]
            mask = np.all((pts >= child_min) & (pts <= child_max), axis=1)
            child_points = p_indices[mask].tolist()
            if len(child_points) > 0:
                child = {
                    'bounds': (child_min, child_max),
                    'points': child_points,
                    'depth': node['depth'] + 1
                }
                node['children'].append(child)
                self._subdivide_octree(child, points, max_depth)

    def get_octree_cell(self, octree, point_idx, points, target_depth=None):
        """Return indices of points in the octree cell containing point_idx."""
        if target_depth is None:
            target_depth = np.random.randint(1, 8)
        node = octree
        while 'children' in node and node['depth'] < target_depth:
            for child in node['children']:
                if point_idx in child['points']:
                    node = child
                    break
        return node['points']

    def select_seeds_by_curvature(self, points, curvature):
        """
        Curvature-guided seed point selection.

        Adaptively filters out high-curvature points based on the
        statistical distribution of surface variation. Only locally
        flat points participate in random sampling.
        """
        if curvature is None or len(curvature) < 10:
            return np.arange(len(points))

        mean_curv = np.mean(curvature)
        std_curv = np.std(curvature)
        threshold = mean_curv + std_curv

        flat_mask = curvature < threshold
        flat_indices = np.where(flat_mask)[0]

        if len(flat_indices) < self.k:
            return np.arange(len(points))

        return flat_indices

    def local_sampling(self, points, octree, flat_indices):
        """
        Octree-based spatial local sampling.

        The k = 3 seed points are randomly sampled from the same octree
        leaf node, ensuring the candidate plane originates from a
        spatially coherent local neighborhood.
        """
        if len(flat_indices) < self.k:
            return np.random.choice(len(points), self.k, replace=False)

        p1_idx = np.random.choice(flat_indices)
        cell_points = self.get_octree_cell(octree, p1_idx, points)
        cell_flat = np.intersect1d(cell_points, flat_indices)

        if len(cell_flat) < self.k:
            other = np.random.choice(
                flat_indices, self.k - 1, replace=False
            )
        else:
            other = np.random.choice(
                cell_flat, self.k - 1, replace=False
            )
        return np.array([p1_idx] + list(other))

    # ------------------------------------------------------------------
    # 6. Dual Geometric Constraint Inlier Screening
    # ------------------------------------------------------------------

    def screen_inliers(self, points, normals, plane_params):
        """
        Dual geometric constraint inlier screening.

        Constraint 1 - Spatial distance: |dist(p_i, Plane)| < ε_d
        Constraint 2 - Normal consistency:
            arccos(|n_i · n_model|) < T_angle

        Args:
            points: Nx3 point cloud
            normals: Nx3 normal vectors
            plane_params: (A, B, C, D) plane parameters

        Returns:
            candidate_indices: indices of points satisfying both constraints
        """
        # Constraint 1: spatial distance
        distances = self.point_to_plane_distance(points, plane_params)
        dist_mask = distances < self.distance_threshold

        # Constraint 2: normal vector consistency
        plane_normal = plane_params[:3]
        dot_products = np.abs(np.dot(normals, plane_normal))
        angle_mask = (
            np.arccos(np.clip(dot_products, -1.0, 1.0)) < self.angle_threshold
        )

        candidate_mask = dist_mask & angle_mask
        return np.where(candidate_mask)[0]

    # ------------------------------------------------------------------
    # 7. Euclidean Connectivity Topology Verification (DBSCAN)
    # ------------------------------------------------------------------

    def topology_review_dbscan(self, inlier_points, inlier_indices):
        """
        Euclidean connectivity topology verification.

        Applies DBSCAN to the candidate inlier set, discards isolated
        noise (label = -1), and retains the largest connected component
        as the final topological instance.

        Args:
            inlier_points: candidate inlier point cloud
            inlier_indices: indices of inliers in the original array

        Returns:
            valid_indices: topology-verified point indices (in original
                array)
        """
        if len(inlier_points) < self.min_points:
            return np.array([])

        clustering = DBSCAN(
            eps=self.dbscan_eps, min_samples=3, n_jobs=-1
        ).fit(inlier_points)
        labels = clustering.labels_

        unique_labels = np.unique(labels)
        valid_labels = [l for l in unique_labels if l != -1]

        if len(valid_labels) == 0:
            return np.array([])

        largest_label = max(valid_labels, key=lambda l: np.sum(labels == l))
        largest_mask = labels == largest_label

        return inlier_indices[largest_mask]

    # ------------------------------------------------------------------
    # 8. PCA Refitting & RMSE
    # ------------------------------------------------------------------

    def pca_refit(self, points):
        """
        PCA least-squares refitting for globally optimal plane parameters.

        Constructs the covariance matrix and extracts the eigenvector
        corresponding to the smallest eigenvalue as the optimal normal.
        """
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov_matrix = np.dot(centered.T, centered) / len(points)

        _, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, 0]
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        d = -np.dot(normal, centroid)

        return np.append(normal, d)

    def compute_rmse(self, points, plane_params):
        """
        Compute Root Mean Square Error:
            RMSE = sqrt((1/N) * Σ(p_j · n_model + d_model)²)
        """
        A, B, C, D = plane_params
        denom = np.sqrt(A*A + B*B + C*C)
        if denom < 1e-12:
            return float('inf')
        distances = np.abs(A * points[:, 0] + B * points[:, 1] +
                           C * points[:, 2] + D) / denom
        return float(np.sqrt(np.mean(distances ** 2)))

    # ------------------------------------------------------------------
    # 9. Probability
    # ------------------------------------------------------------------

    def compute_probability(self, n, N, S):
        """Compute the probability of having detected a valid plane."""
        if N == 0 or N < self.k:
            return 0
        p_n = (n / N) ** self.k
        probability = 1 - (1 - p_n) ** S
        return probability

    # ------------------------------------------------------------------
    # 10. Main Plane Extraction Pipeline
    # ------------------------------------------------------------------

    def _ransac_search_batch(self, points, normals, flat_indices, octree,
                              iterations):
        """
        Run a fixed number of RANSAC iterations in a single thread.

        DBSCAN is executed during scoring inside each iteration, and the
        candidate is evaluated by the size of its largest connected
        component. This eliminates over-segmentation at the source.
        """
        local_candidates = []
        for _ in range(iterations):
            sample_indices = self.local_sampling(points, octree, flat_indices)
            sample_points = points[sample_indices]

            plane_params = self.fit_plane(sample_points)

            candidate_inliers_idx = self.screen_inliers(
                points, normals, plane_params
            )

            if len(candidate_inliers_idx) < self.min_points:
                continue

            candidate_pts = points[candidate_inliers_idx]
            valid_idx = self.topology_review_dbscan(
                candidate_pts, candidate_inliers_idx
            )

            if len(valid_idx) >= self.min_points:
                local_candidates.append({
                    'params': plane_params,
                    'inlier_indices': valid_idx,
                    'score': len(valid_idx)
                })

        return local_candidates

    def extract_planes(self, points, normals, curvature=None,
                       max_iterations=10000):
        """
        Extract all discontinuity instances (core TI-RANSAC pipeline).

        Args:
            points: Nx3 point cloud array
            normals: Nx3 normal vector array
            curvature: length-N curvature array (optional)
            max_iterations: maximum RANSAC iterations per round

        Returns:
            planes: list of discontinuity instance dicts
            labels: point label array (-1 = unassigned)
        """
        print(f"\n{'='*60}")
        print("TI-RANSAC Discontinuity Extraction")
        print(f"{'='*60}")
        print(f"Total points: {len(points)}")
        print(f"Parameters: "
              f"eps_d={self.distance_threshold*1000:.1f}mm, "
              f"T_angle={math.degrees(self.angle_threshold):.1f}deg, "
              f"dbscan_eps={self.dbscan_eps*1000:.1f}mm")

        remaining_mask = np.ones(len(points), dtype=bool)
        planes = []
        labels = np.full(len(points), -1, dtype=int)
        plane_id = 0

        while np.sum(remaining_mask) >= self.min_points:
            remaining_idx = np.where(remaining_mask)[0]
            remaining_pts = points[remaining_mask]
            remaining_normals = normals[remaining_mask]
            remaining_curvature = (
                curvature[remaining_mask] if curvature is not None else None
            )

            print(f"\n--- Round {plane_id + 1} --- "
                  f"(remaining points: {len(remaining_pts)})")

            flat_indices = self.select_seeds_by_curvature(
                remaining_pts, remaining_curvature
            )
            print(f"  Flat seeds: {len(flat_indices)}/{len(remaining_pts)} "
                  f"({100*len(flat_indices)/len(remaining_pts):.1f}%)")

            octree = self.build_octree(remaining_pts)

            candidates = []
            start_time = time.time()

            batch_count = 10
            iters_per_batch = max(10, max_iterations // batch_count)
            iters_per_thread = max(1, iters_per_batch // self.n_jobs)

            found_good_plane = False
            total_iters_run = 0

            for b in range(batch_count):
                batch_candidates = []

                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = [
                        executor.submit(
                            self._ransac_search_batch,
                            remaining_pts, remaining_normals,
                            flat_indices, octree, iters_per_thread
                        )
                        for _ in range(self.n_jobs)
                    ]
                    for future in as_completed(futures):
                        batch_candidates.extend(future.result())

                candidates.extend(batch_candidates)
                total_iters_run += iters_per_thread * self.n_jobs

                if len(candidates) > 0:
                    best = max(candidates, key=lambda x: x['score'])
                    prob = self.compute_probability(
                        best['score'], len(remaining_pts), total_iters_run
                    )
                    if prob > self.pt:
                        print(f"  Best candidate: "
                              f"inliers={best['score']}, prob={prob:.4f}")
                        found_good_plane = True
                        break

                print(f"  Batch {b+1}/{batch_count} done, "
                      f"candidates: {len(candidates)}", end='\r')

            if not found_good_plane and len(candidates) == 0:
                print("\nNo more planes found, stopping.")
                break

            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidate = candidates[0]

            valid_local_idx = best_candidate['inlier_indices']
            valid_global_idx = remaining_idx[valid_local_idx]
            instance_pts = points[valid_global_idx]
            refit_params = self.pca_refit(instance_pts)

            rmse = self.compute_rmse(instance_pts, refit_params)

            planes.append({
                'id': plane_id,
                'params': refit_params,
                'initial_params': best_candidate['params'],
                'inliers': valid_global_idx,
                'count': len(valid_global_idx),
                'rmse': rmse
            })

            labels[valid_global_idx] = plane_id

            elapsed = time.time() - start_time
            print(f"\n  >> Plane {plane_id}: {len(valid_global_idx)} points, "
                  f"RMSE={rmse*1000:.2f}mm, time {elapsed:.2f}s")

            remaining_mask[valid_global_idx] = False
            plane_id += 1

        print(f"\n{'='*60}")
        print("TI-RANSAC Extraction Complete")
        print(f"Total instances: {len(planes)}")
        print(f"Classified: {np.sum(labels >= 0)} points "
              f"({100*np.sum(labels >= 0)/len(points):.1f}%)")
        print(f"Unclassified: {np.sum(labels == -1)} points "
              f"({100*np.sum(labels == -1)/len(points):.1f}%)")
        print(f"{'='*60}")

        return planes, labels
