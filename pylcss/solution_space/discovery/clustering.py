# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

class SeedClusteringMixin:
    def _cluster_seeds_hdbscan(self, seeds_phys: np.ndarray) -> List[np.ndarray]:
        """
        Cluster deflation seed points directly using HDBSCAN.
        
        This method groups nearby seeds into clusters and treats isolated seeds 
        (HDBSCAN noise) as individual single-point basins.
        
        Args:
            seeds_phys: Feasible seed points in physical units (dim, n_seeds)
            
        Returns:
            List of cluster sample arrays, each of shape (dim, n_cluster_seeds)
        """
        n_seeds = seeds_phys.shape[1]
        
        if n_seeds <= 1:
            # Single seed → single cluster
            return [seeds_phys]
        
        # Normalize seeds to [0,1] for distance calculation
        seeds_norm = ((seeds_phys - self.active_dsl.reshape(-1, 1)) 
                      / self.active_dv_norm.reshape(-1, 1))
        X = seeds_norm.T  # (n_seeds, dim) for sklearn
        
        # Optional dimensionality reduction before HDBSCAN in high dimensions.
        #if self.dim > 15 and n_seeds >= 15:
        #    try:
        #        import umap
        #        logger.info(f"  Applying UMAP projection (dim={self.dim} > 15, seeds={n_seeds} >= 15)")
        #        reducer = umap.UMAP(n_components=5, random_state=42)
        #        X = reducer.fit_transform(X)
        #    except ImportError:
        #        logger.warning("umap-learn not available. Falling back to original space.")
        
        try:
            import hdbscan

            min_cluster_size = int(getattr(self.params, "min_cluster_size", 0))

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=None,
                metric='euclidean',
                allow_single_cluster=False
            )

            labels = clusterer.fit_predict(X)
            
            unique_labels = set(labels)
            n_noise = np.sum(labels == -1)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            logger.info(
                f"  HDBSCAN seed clustering: "
                f"{n_clusters} clusters, {n_noise} noise (individual basins), "
                f"min_cluster_size={min_cluster_size}"
            )
            
            clusters = []
            
            # Add proper clusters
            for label in sorted(unique_labels):
                if label == -1:
                    continue
                mask = labels == label
                cluster_seeds = seeds_phys[:, mask]
                clusters.append(cluster_seeds)
                logger.info(f"    Cluster {label}: {np.sum(mask)} seeds")
            
            # Add noise points as individual single-point basins
            noise_mask = labels == -1
            noise_indices = np.where(noise_mask)[0]
            for idx in noise_indices:
                single_seed = seeds_phys[:, idx:idx+1]  # Keep (dim, 1) shape
                clusters.append(single_seed)
                logger.info(f"    Noise seed {idx}: individual basin")
            
            # K is unknown and normally inferred by HDBSCAN. An explicit
            # max_modes/max_clusters value is only a compatibility safety cap.
            max_modes = int(getattr(self.params, "max_modes", 0))
            if max_modes <= 0:
                max_modes = int(getattr(self.params, "max_clusters", 0))
            if max_modes > 0 and len(clusters) > max_modes:
                clusters.sort(key=lambda c: c.shape[1], reverse=True)
                clusters = clusters[:max_modes]
            
            return clusters
            
        except ImportError:
            logger.warning("hdbscan not available. Treating each seed as individual basin.")
            return [seeds_phys[:, i:i+1] for i in range(n_seeds)]


