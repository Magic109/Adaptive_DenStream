import sys
import numpy as np
from copy import copy
from MicroCluster import MicroCluster
from math import ceil
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

class DenStream:

    def __init__(self, lambd=1, eps=1, beta=0.2, mu=4,dbscan_eps=2,init_size=100, stream_speed=1):
        """
        DenStream - Density-Based Clustering over an Evolving Data Stream with
        Noise.

        Parameters
        ----------
        lambd: float, optional
            The forgetting factor. The higher the value of lambda, the lower
            importance of the historical data compared to more recent data.
        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.

        Attributes
        ----------
        labels_ : array, shape = [n_samples]
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.

        Notes
        -----


        References
        ----------
        Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-Based
        Clustering over an Evolving Data Stream with Noise.
        """
        self.lambd = lambd
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.dbscan_eps = dbscan_eps
        self.t = 0
        self.stream_speed = stream_speed
        self.point_count = 0
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        self.init_size = init_size
        self.is_initialized = False
        self._init_buffer_X = []
        self._init_buffer_w = []
        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log2((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Online learning.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """

        #X = check_array(X, dtype=np.float64, order="C")
        (n_samples,_) = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # 2. Intercept points for initialization
        if not self.is_initialized:
            for x, w in zip(X, sample_weight):
                self._init_buffer_X.append(x)
                self._init_buffer_w.append(w)
                #self.t += 1
            
            # If buffer is full, trigger initialization!
            if len(self._init_buffer_X) >= self.init_size:
                self.initialize_offline(np.array(self._init_buffer_X), np.array(self._init_buffer_w))
                self.is_initialized = True
                
                # Delete buffers
                self._init_buffer_X = [] 
                self._init_buffer_w = []
            return self # Skip the rest of partial_fit until initialized

        #normal online phase
        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
        return self

    def _get_nearest_micro_cluster(self, sample, micro_clusters):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            current_distance = np.linalg.norm(micro_cluster.center() - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample, weight,self.t)
            if micro_cluster_copy.radius() <= self.eps:
                micro_cluster.insert_sample(sample, weight,self.t)
                return True
        return False

    def _merging(self, sample, weight):
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, weight, nearest_p_micro_cluster)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, weight, nearest_o_micro_cluster)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, self.t)
                micro_cluster.insert_sample(sample, weight,self.t)
                self.o_micro_clusters.append(micro_cluster)

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def _partial_fit(self, sample, weight):
        self._merging(sample, weight)
        if self.t % self.tp == 0:
            #print('updating weigths {}'.format(self.t))
            for p in self.p_micro_clusters:
                p.update_weight(self.t)
            # ---> ADD THESE TWO EXACT LINES HERE <---
            for o in self.o_micro_clusters:
                o.update_weight(self.t)
            # ----------------------------------------  
            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                     in self.p_micro_clusters if
                                     p_micro_cluster.weight() >= self.beta *
                                     self.mu]
            Xis = [((self._decay_function(self.t - o_micro_cluster.creation_time
                                          + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in
                   self.o_micro_clusters]
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]                  
        self.point_count += 1
        if self.point_count % self.stream_speed == 0:
            self.t += 1
         
    

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                    order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")
        return sample_weight

    

    def cluster_p_mcs(self):
        if len(self.p_micro_clusters) == 0:
            return []

        centers = np.array([p.center() for p in self.p_micro_clusters])
        weights = np.array([p.weight() for p in self.p_micro_clusters])

        # 1. CORE-MC DEFINITION: intrinsic weight threshold (DenStream variant)
        core_mask = weights >= self.mu
        self.core_pmc_indices = np.where(core_mask)[0]
        
        labels = np.full(len(centers), -1)
        if len(self.core_pmc_indices) == 0:
            return labels

        # 2. SPATIAL ADJACENCY: Get edges shorter than dbscan_eps
        # This is the "Connectivity" part of DBSCAN
        tree = cKDTree(centers[core_mask])
        
        # Adjacency Matrix (The standard way to represent graphs)
        adj_matrix = tree.sparse_distance_matrix(tree, max_distance=self.dbscan_eps)

        # 3. GRAPH PARTITIONING: The actual clustering step
        n_clusters, core_labels = connected_components(csgraph=adj_matrix, directed=False)
        
        labels[core_mask] = core_labels

        # 4. BORDER ASSIGNMENT: Standard DBSCAN reachability
        # Assign non-cores to their nearest core island
        non_core_idx = np.where(~core_mask)[0]
        if len(non_core_idx) > 0 and len(self.core_pmc_indices) > 0:
            dists, closest_core_rel = tree.query(centers[non_core_idx], k=1)
            valid = dists <= self.dbscan_eps
            labels[non_core_idx[valid]] = core_labels[closest_core_rel[valid]]

        return labels
        
        

    def initialize_offline(self, X_init, sample_weight=None):
        """
        Redesign 3: General-Purpose Sequential Warmup.
        Uses sequential mechanics to avoid the O(N^2) DBSCAN memory spike, 
        but enforces a strict 'Mass Purge' of ungraduated OMCs at the end 
        to prevent the OMC Explosion problem on scattered datasets.
        """
        X_init = np.asarray(X_init)
        n_samples = X_init.shape[0]
        sample_weight = self._validate_sample_weight(sample_weight, n_samples)
        
        for sample, weight in zip(X_init, sample_weight):
            #self._partial_fit(sample, weight)
            self._merging(sample, weight)
            
            
        # 2. THE OUTLIER PURGE
        # Delete all OMCs that failed to reach PMC status during warmup.
        # This guarantees the online stream starts with zero OMC bloat, 
        purged_count = len(self.o_micro_clusters)
        self.o_micro_clusters = [] 
            
        #print(f"[Init] General Warmup complete at t={self.t}.")
        #print(f"[Init] Formed {len(self.p_micro_clusters)} p-MCs. Purged {purged_count} failed o-MCs to save resources.")



    def roller(self, sorted_edges, eps, g=0.3):
        """Single source of truth for the roller. Walks the sorted MST edges and
        returns the fracture index: the first i where sorted_edges[i] exceeds its
        rolling boundary, or len(sorted_edges) if the whole set survives.
        Used by BOTH the cut (Phase 2) and the marriage test (Phase 4) so the two
        can never disagree -> no cut/merge/cut oscillation.
        NOTE: the small-sample d2 constants live here (commented OLD + active NEW);
        swapping them here changes cutting AND marriage consistently."""
        fracture_index = len(sorted_edges)
        for i in range(1, len(sorted_edges)):
            current_edge = sorted_edges[i]
            density_shield = 2 * eps
            core = sorted_edges[:i]

            if len(core) >= 5:
                # PATH A: MATURE SET MATH (Tukey 1.5 IQR)
                q25_roll = np.percentile(core, 25, method='median_unbiased')
                q75_roll = np.percentile(core, 75, method='median_unbiased')
                iqr_eff_roll = max(q75_roll - q25_roll, g * np.median(core), g * eps)
                topo_shield = max(q75_roll, eps) + (1.5 * iqr_eff_roll)
            elif len(core) > 1:
                # PATH B: SMALL SAMPLE MATH
                tukey_k_d2 = 1.31 if len(core) == 4 else 1.59 if len(core) == 3 else 2.39
                #tukey_k_d2 = 0.97 if len(core) == 4 else 1.18 if len(core) == 3 else 1.77
                core_anchor = np.mean(core)
                inner_gap = core[-1] - core[0]
                iqr_eff_rolling = max(inner_gap, g * np.median(core), g * eps)
                topo_shield = max(core_anchor, eps) + (tukey_k_d2 * iqr_eff_rolling)
            else:
                # PATH C: SINGLE EDGE
                topo_shield = 1.5 * max(core[0], eps)

            current_boundary = min(max(topo_shield, density_shield),4*eps)
            if current_edge > current_boundary:
                fracture_index = i
                break
        return fracture_index


    def audit_clusters(self, labels,g=0.3):
            labels = np.array(labels)
            n_pmcs = len(labels)
            if n_pmcs == 0:
                return "keep"
                
            # 1. Identify Valid Clusters (O(N) pass)
            unique_vals, counts = np.unique(labels, return_counts=True)
            
            valid_macro_clusters = [
                val for val, count in zip(unique_vals, counts) 
                if val != -1 and count >= 2
            ]
            
            # Dictionaries to store limits for the Macro-Auditor at the end
            cluster_jump_limits = {} 
            cluster_pmc_indices = {}
            cluster_healthy_edges = {} 
            
            # --- CUT SECTOR TRACKERS ---
            proposed_cuts = []
            # ==========================================
            # 2. THE VOID DETECTOR & BOUNDARY ARCHITECT
            # ========================================== 
    
            for label in valid_macro_clusters:
                    
                idx = np.where(labels == label)[0]
                    
                centers = np.array([self.p_micro_clusters[i].center() for i in idx])
                #weights = np.array([self.p_micro_clusters[i].weight() for i in idx])
                #"""
                tree = cKDTree(centers)
                
                search_radius = self.dbscan_eps * 1.01 
                sparse_dist_matrix = tree.sparse_distance_matrix(tree, max_distance=search_radius)
                # 1. Convert to COO first for instant data access
                dist_coo = sparse_dist_matrix.tocoo()
                # 2. Prevent Scipy from dropping perfectly overlapping PMCs
                dist_coo.data[dist_coo.data == 0] = 1e-9 
                # 3. Build MST
                mst = minimum_spanning_tree(dist_coo)
                mst_coo = mst.tocoo()
                edge_data = mst_coo.data[mst_coo.data > 0]
                    
                #edge_data.sort(key=lambda x: x[0])
                #sorted_edges = np.array([e[0] for e in edge_data])
                sorted_edges = np.sort(edge_data)
                #fracture_index = len(sorted_edges)
                # --- 3. THE CUTTER LOOP (The Roller) ---
                fracture_index = self.roller(sorted_edges, self.eps, g)
                
                # --- POST-PROCESSING ---
                # 1. Isolate the healthy edges
                healthy_edges = sorted_edges[:fracture_index]
                weakest_link = healthy_edges[-1]
                
                # --- CALCULATE THE EXPLICIT MERGE THRESHOLD ---
                if len(healthy_edges) >= 5:
                    q25_h = np.percentile(healthy_edges, 25, method='median_unbiased')
                    q75_h = np.percentile(healthy_edges, 75, method='median_unbiased')
                    iqr_eff_h = max(q75_h - q25_h, g * np.median(healthy_edges), g*self.eps)
                    
                    merge_threshold = max(q75_h, self.eps) + (1.5 * iqr_eff_h)
                    new_merge_threshold = merge_threshold * 0.99
                    
                elif len(healthy_edges) > 1: 
                    
                    d2_mult = 1.31 if len(healthy_edges) == 4 else 1.59 if len(healthy_edges) == 3 else 2.39
                    #d2_mult = 0.97 if len(healthy_edges) == 4 else 1.18 if len(healthy_edges) == 3 else 1.77
                    core_anchor_h = np.mean(healthy_edges)
                    inner_gap_h = healthy_edges[-1] - healthy_edges[0]
                    iqr_eff_h = max(inner_gap_h, g * np.median(healthy_edges), g * self.eps)
                    
                    # 1. The small sample Limit (d2 math)
                    proposed_limit = max(core_anchor_h, self.eps) + (d2_mult * iqr_eff_h)
                    
                    # 2. THE PROMOTION SURVIVAL TEST (ONLY FOR 4 EDGES)
                    if len(healthy_edges) == 4:
                        # If this accepts a gap, it becomes 5 edges next cycle. 
                        # It MUST survive the Mature IQR test.
                        hypo_array = np.append(healthy_edges, proposed_limit)
                        q25_hypo = np.percentile(hypo_array, 25, method='median_unbiased')
                        q75_hypo = np.percentile(hypo_array, 75, method='median_unbiased')
                        iqr_eff_hypo = max(q75_hypo - q25_hypo, g * np.median(hypo_array), g * self.eps)
                        
                        hypo_mature_limit = max(q75_hypo, self.eps) + (1.5 * iqr_eff_hypo)
                        
                        merge_threshold = min(proposed_limit, hypo_mature_limit)
                    else:
                        
                        merge_threshold = proposed_limit
                        
                    new_merge_threshold = merge_threshold * 0.99
    
                else:
                    core_anchor_h = healthy_edges[0]
                    merge_threshold = 1.5 * max(core_anchor_h, self.eps)
                    new_merge_threshold = merge_threshold * 0.99
    
    
                
                clamped_threshold = new_merge_threshold
                physical_merge_floor = max(weakest_link, 2*self.eps)
                merge_limit = max(clamped_threshold, physical_merge_floor)
                limit_cap = 4 * self.eps if len(healthy_edges) == 1 else 4.0 * self.eps
                merge_limit = min(merge_limit, limit_cap)
    
                
                # MACRO-AUDITOR CONSENT LIMIT 
                cluster_jump_limits[label] = merge_limit
                cluster_pmc_indices[label] = idx
                cluster_healthy_edges[label] = healthy_edges 
    
                # ==========================================
                # 3. UNWANTED MERGING DETECTOR (Phase 1: Propose Cut)
                # ========================================== 
                # If roller found a bad bridge, submit it to the cut sector
                if fracture_index < len(sorted_edges):
                    bad_tail = sorted_edges[fracture_index:]
                    for bad_bridge in bad_tail:
                        target_eps = bad_bridge * 0.99
                        proposed_cuts.append((target_eps, label, bad_bridge))
                    
            # [END OF valid_macro_clusters LOOP]
    
            # ==========================================
            # 3.5 CUT SELECTION SECTOR 
            # ==========================================
            if proposed_cuts:
    
                # Dropping to the lowest proposed EPS will sever all identified bad bridges in 1 cycle.
                proposed_cuts.sort(key=lambda x: x[0])
                
                # Execute the deepest cut
                best_cut_eps, bad_label, bad_bridge = proposed_cuts[0]
                self.dbscan_eps = max(best_cut_eps, 2 * self.eps) 
                
                return "changed"
    
                
            # ==========================================
            # 4. MACRO-AUDITOR (The Dual-consent Merge Check)
            # ==========================================
            valid_labels = list(cluster_jump_limits.keys())
            core_pmc_set = set(self.core_pmc_indices) 
            danger_ceiling = float('inf')
            cluster_data = {}
    
            # --- GLOBAL NOISE EXTRACTION (Available to Phases 4, 5 & 6) ---
            invalid_mask = ~np.isin(labels, valid_labels)
            noise_idx = np.where(invalid_mask)[0]
            if len(noise_idx) > 0:
                noise_centers = np.array([self.p_micro_clusters[i].center() for i in noise_idx])
                noise_core_mask = np.array([i in core_pmc_set for i in noise_idx])
                #noise_radii = np.array([self.p_micro_clusters[i].radius() for i in noise_idx])
            else:
                noise_centers = np.array([])
                noise_core_mask = np.array([])
                #noise_radii = np.array([])
            
            if len(valid_labels) > 0:
                for label in valid_labels:
                    idx = cluster_pmc_indices[label]
                    #weights = np.array([self.p_micro_clusters[i].weight() for i in idx])
                    centers = np.array([self.p_micro_clusters[i].center() for i in idx])
                    core_mask = np.array([i in core_pmc_set for i in idx])
                    
                    cluster_data[label] = {
                        #'weights': weights,
                        'centers': centers,
                        #'q50': np.percentile(weights, 50),
                        'core_mask': core_mask
                    }
                # --------------------------------------------------------------
                
                requested_targets = []
                if len(noise_idx) > 0:
                    
                    for label_A in valid_labels:
                        limit_A = cluster_jump_limits[label_A]
                        #idx_A = cluster_pmc_indices[label_A]
                        centers_A = cluster_data[label_A]['centers']
                        #radii_A = np.array([self.p_micro_clusters[i].radius() for i in idx_A])
                        core_mask_A = cluster_data[label_A]['core_mask']
                        core_centers_A = centers_A[core_mask_A]
                        
                        #ghost_distances = cdist(core_centers_A, noise_centers)
                        tree_cores_A = cKDTree(core_centers_A)
    
                        # 2. Query: For every noise point, find the distance to its nearest core in A
                        # dists_to_nearest_core is a vector of length N_Noise
                        min_dists_to_cluster, _ = tree_cores_A.query(noise_centers, k=1)
                        
                        # Filter: Only trigger danger for noise points outside the limit
                        danger_mask = (min_dists_to_cluster > limit_A) 
                    
                        if np.any(danger_mask):
                            # Find the smallest danger gap
                            min_danger_gap = np.min(min_dists_to_cluster[danger_mask])
                            if min_danger_gap < danger_ceiling:
                                danger_ceiling = min_danger_gap
    
                        # ---------------------------------------------------------
                        # 2. THE TARGET SCANNER (Uses the closest Core PMC)
                        # ---------------------------------------------------------
                        
                        # single-edge clusters may set the danger ceiling but dont expand
                        if len(cluster_healthy_edges[label_A]) >= 2 and np.any(noise_core_mask):
                            valid_noise_mask = noise_core_mask
                            
                            # Filter: Keep all noise points that are inside the limit
                            target_mask = (min_dists_to_cluster <= limit_A) & valid_noise_mask
                            
                            if np.any(target_mask):
                                safe_gaps = min_dists_to_cluster[target_mask]
                                requested_targets.extend(safe_gaps)
                # ---------------------------------
                
                # ---------------------------------
                # --- TWO-PASS GREEDY AUDITOR ---
                # ---------------------------------
                
                # PASS 1: Evaluate all pairs, lock in the final danger_ceiling, and collect candidates
            if len(valid_labels) > 1:
                consenting_candidates = []
                
                for label_A, label_B in combinations(valid_labels, 2):
                    
                    """
                    # Find the distance between all pairs
                    border_distances = cdist(centers_A, centers_B)
                    min_idx = np.unravel_index(np.argmin(border_distances), border_distances.shape)
                    minimum_gap = border_distances[min_idx]
                    """
                    
                    core_mask_A = cluster_data[label_A]['core_mask']
                    core_mask_B = cluster_data[label_B]['core_mask']
                    core_centers_A = cluster_data[label_A]['centers'][core_mask_A]
                    core_centers_B = cluster_data[label_B]['centers'][core_mask_B]
                    tree_B = cKDTree(core_centers_B)
                    
                    # Tree B query: "For each point in A, what is the one closest point in B?"
                    dists_to_B, _ = tree_B.query(core_centers_A, k=1)
                    
                    minimum_gap = np.min(dists_to_B)
                    
                    limit_A = cluster_jump_limits[label_A]
                    limit_B = cluster_jump_limits[label_B]
                    
                    # --- PRE-MERGE SIMULATION (The Marriage Test) ---
                    # --- The Merge Conditions 
                    spatial_consent = (minimum_gap <= limit_A and minimum_gap <= limit_B)
                    
                    is_physically_touching = (minimum_gap <= 2*self.eps)
                    
                    future_survival = False
                    
                    if is_physically_touching:
                        future_survival = True 
                        
                    elif spatial_consent:
                        # --- PRE-MERGE SIMULATION (The Marriage Test) ---
                        edges_A = cluster_healthy_edges[label_A]
                        edges_B = cluster_healthy_edges[label_B]
                        combined_with_bridge = np.sort(np.concatenate([edges_A, edges_B, [minimum_gap]]))
                        if self.roller(combined_with_bridge, self.eps, g) == len(combined_with_bridge):
                            future_survival = True
                        
                    # --- Final Execution ---
                    if is_physically_touching or (spatial_consent and future_survival):
                        # TRACKING: They want to merge and survived the simulation
                        consenting_candidates.append((label_A, label_B, minimum_gap))
                        
                    else:
                        # TRACKING: Found a pair that refuses to merge. Lower the ceiling.
                        if minimum_gap < danger_ceiling:
                            danger_ceiling = minimum_gap
                            
    
                # PASS 2: The Greedy Search (Finding the largest gap that fits under the ceiling)
                best_gap = -1.0   
                best_labels = None
                
                for label_A, label_B, gap in consenting_candidates:
                    proposed_eps_candidate = gap * 1.01
                    
                    # If this gap is safe from the ceiling AND it is the largest found...
                    if proposed_eps_candidate < danger_ceiling*0.99 and gap > best_gap:
                        best_gap = gap
                        best_labels = (label_A, label_B)
    
                # --- EXECUTION PHASE ---
                if best_labels is not None:
                    label_A, label_B = best_labels
                    proposed_eps = best_gap * 1.01
                    
                    if proposed_eps > self.dbscan_eps:
                        self.dbscan_eps = proposed_eps
                        return "changed"
                        
            # ==========================================
            # 5. TARGETED EPS EXPANSION (Swept Clearance)
            # ==========================================
            if len(valid_labels) > 0 and len(requested_targets) > 0:
                
                # 1. The Clearance Filter: Proves the target is valid
                surviving_targets = [t for t in requested_targets if (t * 1.01) < (danger_ceiling * 0.99)]
                
                if len(surviving_targets) > 0:
                    best_target = max(surviving_targets)
                    final_eps = best_target * 1.01
                    
                    if final_eps > self.dbscan_eps:
                        self.dbscan_eps = final_eps
                        return "changed"
