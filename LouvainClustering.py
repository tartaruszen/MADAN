#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:22:50 2019

@author: Alexandre Bovet
"""

import numpy as np
from copy import deepcopy


class Clustering(object):
    """ Clustering of a temporal network between two times
        t1, t2 given by a list of clusters
    """

    def __init__(self, traj=None, t1=None, t2=None,
                       tempNet=None,
                       p1=None, p2=None,T=None,
                       cluster_list=None, node_to_cluster_dict=None):
        """ traj is an instance of Trajectories
            t1 and t2 are the start and end times of the
            clustering and cluster_list is a list
            of set of nodes defining the clustering
            
            Can be initialized either by providing
            
            `traj`, `t1` and `t2`
            - for random walk based transition probabilities
            
            `tempNet`, `t1` and `t2` (optionally p1, default=uniform dist.)
            - for exact transition probabilities based on a temporal network
            (only for discrete time temp. net.)
                
            `T`, `p1` and `p2`
            - for externally computed transition probabilities
            
            Clusters can either be initilized with
            - `cluster_list` : a list of set of nodes or
            - `node_to_cluster_dict` : a dictionary with mapping between nodes
            and cluster number.
            
            Default is each node in a different cluster.
            
        """

        if traj is not None:
        
            if t1 is None or t2 is None:
                raise ValueError('if `traj` is provided, t1 and t2 ' + \
                                 'must be provided')
            
            self.traj = traj
            self.t1 = traj._get_closest_time(t1)
            self.t2 = traj._get_closest_time(t2)
            self.num_nodes = traj.num_nodes
            
        elif tempNet is not None:
            if t1 is None or t2 is None:
                raise ValueError('if `tempNet` is provided, t1 and t2' + \
                                 'must be provided')
                
            self.tempNet = tempNet
            self.t1 = t1
            self.t2 = t2
            #compute exact transition probabilities from temporal network
            # only for discrete time temporal networks
            
            self.num_nodes = tempNet.num_nodes
            
            T_exact = tempNet.compute_exact_transition_matrix(start_time=t1,
                                                             end_time=t2)
            
            if p1 is None:
                self.p1 = np.ones(self.num_nodes)/self.num_nodes
                
            self.p2 = np.array(np.matmul(T_exact,self.p1)).flatten()
            
            # transpose to have prob of going form i to j instead of j to i
            self.T = np.array(T_exact.T)
            
            
        else:
            if p1 is None or p2 is None or T is None:
                raise ValueError('if `traj` or `tempNot` are not provided, p1, ' + \
                                 'p2 and T must be provided')
            
            self.p1 = p1
            self.p2 = p2
            self.T = T
            self.num_nodes = p1.size
        
        # initialize clusters
        if cluster_list is None and node_to_cluster_dict is None:
            # default clustering is one node per cluster
            self.cluster_list = [set([n]) for n in range(self.num_nodes)]
            self.node_to_cluster_dict = {n : n for n in range(self.num_nodes)}
            
        elif cluster_list is not None and node_to_cluster_dict is None:
            self.cluster_list = cluster_list
            self.node_to_cluster_dict = {}
            for i, clust in enumerate(self.cluster_list):
                for node in clust:
                    self.node_to_cluster_dict[node] = i
    
        elif cluster_list is None and node_to_cluster_dict is not None:
            self.node_to_cluster_dict = node_to_cluster_dict
            self.cluster_list = [set() for _ in \
                                 range(max(node_to_cluster_dict.values()) + 1)]            
            for node, clust in node_to_cluster_dict.items():
                self.cluster_list[clust].add(node)
                
        elif cluster_list is not None and node_to_cluster_dict is not None:
            raise ValueError('cluster_list and node_to_cluster_dict ' +\
                             'cannot be provided together')
            
        
    def compute_S(self, p1, p2, T):
        
        """ compute the internal matrix comparing probabilities for each
            node
                    S[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]
                    
        """
        
        S = np.zeros_like(T)
        
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                
                S[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]
                
        return S

    def compute_stability(self):
        """ returns the stability of the clusters given in `cluster_list` 
            computed between times `t1` and `t2`
        """
        
        num_clusters = len(self.cluster_list)
        
        if not hasattr(self,'p1'):
            self.p1 = self.traj.compute_prob_dist(self.t1)
        
        if not hasattr(self,'p2'):
            self.p2 = self.traj.compute_prob_dist(self.t2)
            
        if not hasattr(self,'T'):
            self.T = self.traj.compute_transition_matrix(self.t1, self.t2)

        if not hasattr(self,'S'):    
            self.S = self.compute_S(self.p1, self.p2, self.T)
        

        #indicator matrix
        H = np.zeros((self.num_nodes, num_clusters), dtype=int)
        for node in range(self.num_nodes):
            for c in range(num_clusters):
                H[node,c] = int(node in self.cluster_list[c])    
        
        #compute trace
        R_ii = np.zeros(num_clusters, dtype=np.float64)
        for c in range(num_clusters):
            temp_sum = np.zeros(self.num_nodes, dtype=np.float64)
            for node in range(self.num_nodes):
                temp_sum[node] = sum(self.S[node,:]*H[:,c])
            R_ii[c] = sum(H[:,c]*temp_sum)
            
        self.R = R_ii.sum()
        
        return self.R 
        

    def _compute_delta_stability(self, k, c_i, c_f):
        
        """ return the gain in stability obtained by moving node
            k from community c_i to community c_f
        """
        if k not in self.cluster_list[c_i]:
            raise ValueError('node k must be in cluster c_i')
            
        if c_i == c_f:
            return 0
                        
        if not hasattr(self,'p1'):
            self.p1 = self.traj.compute_prob_dist(self.t1)
        
        if not hasattr(self,'p2'):
            self.p2 = self.traj.compute_prob_dist(self.t2)
            
        if not hasattr(self,'T'):
            self.T = self.traj.compute_transition_matrix(self.t1, self.t2)

        
        # gain in stability from moving node k to community c_f
        delta_r1 = self.p1[k]*sum([self.T[k,j] - self.p2[j] for j in self.cluster_list[c_f]]) + \
                       sum([self.p1[i]*self.T[i,k] for i in self.cluster_list[c_f]]) - \
                       self.p2[k]*sum([self.p1[i] for i in self.cluster_list[c_f]]) + \
                       self.p1[k]*self.T[k,k] - self.p1[k]*self.p2[k]
                       
                       
        # gain in stability from moving node k out of community c_i 
        delta_r2 = - self.p1[k]*sum([self.T[k,j] - self.p2[j] for j in self.cluster_list[c_i]]) - \
                       sum([self.p1[i]*self.T[i,k] for i in self.cluster_list[c_i]]) + \
                       self.p2[k]*sum([self.p1[i] for i in self.cluster_list[c_i]]) + \
                       self.p1[k]*self.T[k,k] - self.p1[k]*self.p2[k]
                       
        return delta_r1 + delta_r2


    def _louvain_1st_phase(self, delta_r_threshold=np.finfo(float).eps,
                           verbose=False, rnd_state=None):
        """ return delta_r_tot, n_loop, self.cluster_list
        """
            
        if not hasattr(self,'T'):
            self.T = self.traj.compute_transition_matrix(self.t1, self.t2)
        
        if rnd_state is None:
            # random numbers with a random seed
            rnd_state = np.random.RandomState()
            
        
        delta_r_tot = 0
        delta_r_loop = 1
        n_loop = 1
        while delta_r_loop > delta_r_threshold:
            
            delta_r_loop = 0
            
            # shuffle order to process the nodes
            node_ids = np.arange(self.num_nodes)
            rnd_state.shuffle(node_ids)
                        
            for node in node_ids:
                # test gain of stability if we move node to neighbours communities
                
                # initial cluster of node
                c_i = self.node_to_cluster_dict[node]
                
                # tuple of neighbors, transition prob 
                neighs_prob = sorted([(neigh, 
                                        self.T[node,neigh]) for \
                                        neigh in np.where(self.T[node,:] > 0)[0]
                                        if neigh != node],
                                       key=lambda x:x[1], reverse=True)
                
                # keep only neighbors that are in a different cluster
                neighs = np.array([neigh for neigh,_ in neighs_prob if \
                                   self.node_to_cluster_dict[neigh] != c_i])
            
                if neighs.size == 0:
                    # all neighbours are in the same cluster
                    break
            
                neighs_delta_r = np.zeros(neighs.shape)
                for i, neigh in enumerate(neighs):
                    
                    c_n = self.node_to_cluster_dict[neigh]
                    
                    # delta of stability if moving node to c_n
                    neighs_delta_r[i] = self._compute_delta_stability(node,c_i,c_n)
                    
                best_neigh_id = np.argmax(neighs_delta_r)
                    
                if neighs_delta_r[best_neigh_id] > 0:
                    #move node to best_neigh's cluster
                    self.node_to_cluster_dict[node] = self.node_to_cluster_dict[neighs[best_neigh_id]]
                    self.cluster_list[c_i].remove(node)
                    self.cluster_list[self.node_to_cluster_dict[neighs[best_neigh_id]]].add(node)
                    
                    delta_r_loop += neighs_delta_r[best_neigh_id]
                                        
                    # else do nothing
            delta_r_tot += delta_r_loop

                
            if verbose:
                print('loop number ' + str(n_loop))
                print('delta r loop : ' + str(delta_r_loop))
                print('delta r total : ' + str(delta_r_tot))
                print('number of clusters : ' + \
                      str(len([c for c in self.cluster_list if len(c) > 0 ])))
                    
                print('clusters : ' + str([c for c in self.cluster_list if len(c) > 0 ]))
            
            n_loop += 1
            
        return delta_r_tot, n_loop, self.cluster_list
            
    def find_optimal_clustering(self, delta_r_threshold=np.finfo(float).eps,
                                verbose=False, rnd_seed=42):
        """return delta_r_tot, n_meta_loop, cluster_list
        """
        # implement loop control based on delta r difference
        
        # random numbers from seed rnd_seed
        rnd_state = np.random.RandomState(rnd_seed)
        
        
        if not hasattr(self,'p1'):
            self.p1 = self.traj.compute_prob_dist(self.t1)
        
        if not hasattr(self,'p2'):
            self.p2 = self.traj.compute_prob_dist(self.t2)
            
        self.cluster_list_init = deepcopy(self.cluster_list)
        self.node_to_cluster_dict_init = deepcopy(self.node_to_cluster_dict)
        
        # 1st pass
        delta_r_tot, _, _ = self._louvain_1st_phase(delta_r_threshold, verbose,
                                                    rnd_state)
        
        delta_r_meta_loop = 1
        n_meta_loop = 0
        while delta_r_meta_loop > delta_r_threshold:
        
            # cluster aggregation
        
            # clusters of original nodes forming meta nodes
            cluster_meta_nodes = [c for c in self.cluster_list if len(c) > 0 ]
            num_meta_nodes = len(cluster_meta_nodes)

            
            p1 = np.zeros(num_meta_nodes)    
            p2 = np.zeros(num_meta_nodes)
            T = np.zeros((num_meta_nodes,num_meta_nodes))
            
            for i, c_i in enumerate(cluster_meta_nodes):
                p1[i] = sum([self.p1[k] for k in c_i])
                p2[i] = sum([self.p2[k] for k in c_i])
                
                for j, c_j in enumerate(cluster_meta_nodes):
                    for k in c_i:
                        for l in c_j:
                            T[i,j] += self.T[k,l]
                        
                    T[i,j] /= p1[i]
                
                T[i,:] /= T[i,:].sum()
                
            meta_clustering = Clustering(T=T, p1=p1, p2=p2)
            
            delta_r_meta_loop, _, _ = meta_clustering._louvain_1st_phase(delta_r_threshold, verbose, rnd_state)
            
            #reorganize original nodes
            for meta_node, original_nodes in enumerate(cluster_meta_nodes):
                
                for node in original_nodes:
                    
                    self.node_to_cluster_dict[node] = \
                        meta_clustering.node_to_cluster_dict[meta_node]
            
            self.cluster_list = [set() for _ in \
                                 range(max(self.node_to_cluster_dict.values()) + 1)]            
            for node, clust in self.node_to_cluster_dict.items():
                self.cluster_list[clust].add(node)
                
            
            delta_r_tot += delta_r_meta_loop

                
            if verbose:
                print('meta loop number ' + str(n_meta_loop))
                print('delta r meta loop : ' + str(delta_r_meta_loop))
                print('delta r total : ' + str(delta_r_tot))
                print('number of clusters : ' + \
                      str(len([c for c in self.cluster_list if len(c) > 0 ])))
                    
                print('clusters : ' + str([c for c in self.cluster_list if len(c) > 0 ]))
            
            n_meta_loop += 1
                
        return delta_r_tot, n_meta_loop, self.cluster_list
    
    
if __name__ == '__main__':
    import networkx as nx

    G = nx.random_partition_graph((10,20,30), p_in=0.9, p_out=0.1)
    
    A = nx.adjacency_matrix(G).todense()
    degree_vect = np.array(list(G.degree().values()), dtype=np.float64)
    
    # transition matrix
    T = np.matmul(np.diag(degree_vect**(-1)),A)
    
    # stationary distribution
    pi = degree_vect/(2*G.number_of_edges())
    
    clustering = Clustering(p1=pi, p2=pi, T=T)
    
    clustering.find_optimal_clustering()
    
    for i, c in enumerate(clustering.cluster_list):
        print('Cluster ', i, ' has ', len(c), ' nodes')
    
    
    