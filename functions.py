import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
import networkx as nx
from Plotters import *
from Nets import *
import matplotlib as mpl
import pandas as pd
import pygsp

"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""


def local_smoothness(net, A, f,i):
	neighbors = list(net.neighbors(i))
	res = []
	for n in neighbors:
		res.append(A[i,n]*np.square(f[n] - f[i]))

	return np.sqrt(np.sum(res))	



def get_weigth_matrix_vectors(A,attrib_mat,sigma=1):
	
	N    =  A.shape[0]
	W    =  np.zeros((N,N))


	for i in range(0,N):
		for j in range(i,N):
			if A[i,j] == 1:	
							
				W[i,j] = np.linalg.norm(attrib_mat[i] - attrib_mat[j])
				W[j,i] = W[i,j]
		
	W_res  = np.exp(-0.5*np.square(W)/(sigma**2))
	
	return np.multiply(W_res,A)	


def interpolate_comm(G,part):

	new_part = np.zeros(G.order())
	for node in G.nodes():
		neigh_n =  list(G.neighbors(node))
		res     =  dict(Counter(part[neigh_n]))
		max_key =  max(res, key=lambda k: res[k])

		new_part[node] = max_key

	keys_comm = list(set(new_part))
	vals_comm = range(0,len(keys_comm))
	mapping = dict(zip(keys_comm, vals_comm))
	interp_comm_ok = np.array([mapping[val] for val in new_part])
	
	return interp_comm_ok

# Only interpolate anomalies
def interpolate_comm2(G,part, true_nodes):

	new_part = part
	for node in true_nodes:
		neigh_n =  list(G.neighbors(node))
		
		flag = len(set(part[neigh_n])) == len(part[neigh_n])
		if not(flag): 
			res            =  dict(Counter(part[neigh_n]))
			max_key        =  max(res, key=lambda k: res[k])
			new_part[node] =  max_key

	keys_comm = list(set(new_part))
	vals_comm = range(0,len(keys_comm))
	mapping = dict(zip(keys_comm, vals_comm))
	interp_comm_ok = np.array([mapping[val] for val in new_part])
	
	return interp_comm_ok	
	


# Given a node id, return the attributes of all nodes of its community
def get_node_community_attribs(node, net, partition, attribs):

	node_class = partition[node]
	node_comm = []
	for n, group in enumerate(partition):
		if partition[n] == node_class:
			node_comm.append(n)

	mat = []
	for n in node_comm:

		row = [n]+[net.node[n][val] for val in attribs]
		mat.append(row)


	mat = np.array(mat)	
	titles = attribs
	res = pd.DataFrame(mat[:,1:], index=mat[:,0].astype('int'), columns=titles)	
	return res



def evaluate_auc(y_true, y_pred):	
	
	precision, recall, _  =  precision_recall_curve(y_true,y_pred)
	auc_score             =  auc(recall, precision)

	return auc_score

# Nodes must be between 0..N-1	
def nodes_to_list(nodes, N):
	
	y_bin  =  np.zeros(N)
	y_bin[nodes] = 1

	return y_bin	

def tic():
	#Homemade version of matlab tic and toc functions
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()

def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
	else:
		print("Toc: start time not set"	)
