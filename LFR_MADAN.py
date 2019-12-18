import numpy as np
import networkx as nx
import matplotlib as mpl
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
from LFR_nets import *
from functions import *
from Plotters import *
import argparse

"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""


mpl.interactive(True)
plt.close('all')
plt.rcParams['image.cmap'] = 'jet'
pl = Plotters()


def compute_heat(G,t=0):
	  
	kernel     =  np.exp(-((t*G.e)/(avg_d*G.lmax))) # CL    
	exp_sigma  =  np.diag(kernel)
	Ht         =  np.dot(np.dot(G.U,exp_sigma),G.U.T)

	concent    =  np.linalg.norm(Ht, axis=0, ord=2)
   
	return concent


#------------------------------------------------------------------------------------------------

list_percentage = [1,5,10,15,20,25,30]

num_attribs              =  20  
num_anomalous_attribs    =  6

#-------------------------------------------------------------------------------------------------
LFR         =   LFR_nets()
time        =   10**np.linspace(0,5,200)
time        =   np.concatenate([np.array([0]),time])
t           =   time[90]
sigma       =   0.1
#-------------------------------------------------------------------------------------------------

df_results = pd.DataFrame(columns=['avg_auc','avg_avg_prec','avg_roc_auc','mu','net_id','perturbation','std_auc','std_avg_prec','std_roc_auc'])

for perc_anomalous_nodes in list_percentage:

	print("Attributes dimension: %d"%num_attribs); 
	print("Anomalous attributes: %d"%num_anomalous_attribs);
	print("Percentage of anomalous nodes: %d"%perc_anomalous_nodes);
	print("Generating artificial networks with node attributes and anomalies...")
	print("\n")


	# Results for each percentage of anomalous nodes
	pr_auc_values      = []
	roc_auc_values     = []
	avg_precision_values = []

	
	for net_id in range(0,50): #LFR.num_nets
	
		mu = '0.1'		
		LFR.select_net(net_id)
		attrib_partition = LFR.get_attrib_clusters()
		groups_partition = LFR.get_true_communities()

		LFR.creating_node_attributes(attrib_partition,num_attribs) # dimension of anomalies
		LFR.injecting_anomalies(perc_anomalous_nodes, num_anomalous_attribs)


		#-------------------------------------------------------------------------------------------
		# Staring MADAN Algorithm
		#-------------------------------------------------------------------------------------------

		net = LFR.net
		N   = net.order()

		# Defining matrix of attributes
		f_matrix  =  LFR.get_node_attributes()
		f_scaled  =  preprocessing.MinMaxScaler().fit_transform(f_matrix)
		
		#------------------------------------------------------------------------------------------
		# Defining edge weights and computing Fourier basis
		#------------------------------------------------------------------------------------------
		A     =  nx.adjacency_matrix(net).toarray()
		W     =  get_weigth_matrix_vectors(A,f_scaled,sigma)
		G     =  graphs.Graph(W)
		G.compute_fourier_basis() 

		#-----------------------------------------------------------------------------------------------
		# 
		v_ones      =  np.matrix(np.ones((N,1)))
		degree_vect =  W.sum(axis=1)                        # strengths vector
		D           =  np.matrix(np.diag(degree_vect))    
		avg_d       =  (v_ones.T*D*v_ones)[0,0]/N           # average strength
		pi          =  v_ones.A.T.reshape(N)/N

		#-------------------------------------------------------------------------------------------------
		# Filtering signal with heat kernel 
		#-------------------------------------------------------------------------------------------------
		y_scores    = compute_heat(G,t)

		#-------------------------------------------------------------------------------------------------
		#  Evaluating Precision-Recall AUC 
		#-------------------------------------------------------------------------------------------------
		precision, recall, thresholds = precision_recall_curve(LFR.y_true, y_scores, pos_label=1)
		
		pr_auc = auc(recall,precision)
		pr_auc_values.append(pr_auc)

		#-------------------------------------------------------------------------------------------------
		# Evaluating ROC - AUC
		#--------------------------------------------------------------------------------------------------	
		roc_auc_values.append(roc_auc_score(LFR.y_true, y_scores, average='macro'))	
			
		#-------------------------------------------------------------------------------------------------
		# Evaluating average precision (AP)
		#-------------------------------------------------------------------------------------------------
		avg_precision_values.append(average_precision_score(LFR.y_true, y_scores, average='macro'))	


				
	#---------------------------------------------------------------------------------------

	avg_pr_auc    =  round(np.mean(pr_auc_values),3)
	std_pr_auc    =  round(np.std(pr_auc_values),3)

	avg_roc_auc   =  round(np.mean(roc_auc_values),3)
	std_roc_auc   =  round(np.std(roc_auc_values),3)

	avg_avg_prec   = round(np.mean(avg_precision_values),3)
	std_avg_prec   = round(np.std(avg_precision_values),3)

	df_results = df_results.append({'perturbation': perc_anomalous_nodes, 
									'avg_auc': avg_pr_auc,   'std_auc': std_pr_auc, 
									'avg_roc_auc': avg_roc_auc, 'std_roc_auc': std_roc_auc,
									'avg_avg_prec': avg_avg_prec, 'std_avg_prec': std_avg_prec,
									'mu': mu, 'net_id': 0}, ignore_index=True)


print(df_results)

df_results.to_csv('plot_LFR/MADAN_LFR_pert_20.csv')

print("Now, plot the results: cd plot_LFR/")
print("python plot_scores.py")

