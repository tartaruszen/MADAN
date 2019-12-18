"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""


from sklearn.metrics import roc_curve, f1_score, precision_recall_curve, auc, roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
import networkx as nx
from Plotters import *
from Nets import *
import matplotlib as mpl
import pandas as pd
import pygsp
from functions import *
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from exp_chebyshev import *

mpl.interactive(True)
plt.rcParams['image.cmap'] = 'jet'
pl = Plotters()

# Performing anomalu detection on real life networks (disney, enron, books)
#--------------------------------------------------------------------------------------------------------------------------------------------------

#disney_long  =  ['Max_Votes', 'MinPriceUsedItem', 'Sales_Rank', 'Rating_4_Ratio', 'MinPricePrivateSeller', 'Max_Categories_Depth_of_this_Product', 'Rating_3_Ratio', 'Number_of_reviews', 'Avg_Rating', 'Rating_1_Ratio', 'Helpful_votes_ratio', 'Review_frequency', 'Min_Helpful', 'No_of_Categories', 'Max_Helpful', 'Avg_Helpful', 'Rating_of_review_with_least_votes', 'Number_of_different_authors', 'Min_Categories_Depth_of_this_Product', 'Rating_of_least_helpful_rating', 'Rating_of_review_with_most_votes', 'Rating_2_Ratio', 'Avg_Votes']
disney_attrib =  ['MinPricePrivateSeller','Avg_Rating']
books_attrib  = ['Min_Helpful','MinPriceUsedItem', 'Min_Votes', 'Rating_3_Ratio', 'Rating_span', 'Sales_Rank', 'Review_frequency', 'Helpful_votes_ratio', 'Avg_Votes', 'Rating_5_Ratio',  'Max_Votes',  'Max_Helpful',  'Rating_2_Ratio', 'Amazon_price', 'MinPricePrivateSeller','Avg_Rating', 'Rating_4_Ratio', 'Rating_1_Ratio', 'Number_of_reviews', 'Avg_Helpful']
enron_attrib = ['AverageContentReplyCount',  'AverageContentLength',  'EnronMailsBcc',  'AverageContentForwardingCount', 'AverageRangeBetween2Mails', 'OtherMailsBcc',  'AverageNumberBcc', 'AverageDifferentSymbolsContent',  'AverageDifferentSymbolsSubject',  'OtherMailsTo',  'DifferentCharsetsCount',  'DifferntCosCount',  'OtherMailsCc',  'AverageNumberTo',  'AverageNumberCc', 'EnronMailsTo', 'DifferentEncodingsCount', 'MimeVersionsCount']  
enron_short   = ['AverageContentForwardingCount','OtherMailsBcc','AverageDifferentSymbolsSubject','OtherMailsCc','EnronMailsTo']                    

#--------------------------------------------------------------------------------------------------------------------------------------------------
# disney 1
# books 3
# enron 4

num_net     =    1
name        =   'disney'
attributes  =    disney_attrib

#--------------------------------------------------------------------------------------------------------------------------------------------------


data        =   Nets(num_net,attributes[0]) 

net         =   data.net
N           =   net.order()
partition   =   nx.get_node_attributes(net,"ModularityClass")

y_true      =   [net.node[n]['anomaly'] for n in net.nodes()] 
true_nodes  =   [i for i in range(0,len(y_true)) if y_true[i]==1]

#-----------------------------------------------------------------------------------------------------
taus        =   10**np.linspace(0,4,200)
taus        =   np.concatenate([np.array([0]),taus])
sigma_real  =   {'disney': 0.32, 'books': 0.15, 'enron': 0.1}

#-----------------------------------------------------------------------------------------------------
# Defining matrix of attributes

f_matrix = np.zeros((N,len(attributes)))
for i, att in enumerate(attributes):
    attribs =  nx.get_node_attributes(net,att)
    f_matrix[:,i] =  list(attribs.values())


f_scaled  =  preprocessing.MinMaxScaler().fit_transform(f_matrix)
rr        =  metrics.pairwise.euclidean_distances(f_scaled)
A         =  nx.adjacency_matrix(net).toarray()
#-----------------------------------------------------------------------------------------------------

sigma_list  =  [sigma_real[name]]

#-----------------------------------------------------------------------------------------------------

print(num_net)
print(name)
print(attributes)
print(sigma_list)
print('\n')

res  = dict()

tic();
count_itr = 0
for count_sig, sigma in enumerate(sigma_list):


    print("Computing weight matrix...")
    W = get_weigth_matrix_vectors(A,f_scaled,sigma); 
    G = graphs.Graph(W)

    print("Computing frame g_L and concentration....")
    concentration_times = np.zeros((N,len(taus)))
 
    v_ones      =  np.matrix(np.ones((N,1)))
    degree_vect =  W.sum(axis=1)                        # strengths vector
    D           =  np.diag(degree_vect)    
    avg_d       =  (v_ones.T*D*v_ones)[0,0]/N                # average strength

    #-----------------------------------------------------------------------
    auc_values      = []
    roc_auc         = []
    avg_precision   = []
    
    print("Looking at the best scale...\n")
    for inx, t in enumerate(taus):

        #------------------------------------------------------------------------------        
        concentration_times[:,inx]  = compute_fast_exponential(G,t/avg_d) # Chebychev
        #concentration_times[:,inx]   = chebychev_sequential(G,t/avg_d)
        #------------------------------------------------------------------------------

        y_scores  = concentration_times[:,inx]

        #--------- Evaluate roc_auc ----------------------------------------------------------------------- 
        roc_auc.append(roc_auc_score(y_true, y_scores, average='macro'))    
        #---------------------------------------------------------------------------------------
        if count_itr%10 == 0:
            print('Itr: %f / %f'%(count_itr, len(sigma_list)*len(taus)))
        count_itr+=1    

    #---------------------------------------------------------------------            
    # ROC-auc
    inx = np.argmax(roc_auc)
    res[(sigma,inx)] = roc_auc[inx]
    print('\n')
    print('Best scale, time= %f' % (taus[inx]/(avg_d)))
    print('ROC/AUC: %f'% (roc_auc[inx]))
    print('\n')


toc();

#--------------------------------------------------------------------------------------------------
