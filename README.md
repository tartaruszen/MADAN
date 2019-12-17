## MADAN
MADAN is the acronym of Multi-scale Anomaly Detection on Attributed Networks.
This is an unsupervised algorithm allowing to detect anomalous nodes and their context at all scales of the network.

Here you can find the code of the algorith with some examples implemented on our paper:
Multi-scale Anomaly Detection For Attributed Networks (MADAN algorithm), published at AAAI-20 conference.
[Preprint](https://arxiv.org/abs/1912.04144)


Tested on Jupyter notebook 5.7.0 with Python 3.7.3
Dependences: pygsp, pandas, sklearn, networkx, matplotlib

**Note:** For efficiency reasons some functions were written in cython. We recommend you to compile them before, running the following script:
```
python setup.py build_ext --inplace 
```

--------------------------------------------------------------------------------------------------------------------

#### Jupyter notebooks ######

* Running MADAN algorithn on a toy example. (Figure 2 of the paper).

```
toy_example.ipynb
```

* MADAN algorithn on a real life dataset, the Disney copurchasing network (Figure 5 of the paper)

```
Case_of_study_Disney.ipynb
```

#### Benchmarking on synthetic networks ######

* MADAN algorith on synthetic networks
We generate artificial attributed networks with ground truth anomalies and evaluate the performance of recovering anomalous nodes varing the percentage of anmalies in the network (Figure 3 of the paper).

The following script will compute the ROC/AUC and PR/AUC metrics for the synthetic networks:

```
python LFR_MADAN.py
```

* Then plotting the scores as the Figure 3 of the paper:

```
cd plot_LFR/
python plot_scores.py
```

#### Real life examples ######

* Running anomaly detection on the Disney copurchasing network data and computes ROC-AUC score, (Table 2 of the paper).
```
python run_real_data.py
```
    
* Normalized variaiton of information  
This script computes the variation of information V(t,t') (background of Figure 2 and 4), between optimal partitions at times t and t'.
It allows to uncover the intrinsic scales having into account the graph structure and the node attributes.

```
variation_of_information/compare_partitions_over_time.m
```

#### Others ######

This folder contains some *Matlab* figures of the scanning of relevant partitions in the toy example and the Disney network.
```
figures/
```
