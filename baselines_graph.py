import mix_embedding as me
import pickle
import data_prep_not_saving as prep
import numpy as np
import networkx as nx

import keras_classifier as kc

# from numba import cuda
import os

from itertools import permutations  
from sklearn import metrics


# method-agnostic
results_folder = './results/'
'''
train_true_EdgeList_path = './prediction/SemMedDB_treat_MeSH/treat_EdgeList_train_true.txt'
train_false_EdgeList_path = './prediction/SemMedDB_treat_MeSH/treat_EdgeList_train_false.txt'
val_true_EdgeList_path = './prediction/SemMedDB_treat_MeSH/treat_EdgeList_valid_true.txt'
val_false_EdgeList_path = './prediction/SemMedDB_treat_MeSH/treat_EdgeList_valid_false.txt'
test_true_EdgeList_path = './prediction/SemMedDB_treat_MeSH/treat_EdgeList_test_true.txt'
test_false_EdgeList_path = './prediction/SemMedDB_treat_MeSH/treat_EdgeList_test_false.txt'
'''

record_file = 'record_baselines_graph.txt'

# method-specific
# embedding_save_folder = './embeddings/generated/'
# embedding_save_path = embedding_save_folder+setting_name

def read_EdgeList(EdgeList_path):

	edges = []
	with open(EdgeList_path,'r') as f:
		for line in f:
			if line != '':
				i1, i2 = line.split()
				edges.append((i1, i2))
	return edges

def evaluate_baseline(method, train_true_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path):

	# load graph
	G = nx.read_edgelist(train_true_EdgeList_path, nodetype=None, data=[('weight',int)])#read graph

	# evaluate
	true_edges = read_EdgeList(test_true_EdgeList_path)
	true_labels = [1 for i in range(len(true_edges))]
	false_edges = read_EdgeList(test_false_EdgeList_path)
	false_labels = [0 for i in range(len(false_edges))]

	edges = true_edges + false_edges
	test_labels = true_labels + false_labels

	# split edges to valid and invalid
	edges_valid = []
	edges_invalid = []
	test_labels_valid = []
	test_labels_invalid = []
	for i in range(len(edges)):
		if (edges[i][0] in G) and (edges[i][1] in G):
			edges_valid.append(edges[i])
			test_labels_valid.append(test_labels[i])
		else:
			edges_invalid.append(edges[i])
			test_labels_invalid.append(test_labels[i])

	if method == 'Jaccard':
		preds = nx.jaccard_coefficient(G, edges_valid)
	if method == 'preferential_attachment':
		preds = nx.preferential_attachment(G, edges_valid)
	if method == 'Adamic-Adar':
		preds = nx.adamic_adar_index(G, edges_valid)

	pred_valid = [1 if p[2]>=0.5 else 0 for p in preds]
	pred_ts = pred_valid + [0 for i in range(len(edges_invalid))]

	test_labels = test_labels_valid + test_labels_invalid


	acc_score = metrics.accuracy_score(test_labels, pred_ts)

	recall_score = metrics.recall_score(test_labels, pred_ts, average='macro')

	f1_score = metrics.f1_score(test_labels, pred_ts, average='macro')

	map_score = metrics.average_precision_score(test_labels, pred_ts)

	roc_score = metrics.roc_auc_score(test_labels, pred_ts)

	fpr, tpr, thresholds = metrics.precision_recall_curve(test_labels, pred_ts)
	aupr_score = metrics.auc(tpr, fpr)

	scores = [acc_score, recall_score, f1_score, map_score, roc_score, aupr_score]

	return scores


def write_to_record(record_file, setting_name, avg_scores, std_scores):

	print(setting_name)
	print('Finished!\n')
	with open(record_file, 'a') as file:
		file.write('\n\n')
		file.write(setting_name)
		file.write('\n')
		file.write('acc    score: {} ({}) \n'.format(avg_scores[0],std_scores[0]))
		file.write('recall score: {} ({}) \n'.format(avg_scores[1],std_scores[1]))
		file.write('f1     score: {} ({}) \n'.format(avg_scores[2],std_scores[2]))
		file.write('map    score: {} ({}) \n'.format(avg_scores[3],std_scores[3]))
		file.write('roc    score: {} ({}) \n'.format(avg_scores[4],std_scores[4]))
		file.write('aupr   score: {} ({}) \n'.format(avg_scores[5],std_scores[5]))

### group 2
results_dict = {}
for link_rel in ['treat','interact', 'cause', 'affect']:

	print('\n\n')
	print(link_rel)
	print('\n\n')

	train_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format(link_rel, link_rel)
	test_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_true.txt'.format(link_rel, link_rel)
	test_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_false.txt'.format(link_rel, link_rel)											
	
	for method in ['Jaccard', 'preferential_attachment', 'Adamic-Adar']:
		setting_name = 'Sem_{}_{}'.format(link_rel, method)
		scores = evaluate_baseline(method, train_true_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path)
		# write_to_record(record_file, setting_name, avg_scores, std_scores)
		results_dict[setting_name] = scores

pickle.dump(results_dict,open('results_graph_baselines.pkl','wb'))