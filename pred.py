import mix_embedding as me
import pickle
import data_prep_not_saving as prep
import numpy as np


import keras_classifier as kc

from itertools import permutations  

# method-agnostic
corpus_data_path = './data/Corpus/corpus_all_in_index.pkl'
ontology_EdgeList_path = './data/Ontology/MeSH_EdgeList.txt'

'''
link_rel = 'treat' #['treat', 'interact', 'cause', 'affect']
train_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format(link_rel, link_rel)
train_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_false.txt'.format(link_rel, link_rel)
val_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_true.txt'.format(link_rel, link_rel)
val_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_false.txt'.format(link_rel, link_rel)
test_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_true.txt'.format(link_rel, link_rel)
test_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_false.txt'.format(link_rel, link_rel)
'''

record_file = 'record_all_rels.txt'

# method-specific
# embedding_save_folder = './embeddings/generated/'
# embedding_save_path = embedding_save_folder+setting_name	

def evaluate_fix(graph_path_list, sentence_path_list, num_walks_list, walk_length_list, p, q, window, iter, directed, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path):

	# learn embeddings
	embeddings = me.mix_embeddings(graph_path_list, sentence_path_list, num_walks_list, walk_length_list, p=p, q=q, window=window, iter=iter, directed=directed)

	# evaluate
	train_data, train_labels, val_data, val_labels, test_data, test_labels = prep.embeddings_to_data(embeddings, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path, merge_op='avg')
	
	scores_all = []
	for i in range(10):
		scores = kc.evaluate(train_data, train_labels, val_data, val_labels, test_data, test_labels)
		scores_all.append(scores)

	return np.mean(scores_all, axis=0), np.std(scores_all, axis=0)


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

results_dict = {}

for link_rel in ['treat', 'interact', 'cause', 'affect']: # 'treat']: #, 'interact', 'cause', 'affect']:

	print('\n\n')
	print(link_rel)
	print('\n\n')

	train_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format(link_rel, link_rel)
	train_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_false.txt'.format(link_rel, link_rel)
	val_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_true.txt'.format(link_rel, link_rel)
	val_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_false.txt'.format(link_rel, link_rel)
	test_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_true.txt'.format(link_rel, link_rel)
	test_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_false.txt'.format(link_rel, link_rel)

	sentence_path_list = [corpus_data_path]
	graph_path_list = [ontology_EdgeList_path] #, './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format('treat', 'treat'),'./data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format('interact', 'interact'),'./data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format('cause', 'cause'),'./data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format('affect', 'affect')]

	num_walks_list = [80] #,80,80,80,80]
	walk_length_list = [10] #,10,10,10,10]
	p=0.25
	q=4
	window=5
	iter=1
	directed=1

	setting_name = '{}'.format(link_rel)

	avg_scores, std_scores = evaluate_fix(graph_path_list, sentence_path_list, num_walks_list, walk_length_list, p=p, q=q, window=window, iter=iter, directed=directed, train_true_EdgeList_path=train_true_EdgeList_path, train_false_EdgeList_path=train_false_EdgeList_path, val_true_EdgeList_path=val_true_EdgeList_path, val_false_EdgeList_path=val_false_EdgeList_path, test_true_EdgeList_path=test_true_EdgeList_path, test_false_EdgeList_path=test_false_EdgeList_path)

	results_dict[setting_name] = [avg_scores, std_scores]

	write_to_record(record_file, setting_name, avg_scores, std_scores)