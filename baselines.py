import mix_embedding as me
import pickle
import data_prep_not_saving as prep
import numpy as np

import keras_classifier as kc

# from numba import cuda
import os

from itertools import permutations  

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

record_file = 'record_baselines.txt'

# method-specific
# embedding_save_folder = './embeddings/generated/'
# embedding_save_path = embedding_save_folder+setting_name

def evaluate_baseline(embedding_path, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path):

	# load embeddings
	embeddings = pickle.load(open(embedding_path, 'rb'))

	# evaluate
	train_data, train_labels, val_data, val_labels, test_data, test_labels = prep.embeddings_to_data(embeddings, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path)
	
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


### group 0
corpus_embedding_path = 'data/Corpus/w2v.pkl'
for link_rel in ['treat']:  # ['treat', 'interact', 'cause', 'affect']:

	print('\n\n')
	print(link_rel)
	print('\n\n')

	train_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format(link_rel, link_rel)
	train_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_false.txt'.format(link_rel, link_rel)
	val_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_true.txt'.format(link_rel, link_rel)
	val_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_false.txt'.format(link_rel, link_rel)
	test_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_true.txt'.format(link_rel, link_rel)
	test_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_false.txt'.format(link_rel, link_rel)

	setting_name = 'Corpus_word2vec_{}'.format(link_rel)
	avg_scores, std_scores = evaluate_baseline(corpus_embedding_path, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path)
	write_to_record(record_file, setting_name, avg_scores, std_scores)



### group 1
folder_ontolotgy = 'data/Ontology/trained/'
for file in os.listdir(folder_ontolotgy):
	if file[-4:] == '.pkl':

		if file[0:-4] != 'LINE':
			continue

		for link_rel in ['interact', 'cause', 'affect']:

			print('\n\n')
			print(link_rel)
			print('\n\n')

			train_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format(link_rel, link_rel)
			train_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_false.txt'.format(link_rel, link_rel)
			val_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_true.txt'.format(link_rel, link_rel)
			val_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_false.txt'.format(link_rel, link_rel)
			test_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_true.txt'.format(link_rel, link_rel)
			test_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_false.txt'.format(link_rel, link_rel)											
		
			setting_name = 'Ontology_{}_{}'.format(file[0:-4], link_rel)
			avg_scores, std_scores = evaluate_baseline(folder_ontolotgy+file, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path)
			write_to_record(record_file, setting_name, avg_scores, std_scores)

for file in os.listdir(folder_ontolotgy):
	if file[-4:] == '.pkl':

		for link_rel in ['treat']:

			print('\n\n')
			print(link_rel)
			print('\n\n')

			train_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format(link_rel, link_rel)
			train_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_false.txt'.format(link_rel, link_rel)
			val_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_true.txt'.format(link_rel, link_rel)
			val_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_false.txt'.format(link_rel, link_rel)
			test_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_true.txt'.format(link_rel, link_rel)
			test_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_false.txt'.format(link_rel, link_rel)											
		
			setting_name = 'Ontology_{}_{}'.format(file[0:-4], link_rel)
			avg_scores, std_scores = evaluate_baseline(folder_ontolotgy+file, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path)
			write_to_record(record_file, setting_name, avg_scores, std_scores)

### group 2
results_dict = {}
for link_rel in ['treat']:
	sem_foler = './data/SemMedDB/SemMedDB_{}_MeSH/{}_trained/'.format(link_rel, link_rel)
	for file in os.listdir(sem_foler):
		if file[-4:] == '.pkl':

			# if file[0:-4] != 'LINE':
			# 	continue

			print('\n\n')
			print(link_rel)
			print('\n\n')

			train_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_true.txt'.format(link_rel, link_rel)
			train_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_train_false.txt'.format(link_rel, link_rel)
			val_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_true.txt'.format(link_rel, link_rel)
			val_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_valid_false.txt'.format(link_rel, link_rel)
			test_true_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_true.txt'.format(link_rel, link_rel)
			test_false_EdgeList_path = './data/SemMedDB/SemMedDB_{}_MeSH/{}_EdgeList_test_false.txt'.format(link_rel, link_rel)											
		
			setting_name = 'Sem_{}_{}'.format(link_rel, file[0:-4])
			avg_scores, std_scores = evaluate_baseline(sem_foler+file, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path)
			# write_to_record(record_file, setting_name, avg_scores, std_scores)
			results_dict[setting_name] = [avg_scores, std_scores]

pickle.dump(results_dict,open('Sem_results_treat.pkl','wb'))