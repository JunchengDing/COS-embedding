import pickle
import numpy as np
import os

def merge_embeddings(e1, e2, merge_op):

	# input:
	# merge_op: operations to merge the embeddings, can be 'avg', 'hada', 'l1', or 'l2'.

	if merge_op not in ['avg', 'hada', 'l1', 'l2']:
		raise NameError('No such operations!')

	if merge_op == 'avg':
		return (e1+e2)/2
	elif merge_op == 'hada':
		return e1*e2
	elif merge_op == 'l1':
		return np.array([abs(a-b) for a,b in zip(e1,e2)])
	elif merge_op == 'l2':
		return np.array([np.linalg.norm(a-b) for a,b in zip(e1,e2)])

def node2edges_embedding(embeddings, EdgeList_path, merge_op):

	# input:
	# embedding path: path of the embeddings as dictionary {MeSH term index: embedding vector} stored in pickle file.
	# EdgeList path: path of edge list.
	# merge_op: operations to merge the embeddings, can be 'avg', 'hada', 'l1', or 'l2'.

	# embeddings = pickle.load(open(embedding_path, 'rb')) # dict {index: embedding vector}
	output_vectors = []

	with open(EdgeList_path,'r') as f:
		for line in f:
			if line != '':
				i1, i2 = line.split()
				if i1 in embeddings and i2 in embeddings:
					output_vectors.append(merge_embeddings(embeddings[i1], embeddings[i2], merge_op))

	return np.array(output_vectors)

def embeddings_to_data(embeddings, train_true_EdgeList_path, train_false_EdgeList_path, val_true_EdgeList_path, val_false_EdgeList_path, test_true_EdgeList_path, test_false_EdgeList_path, merge_op='hada'):

	train_true_data = node2edges_embedding(embeddings, train_true_EdgeList_path, merge_op)
	labels_tr_1 = np.ones(train_true_data.shape[0], dtype=int)

	train_false_data = node2edges_embedding(embeddings, train_false_EdgeList_path, merge_op)
	labels_tr_2 = np.zeros(train_false_data.shape[0], dtype=int)

	train_data = np.concatenate((train_true_data, train_false_data), axis=0)
	train_labels = np.concatenate((labels_tr_1, labels_tr_2), axis=0)

	val_true_data = node2edges_embedding(embeddings, val_true_EdgeList_path, merge_op)
	val_labels_1 = np.ones(val_true_data.shape[0], dtype=int)

	val_false_data = node2edges_embedding(embeddings, val_false_EdgeList_path, merge_op)
	val_labels_2 = np.zeros(val_false_data.shape[0], dtype=int)

	val_data = np.concatenate((val_true_data, val_false_data), axis=0)
	val_labels = np.concatenate((val_labels_1, val_labels_2), axis=0)

	test_true_data = node2edges_embedding(embeddings, test_true_EdgeList_path, merge_op)
	test_labels_1 = np.ones(test_true_data.shape[0], dtype=int)

	test_false_data = node2edges_embedding(embeddings, test_false_EdgeList_path, merge_op)
	test_labels_2 = np.zeros(test_false_data.shape[0], dtype=int)

	test_data = np.concatenate((test_true_data, test_false_data), axis=0)
	test_labels = np.concatenate((test_labels_1, test_labels_2), axis=0)

	return train_data, train_labels, val_data, val_labels, test_data, test_labels

def mix_embeddings(embeddings1, embeddings2, mix_embedding_output_path, merge_op='hada'):

	# embeddings1 = pickle.load(open(embedding_path_1, 'rb')) # dict {index: embedding vector}
	# embeddings2 = pickle.load(open(embedding_path_2, 'rb')) # dict {index: embedding vector}

	if type(embeddings1) != dict:
		mesh1 = {mesh for mesh in embeddings1.index2word}
	else:
		mesh1 = {mesh for mesh in embeddings1}

	if type(embeddings2) != dict:
		mesh2 = {mesh for mesh in embeddings2.index2word}
	else:
		mesh2 = {mesh for mesh in embeddings2}

	avail_mesh = mesh1.intersection(mesh2)

	embeddings_mix = {}
	for mesh in avail_mesh:
		embeddings_mix[mesh] =  merge_embeddings(embeddings1[mesh], embeddings2[mesh], merge_op) #(embeddings1[mesh]+embeddings2[mesh])/2

	# pickle.dump(embeddings_mix, open(mix_embedding_output_path, 'wb'))

	return embeddings_mix