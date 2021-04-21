# import ge
import pickle
from ge.walker import RandomWalker
from ge import Struc2Vec
import networkx as nx
from gensim.models import Word2Vec
import numpy as np

def gen_walks(graph, p=0.25, q=4, num_walks=80, walk_length=10):
	walker = RandomWalker(graph, p, q, use_rejection_sampling=0)
	walker.preprocess_transition_probs()
	sentences = walker.simulate_walks(num_walks, walk_length, workers=1, verbose=1)
	print('Walks generated! Number of sentences is {}'.format(len(sentences)))
	return sentences

def gen_walks_s2v(graph, num_walks=80, walk_length=10):
	model = Struc2Vec(graph, walk_length=walk_length, num_walks=num_walks, workers=4, verbose=0) #init model
	sentences = model.sentences
	print('Walks generated! Number of sentences is {}'.format(len(sentences)))
	return sentences

def gen_embeddings(sentences, size=128, workers=4, window=10, iter=1):
	print('Learning embedding...')
	model = Word2Vec(sentences=sentences, size=size, window=window, workers=workers, iter=iter)
	print('Learning embeddings done!')
	return model.wv

def mix_embeddings(graph_path_list, sentence_path_list, num_walks_list, walk_length_list, p=0.25, q=4, window=10, size=128, workers=4, iter=1, directed=1):
	
	corpus = []

	for i in range(len(graph_path_list)):
		if directed == 1:
			G = nx.read_edgelist(graph_path_list[i], create_using=nx.DiGraph(), nodetype=None, data=[('weight',int)])#read graph
		else:
			G = nx.read_edgelist(graph_path_list[i], nodetype=None, data=[('weight',int)])#read graph
		docs = gen_walks(G, p, q, num_walks_list[i], walk_length_list[i])
		# corpus += docs
		corpus += list(np.random.choice(docs, size=15000000))

	for path in sentence_path_list:
		docs = pickle.load(open(path, 'rb'))
		# corpus += docs 
		corpus += list(np.random.choice(docs, size=15000000))

	embeddings = gen_embeddings(corpus, window=window, iter=iter)

	return embeddings

def mix_embeddings_n2v_s2v(graph_path_list, sentence_path_list, num_walks_list, walk_length_list, p=0.25, q=4, window=10, size=128, workers=4, iter=1, directed=1):
	
	corpus = []

	for i in range(len(graph_path_list)):
		if directed == 1:
			G = nx.read_edgelist(graph_path_list[i], create_using=nx.DiGraph(), nodetype=None, data=[('weight',int)])#read graph
		else:
			G = nx.read_edgelist(graph_path_list[i], nodetype=None, data=[('weight',int)])#read graph
		if i == 0:
			docs = gen_walks(G, p, q, num_walks_list[i], walk_length_list[i])
		if i == 1:
			docs = gen_walks_s2v(G, num_walks_list[i], walk_length_list[i])
		corpus += docs

	for path in sentence_path_list:
		docs = pickle.load(open(path, 'rb'))
		corpus += docs

	embeddings = gen_embeddings(corpus, window=window, iter=iter)

	return embeddings


def mix_embeddings_s2v(graph_path_list, sentence_path_list, num_walks_list, walk_length_list, p=0.25, q=4, window=10, size=128, workers=4, iter=1, directed=1):
	
	corpus = []

	for i in range(len(graph_path_list)):
		if directed == 1:
			G = nx.read_edgelist(graph_path_list[i], create_using=nx.DiGraph(), nodetype=None, data=[('weight',int)])#read graph
		else:
			G = nx.read_edgelist(graph_path_list[i], nodetype=None, data=[('weight',int)])#read graph
		docs = gen_walks_s2v(G, num_walks_list[i], walk_length_list[i])
		corpus += docs

	for path in sentence_path_list:
		docs = pickle.load(open(path, 'rb'))
		corpus += docs

	embeddings = gen_embeddings(corpus, window=window, iter=iter)

	return embeddings
 
sentence_path_list = ['./Corpus/corpus_all_in_index.pkl']
graph_path_list = ['./Ontology/MeSH_EdgeList.txt', './SemMedDB_treat_MeSH/treat_EdgeList_train_true.txt']
num_walks_list = [80,80]
walk_length_list = [10,10]

'''
embeddings = mix_embeddings(graph_path_list, sentence_path_list)
pickle.dump(embeddings, open('fused_embeddings_default_graph_setting_iter5.pkl','wb'))



corpus = []

graph = nx.read_edgelist(graph_path_list[0], create_using=nx.DiGraph(), nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=0.25, q=4, num_walks=507, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

graph = nx.read_edgelist(graph_path_list[1], create_using=nx.DiGraph(), nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=0.25, q=4, num_walks=1674, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

docs = pickle.load(open(sentence_path_list[0], 'rb'))
corpus += docs

model = Word2Vec(sentences=sentences, size=128, window=10, workers=4)
embeddings = model.wv

pickle.dump(embeddings, open('fused_embeddings_1674_10_507_10.pkl','wb'))

model = Word2Vec(sentences=sentences, size=128, window=5, workers=4)
embeddings = model.wv

pickle.dump(embeddings, open('fused_embeddings_1674_10_507_10_w5.pkl','wb'))


corpus = []

graph = nx.read_edgelist(graph_path_list[0], create_using=nx.DiGraph(), nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=1, q=1, num_walks=80, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

graph = nx.read_edgelist(graph_path_list[1], create_using=nx.DiGraph(), nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=1, q=1, num_walks=80, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

docs = pickle.load(open(sentence_path_list[0], 'rb'))
corpus += docs

model = Word2Vec(sentences=sentences, size=128, window=10, workers=4)
embeddings = model.wv

pickle.dump(embeddings, open('fused_embeddings_deepwalk.pkl','wb'))


corpus = []

graph = nx.read_edgelist(graph_path_list[0], nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=0.25, q=4, num_walks=80, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

graph = nx.read_edgelist(graph_path_list[1], nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=0.25, q=4, num_walks=80, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

docs = pickle.load(open(sentence_path_list[0], 'rb'))
corpus += docs

model = Word2Vec(sentences=sentences, size=128, window=10, workers=4)
embeddings = model.wv

pickle.dump(embeddings, open('fused_embeddings_default_undirect.pkl','wb'))

corpus = []

graph = nx.read_edgelist(graph_path_list[0], nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=0.25, q=4, num_walks=507, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

graph = nx.read_edgelist(graph_path_list[1], nodetype=None, data=[('weight',int)])#read graph
sentences = gen_walks(graph, p=0.25, q=4, num_walks=1674, walk_length=10, workers=1, use_rejection_sampling=0)
corpus += sentences

docs = pickle.load(open(sentence_path_list[0], 'rb'))
corpus += docs

model = Word2Vec(sentences=sentences, size=128, window=10, workers=4)
embeddings = model.wv

pickle.dump(embeddings, open('fused_embeddings_default_undirect_balance.pkl','wb'))

'''