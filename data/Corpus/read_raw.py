import os
import pubmed_parser as pp #https://github.com/titipata/pubmed_parser
import json
import tqdm
import pickle

raw_path = "./raw"

store_path = "./mesh_yearly/"

yearly_data_dict = {}

for file in tqdm.tqdm(os.listdir(raw_path)):
	if file.endswith(".gz"):
		print(file)
		#dict_id = {}
		dicts_file = pp.parse_medline_xml(raw_path+'/'+file, year_info_only=True, nlm_category=False) # return list of dictionary
		for dict_full in dicts_file:

			pmid = dict_full['pmid']

			year = dict_full['pubdate']
			if year not in yearly_data_dict:
				yearly_data_dict[year] = {}

			mesh_terms = dict_full['mesh_terms']
			if mesh_terms == '':
				continue
			mesh_terms_list = mesh_terms.split(';')
			#mesh_terms_id = []
			mesh_terms_word = []
			for term in mesh_terms_list:
				#mesh_terms_id.append(term.split(':')[0].lstrip())
				mesh_terms_word.append(term.split(':')[1])

			yearly_data_dict[year][pmid] = mesh_terms_word



all_terms = []
for year in yearly_data_dict:
	for ppr in yearly_data_dict[year]:
		all_terms.append(yearly_data_dict[year][ppr])
pickle.dump(all_terms, open('corpus_all_in_index.pkl', 'wb'))