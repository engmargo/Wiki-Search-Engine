'''
Author: Zim Gong
Edited by: Lea Lei
This file is a template code file for the Search Engine. 
'''

import csv
import gzip
import json
import jsonlines
import os
import pickle
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

from models import BaseSearchEngine, SearchResponse

# project library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType,BasicInvertedIndex
from ranker import *
from l2r import L2RRanker, L2RFeatureExtractor
from vector_ranker import VectorRanker
from network_features import NetworkFeatures

DATA_PATH = './data/'
CACHE_PATH = './__cache__/'

STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
DOC2QUERY_PATH = DATA_PATH + 'doc2query.csv'
DATASET_PATH = DATA_PATH + 'wikipedia_200k_dataset.jsonl.gz'
RELEVANCE_TRAIN_PATH = DATA_PATH + 'relevance.train.csv'
EDGELIST_PATH = DATA_PATH + 'edgelist.csv.gz'
ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = DATA_PATH + \
    'wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy'
NETWORK_STATS_PATH = DATA_PATH + 'network_stats.csv'
DOC_IDS_PATH = DATA_PATH + 'document-ids.txt'
MAIN_INDEX = '1012body_index_aug' 
TITLE_INDEX = '1012title_index'

class SearchEngine(BaseSearchEngine):
    def __init__(self,
                 max_docs: int = -1,
                 ranker: str = 'BM25',
                 l2r: bool = False,
                 aug_docs: bool = False
                 ) -> None:
        # 1. Create a document tokenizer using document_preprocessor Tokenizers
        # 2. Load stopwords, network data, categories, etc
        # 3. Create an index using the Indexer and IndexType (with the Wikipedia JSONL and stopwords)
        # 4. Initialize a Ranker/L2RRanker with the index, stopwords, etc.
        # 5. If using L2RRanker, train it here.

        self.l2r = False

        print('Initializing Search Engine...')
        self.stopwords = set()
        with open(STOPWORD_PATH, 'r') as f:
            for line in f:
                self.stopwords.add(line.strip())

        self.doc_augment_dict = None
        if aug_docs:
            print('Loading doc augment dict...')
            self.doc_augment_dict = defaultdict(lambda: [])
            with open(DOC2QUERY_PATH, 'r') as f:
                data = csv.reader(f)
                for idx, row in tqdm(enumerate(data)):
                    if idx == 0:
                        continue
                    self.doc_augment_dict[row[0]].append(row[2])

        print('Loading indexes...')
        self.preprocessor = RegexTokenizer('\w+')
        if not os.path.exists(MAIN_INDEX):
            self.main_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, DATASET_PATH, self.preprocessor,
            self.stopwords, 50, max_docs=max_docs,
            doc_augment_dict=self.doc_augment_dict
        )
        else: 
            self.main_index = BasicInvertedIndex()
            self.main_index.load(MAIN_INDEX)
        
        if not os.path.exists(TITLE_INDEX):
            self.title_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, DATASET_PATH, self.preprocessor,
            self.stopwords, 2, max_docs=max_docs,
            text_key='title'
        )
        else:
            self.title_index = BasicInvertedIndex()
            self.title_index.load(TITLE_INDEX)
        
        print('Loading raw text dict...')
        with open(RELEVANCE_TRAIN_PATH, 'r',encoding = 'unicode_escape') as f:
            data = csv.reader(f)
            train_docs = set()
            for idx, row in tqdm(enumerate(data)):
                if idx == 0:
                    continue
                train_docs.add(row[2])

        if not os.path.exists(CACHE_PATH + 'raw_text_dict_train.pkl'):
            if not os.path.exists(CACHE_PATH):
                os.makedirs(CACHE_PATH)
            self.raw_text_dict = defaultdict()
            file = gzip.open(DATASET_PATH, 'rt')
            with jsonlines.Reader(file) as reader:
                while True:
                    try:
                        data = reader.read()
                        if str(data['docid']) in train_docs:
                            self.raw_text_dict[str(
                                data['docid'])] = data['text'][:500]
                    except:
                        break
            pickle.dump(
                self.raw_text_dict,
                open(CACHE_PATH + 'raw_text_dict_train.pkl', 'wb')
            )
        else:
            self.raw_text_dict = pickle.load(
                open(CACHE_PATH + 'raw_text_dict_train.pkl', 'rb')
            )
        del train_docs, data

        print('Loading ranker...')
        self.set_personlaized()
        self.set_ranker(ranker)
        self.set_l2r(l2r)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'BM25',user_id:int  = None) -> None:
        if ranker == 'VectorRanker':
            with open(ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH, 'rb') as f:
                self.encoded_docs = np.load(f)
            with open(DOC_IDS_PATH, 'r') as f:
                self.row_to_docid = [int(line.strip()) for line in f]
            self.ranker = VectorRanker(
                'sentence-transformers/msmarco-MiniLM-L12-cos-v5',
                self.encoded_docs, self.row_to_docid
            )
        else:
            if ranker == 'BM25':
                self.scorer = BM25(self.main_index)
            elif ranker == "PersonalizedBM25":
                if not self.rel_doc_indexs:
                    raise ValueError('rel_doc_indexs not set')
                else:
                    self.scorer = PersonalizedBM25(self.main_index,self.rel_doc_indexs[user_id])
            elif ranker == "WordCountCosineSimilarity":
                self.scorer = WordCountCosineSimilarity(self.main_index)
            elif ranker == "DirichletLM":
                self.scorer = DirichletLM(self.main_index)
            elif ranker == "PivotedNormalization":
                self.scorer = PivotedNormalization(self.main_index)
            elif ranker == "TF_IDF":
                self.scorer = TF_IDF(self.main_index)
            else:
                raise ValueError("Invalid ranker type")
            self.ranker = Ranker(
                self.main_index, self.preprocessor, self.stopwords,
                self.scorer, self.raw_text_dict)
        if self.l2r:
            self.pipeline.ranker = self.ranker
        else:
            self.pipeline = self.ranker

    def set_l2r(self, l2r: bool = True,pg_weights:dict = None
                ,aug_docs_ce:bool=False
                ) -> None: #pg_weights{docid:weight}
        if self.l2r == l2r:
            return
        if not l2r:
            self.pipeline = self.ranker
            self.l2r = False
            
            if os.path.exists(NETWORK_STATS_PATH):
                os.remove(NETWORK_STATS_PATH)
          
        else:
            print('Loading categories...')
            if not os.path.exists(CACHE_PATH + 'docid_to_categories.pkl'):
                docid_to_categories = defaultdict()
                with gzip.open(DATASET_PATH, 'rt') as f:
                    for line in tqdm(f):
                        data = json.loads(line)
                        docid_to_categories[data['docid']] = data['categories']
                pickle.dump(
                    docid_to_categories,
                    open(CACHE_PATH + 'docid_to_categories.pkl', 'wb')
                )
            else:
                docid_to_categories = pickle.load(
                    open(CACHE_PATH + 'docid_to_categories.pkl', 'rb')
                )

            print('Loading recognized categories...')
            category_counts = Counter()
            for categories in tqdm(docid_to_categories.values()):
                category_counts.update(categories)
            self.recognized_categories = set(
                [category for category, count in category_counts.items()
                 if count > 1000]
            )
            if not os.path.exists(CACHE_PATH + 'doc_category_info.pkl'):
                self.doc_category_info = defaultdict()
                for docid, categories in tqdm(docid_to_categories.items()):
                    self.doc_category_info[docid] = [
                        category for category in categories if category in self.recognized_categories
                    ]
                pickle.dump(
                    self.doc_category_info,
                    open(CACHE_PATH + 'doc_category_info.pkl', 'wb')
                )
            else:
                self.doc_category_info = pickle.load(
                    open(CACHE_PATH + 'doc_category_info.pkl', 'rb')
                )
            del docid_to_categories, category_counts

            print('Loading network features...')
            self.network_features = defaultdict(lambda:defaultdict(float))
            if not os.path.exists(NETWORK_STATS_PATH):
                nf = NetworkFeatures()
                graph = nf.load_network(EDGELIST_PATH, total_edges=92650947)
                if pg_weights:
                    name2id = {name:id for id,name in enumerate(graph.names)}
                    pg_weights = {name2id[name]:weight for name,weight in pg_weights.items()}
                net_feats_df = nf.get_all_network_statistics(graph,weights = pg_weights)
                del graph
                net_feats_df.to_csv(NETWORK_STATS_PATH, index=False)
                cols = net_feats_df.columns.tolist()
                for row in tqdm(net_feats_df.itertuples(index=False)):
                    for idx,col in enumerate(cols):
                        if idx ==0:
                            pass
                        else:
                            self.network_features[row[0]][col] = row[idx]
                del net_feats_df
            else:
                with open(NETWORK_STATS_PATH, 'r') as f:
                    for idx, row in tqdm(enumerate(f)):
                        if idx == 0:
                            continue
                        splits = row.strip().split(',')
                        self.network_features[int(splits[0])] = {
                            'pagerank': float(splits[1]),
                            'hub_score': float(splits[2]),
                            'authority_score': float(splits[3])
                        }
                        
            if aug_docs_ce:
                for docid,text in self.raw_text_dict.items():
                    text = ' '.join(self.doc_augment_dict[docid])+text
                    text = text[:500]
                    self.raw_text_dict[docid] = text
                    
            self.cescorer = CrossEncoderScorer(self.raw_text_dict)
            self.fe = L2RFeatureExtractor(
                self.main_index, self.title_index, self.doc_category_info,
                self.preprocessor, self.stopwords, self.recognized_categories,
                self.network_features, self.cescorer
            )

            print('Loading L2R ranker...')
            self.pipeline = L2RRanker(
                self.main_index, self.title_index, self.preprocessor,
                self.stopwords, self.ranker, self.fe
            )

            print('Training L2R ranker...')
            self.pipeline.train(RELEVANCE_TRAIN_PATH)
            self.l2r = True
            
    def set_personlaized(self,REL_DOCS_PATH:str = None,max_docs: int = -1)->None:
        if REL_DOCS_PATH:
            path = DATA_PATH+REL_DOCS_PATH

            with open(path,'r') as f:
                rel_doc_users = [json.loads(line) for line in f]
            
            rel_paths = []
            for user in rel_doc_users:
                path = CACHE_PATH+f'rel_docs_user{user['user_id']}.jsonl'
                with open(path,'w') as f:
                    for doc in user['seed_docs']:
                        f.write(json.dumps(doc)+'\n')
                rel_paths.append((user['user_id'],path))
            
            rel_doc_indexs = defaultdict(InvertedIndex)
            for user_id,path in rel_paths:
                rel_doc_index = Indexer.create_index(
                    IndexType.BasicInvertedIndex, path, self.preprocessor,
                    self.stopwords, 50, max_docs=max_docs,
                    doc_augment_dict=self.doc_augment_dict
                )
                rel_doc_indexs[user_id] = rel_doc_index
                
            self.rel_doc_indexs= rel_doc_indexs
            
        else:
            self.rel_doc_indexs = None
        
    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    search_obj = SearchEngine(max_docs=1000, ranker='VectorRanker', l2r=True)
    return search_obj


def main():
    search_obj = SearchEngine(max_docs=10000)
    search_obj.set_l2r(True)
    query = "What is the capital of France?"
    results = search_obj.search(query)
    print(results[:5])


# if __name__ == '__main__':
#     engine = SearchEngine()
#     print(len(engine.stopwords))
#     engine.set_personlaized('personalization.jsonl')

    
