import json
import numpy as np
import torch
from torch.utils.data import Dataset

# https://nlp.stanford.edu/projects/tacred/
'''
['id', 'docid', 'relation', 'token', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type', 'stanford_pos', 'stanford_ner', 'stanford_head', 'stanford_deprel']
{'id': '61b3a5c8c9a882dcfcd2', 
'docid': 'AFP_ENG_20070218.0019.LDC2009T13', 
'relation': 'org:founded_by', 
'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 
'subj_start': 10, 
'subj_end': 12, 
'obj_start': 0, 
'obj_end': 1, 
'subj_type': 'ORGANIZATION', 
'obj_type': 'PERSON', 
'stanford_pos': ['NNP', 'NNP', 'VBD', 'IN', 'NNP', 'JJ', 'NN', 'TO', 'VB', 'DT', 'DT', 'NNP', 'NNP', '-LRB-', 'NNP', '-RRB-', ',', 'VBG', 'DT', 'NN', 'IN', 'CD', 'NNS', 'IN', 'NN', ',', 'VBG', 'JJ', 'NN', 'NNP', 'NNP', 'NNP', 'TO', 'VB', 'NN', 'CC', 'VB', 'DT', 'NN', 'NN', '.'], 
'stanford_ner': ['PERSON', 'PERSON', 'O', 'O', 'DATE', 'DATE', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 
# 依存树中父亲结点是哪个节点(下标从1开始)
'stanford_head': [2, 3, 0, 5, 3, 7, 3, 9, 3, 13, 13, 13, 9, 15, 13, 15, 3, 3, 20, 18, 23, 23, 18, 25, 23, 3, 3, 32, 32, 32, 32, 27, 34, 27, 34, 34, 34, 40, 40, 37, 3], 
# 依存关系类型
'stanford_deprel': ['compound', 'nsubj', 'ROOT', 'case', 'nmod', 'amod', 'nmod:tmod', 'mark', 'xcomp', 'det', 'compound', 'compound', 'dobj', 'punct', 'appos', 'punct', 'punct', 'xcomp', 'det', 'dobj', 'case', 'nummod', 'nmod', 'case', 'nmod', 'punct', 'xcomp', 'amod', 'compound', 'compound', 'compound', 'dobj', 'mark', 'xcomp', 'dobj', 'cc', 'conj', 'det', 'compound', 'dobj', 'punct']}
'''

# standford ner -> bioes tagging
def tagging(tags):
    bioes_tags = ['O' for _ in tags]
    i = 0
    while i < len(tags):
        if tags[i] == 'O':
            i += 1
        else:
            bioes_tags[i] = 'B-' + tags[i]
            j = i + 1
            while j < len(tags) and tags[i] == tags[j]:
                bioes_tags[j] = 'I-' + tags[j]
                j += 1
            if j == i + 1:
                bioes_tags[i] = 'S-' + tags[i]
            else:
                bioes_tags[j - 1] = 'E-' + tags[j - 1]
            i = j
    return bioes_tags

# load tarced data file
def load_data_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
        dataset = []
        word_set = set()
        relation_set = set()
        for line in data:
            relation = line['relation']
            token = line['token']
            subj_start = line['subj_start']
            subj_end = line['subj_end']
            obj_start = line['obj_start']
            obj_end = line['obj_end']
            stanford_head = line['stanford_head']
            stanford_ner = line['stanford_ner'] # ner tag, need to transfer
            dataset.append({
                'tokens': token,
                # 'tags': tagging(stanford_ner),
                'subject': {'start': subj_start, 'end': subj_end},
                'object': {'start': obj_start, 'end': obj_end},
                'relation': relation,
                'dependency': stanford_head
            })
            word_set |= set(token)
            relation_set.add(relation)
    print(len(dataset), len(word_set), len(relation_set)) # 68124 62152 13 11
    return dataset, word_set, relation_set

# standford head -> dependency matrix
def create_matrix(head):
    matrix = np.zeros((len(head), len(head)), dtype = int)
    for i, j in enumerate(head):
        matrix[i, i] = 1 # self loop
        if j == 0: # root
            continue
        matrix[i, j - 1] = 1
        matrix[j - 1, i] = 1
    return matrix

# encode with padding
def map_and_pad(dataset, word_dict, relation_dict):
    max_seq = 96
    n_data = len(dataset)
    token_ids = np.zeros((n_data, max_seq), dtype = int)
    dependency = np.zeros((n_data, max_seq, max_seq), dtype = float)
    subject_pos = np.zeros((n_data, 2), dtype = int)
    object_pos = np.zeros((n_data, 2), dtype = int)
    relation = np.zeros((n_data,), dtype = int)
    for i, data in enumerate(dataset):
        n_token = len(data['tokens'])
        token_ids[i, :n_token] = [word_dict[i] for i in data['tokens']]
        dependency[i, :n_token, :n_token] = create_matrix(data['dependency'])
        subject_pos[i] = data['subject']['start'], data['subject']['end']
        object_pos[i] = data['object']['start'], data['object']['end']
        relation[i] = relation_dict[data['relation']]
    return token_ids, dependency, subject_pos, object_pos, relation

# Dataset
class TacredDataset(Dataset):
    def __init__(self, token_ids, dependency, subject_pos, object_pos, relation, batch_size = 16):
        super().__init__()
        self.batch_size = batch_size
        self.n_data = len(token_ids)
        self.token_ids = torch.tensor(token_ids, dtype = torch.long)
        self.dependency = torch.tensor(dependency, dtype = torch.float64)
        self.subject_pos = torch.tensor(subject_pos, dtype = torch.long)
        self.object_pos = torch.tensor(object_pos, dtype = torch.long)
        self.relation = torch.tensor(relation, dtype = torch.long)
    def __len__(self):
        return self.n_data
    def __getitem__(self, i):
        begin = i * self.batch_size
        end = min((i + 1) * self.batch_size, self.n_data)
        
        token_ids = self.token_ids[begin: end].to('cuda')
        dependency = self.dependency[begin: end].to('cuda')
        subject_pos = self.subject_pos[begin: end].to('cuda')
        object_pos = self.object_pos[begin: end].to('cuda')
        relation = self.relation[begin: end].to('cuda')
        
        return token_ids, dependency, subject_pos, object_pos, relation

# load pretrained glove embedding
def load_glove(word_to_ix, dim = 100):
    if dim == 100:
        path = '/data/pretrained/Glove/glove.6B.100d.txt'
    elif dim == 300:
        path = '/data/pretrained/Glove/glove.840B.300d.txt'
    word_emb = []
    word_emb = np.zeros((len(word_to_ix), dim), dtype = float)
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split(' ') # [word emb1 emb2 ... emb n]
            word = data[0]
            if word in word_to_ix:
                word_emb[word_to_ix[word]] = [float(i) for i in data[1:]]
    print('Word embedding: {}'.format(word_emb.shape))
    return torch.tensor(word_emb, dtype = torch.float64)

def load_datasets(args, vocab = None, word_emb = None):
    # load data files
    train_data, train_word_set, train_relation_set = load_data_file(args.path + 'train.json')
    dev_data, dev_word_set, dev_dev_relation_set = load_data_file(args.path + 'dev.json')
    test_data, test_word_set, test_relation_set = load_data_file(args.path + 'test.json')
    
    if vocab is None:
        word_set = train_word_set | dev_word_set | test_word_set
        relation_set = train_relation_set | dev_dev_relation_set | test_relation_set
        
        word_list = list(word_set)
        word_list.insert(0, '<PAD>')
        word_list.insert(1, '<OOV>')
        word_dict = {word_list[i]: i for i in range(len(word_list))}
        relation_list = list(relation_set)
        relation_dict = {relation_list[i]: i for i in range(len(relation_list))}
        print('Words: {}, Relations: {}'.format(len(word_list), len(relation_list))) # 74805, 43

        # load used word embedding
        word_emb = load_glove(word_dict, 300)
        vocab = {
            'word_list': word_list,
            'word_dict': word_dict,
            'relation_list': relation_list,
            'relation_dict': relation_dict,
        }
    else:
        word_dict = vocab['word_dict']
        relation_dict = vocab['relation_dict']
    
    # create datasets
    token_ids, dependency, subject_pos, object_pos, relation = map_and_pad(train_data, word_dict, relation_dict)
    train_set = TacredDataset(token_ids, dependency, subject_pos, object_pos, relation, args.batch)
    token_ids, dependency, subject_pos, object_pos, relation = map_and_pad(dev_data, word_dict, relation_dict)
    dev_set = TacredDataset(token_ids, dependency, subject_pos, object_pos, relation, args.batch)
    token_ids, dependency, subject_pos, object_pos, relation = map_and_pad(test_data, word_dict, relation_dict)
    test_set = TacredDataset(token_ids, dependency, subject_pos, object_pos, relation, args.batch)
    # print(len(train_data), len(dev_data), len(test_data)) # 68124 22631 15509
    
    return train_set, dev_set, test_set, vocab, word_emb

if __name__ == '__main__':
    train_set, dev_set, test_set, word_emb, datas = load_datasets()
    print(datas['relation_list'])
    print(datas['relation_dict']['no_relation'])

    # for k in datas['relation_dict']:
    #     print(datas['relation_dict'][k], k)
    # print(train_set.subject_pos)
    
    # print(tagging(['PERSON', 'PERSON', 'O', 'O', 'DATE', 'DATE', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']))
    # print(create_matrix([2, 3, 0, 5, 3]))

