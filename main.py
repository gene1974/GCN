import argparse
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import data
from model import *
from scorer import score # 官方 scorer
from pytorchtools import EarlyStopping # 开源 EarlyStopping 工具

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
assert(torch.cuda.is_available())

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float)
# torch.set_default_tensor_type(torch.FloatTensor)

def train_one_epoch(model, train_set, optimizer, entrophy):
    for _, batch in enumerate(train_set):
        train_losses = []
        if batch[0].shape[0] == 0:
            break
        token_ids = torch.tensor(batch[0], dtype = torch.long, device = 'cuda')
        dependency = torch.tensor(batch[1], dtype = float, device = 'cuda')
        subject_pos = torch.tensor(batch[2], dtype = torch.long, device = 'cuda')
        object_pos = torch.tensor(batch[2], dtype = torch.long, device = 'cuda')
        relation = torch.tensor(batch[4], dtype = torch.long, device = 'cuda')
        
        optimizer.zero_grad()
        output = model(token_ids, dependency, subject_pos, object_pos) # (n_batch, n_node, n_feature)
        loss = entrophy(output, relation)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

# train with valid
def do_train(model, args, train_set, dev_set = None):
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    entrophy = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience = 7, verbose = False)
    for epoch in range(args.epoch):
        train_losses = []
        valid_losses = []
        model.train()
        for _, batch in enumerate(train_set):
            if batch[0].shape[0] == 0:
                break
            token_ids, dependency, subject_pos, object_pos, relation = batch
            optimizer.zero_grad()
            output = model(token_ids, dependency, subject_pos, object_pos) # (n_batch, n_node, n_feature)
            loss = entrophy(output, relation)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        if dev_set is not None:
            model.eval()
            with torch.no_grad():
                for _, batch in enumerate(dev_set):
                    if batch[0].shape[0] == 0:
                        break
                    token_ids, dependency, subject_pos, object_pos, relation = batch
                    output = model(token_ids, dependency, subject_pos, object_pos) # (n_batch, n_node, n_feature)
                    loss = entrophy(output, relation)
                    valid_losses.append(loss.item())
                avg_valid_loss = np.average(valid_losses)
                early_stopping(avg_valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                time_stamp = time.strftime("%m-%d %H:%M:%S", time.localtime())
                print('[{}][epoch {:d}] Loss: Train: {:.4f} Dev: {:.4f}'.format(time_stamp, epoch + 1, np.average(train_losses), np.average(valid_losses)))
        else:
            time_stamp = time.strftime("%m-%d %H:%M:%S", time.localtime())
            print('[{}][epoch {:d}] Loss: Train: {:.4f}'.format(time_stamp, epoch + 1, np.average(train_losses)))

def train(args):
    train_set, dev_set, test_set, vocab, word_emb = data.load_datasets(args)
    args.n_relation = len(vocab['relation_dict'])
    model = GCNModel(args, word_emb).to('cuda')
    do_train(model, args, train_set, dev_set)
    args.time = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
    if args.savemodel:
        save_model(model, vocab, word_emb, args)
    do_test(model, train_set, vocab['relation_list'])
    do_test(model, dev_set, vocab['relation_list'])
    do_test(model, test_set, vocab['relation_list'])
    print(model)

def save_model(model, vocab, word_emb, args):
    print('Save model: {}'.format(args.time))
    torch.save(model.state_dict(), './savemodel/model_' + args.time)
    with open('./savemodel/vocab_' + args.time, 'wb') as f:
        pickle.dump(vocab, f)
        pickle.dump(word_emb, f)
        pickle.dump(args, f)

def do_test(model, dataset, relation_list):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _, batch in enumerate(dataset):
            if batch[0].shape[0] == 0:
                break
            token_ids, dependency, subject_pos, object_pos, relation = batch
            output = model(token_ids, dependency, subject_pos, object_pos)
            predict = torch.argmax(output, dim = 1)
            y_true.append(relation.cpu().numpy())
            y_pred.append(predict.cpu().numpy())
    y_true = [relation_list[i] for i in np.concatenate(y_true)]
    y_pred = [relation_list[i] for i in np.concatenate(y_pred)]
    prec_micro, recall_micro, f1_micro = score(y_true, y_pred)
    return prec_micro, recall_micro, f1_micro

def test(args):
    model_time = args.time
    with open('./savemodel/vocab_' + model_time, 'rb') as f:
        vocab = pickle.load(f)
        word_emb = pickle.load(f)
        args = pickle.load(f)
    train_set, dev_set, test_set, _, _ = data.load_datasets(args, vocab, word_emb)
    model = GCNModel(args, word_emb).to('cuda')
    model.load_state_dict(torch.load('./savemodel/model_' + model_time))
    do_test(model, train_set, vocab['relation_list'])
    do_test(model, dev_set, vocab['relation_list'])
    do_test(model, test_set, vocab['relation_list'])
    return args

def eval(args):
    model_time = args.time
    with open('./savemodel/vocab_' + model_time, 'rb') as f:
        vocab = pickle.load(f)
        word_emb = pickle.load(f)
        args = pickle.load(f)
    train_set, dev_set, test_set, _, _ = data.load_datasets(args, vocab, word_emb)
    model = GCNModel(args, word_emb).to('cuda')
    model.load_state_dict(torch.load('./savemodel/model_' + model_time))

    model.eval()
    y_pred = []
    y_true = []
    word_list, relation_list = vocab['word_list'], vocab['relation_list']
    with torch.no_grad():
        for _, batch in enumerate(test_set):
            if batch[0].shape[0] == 0:
                break
            token_ids, dependency, subject_pos, object_pos, relation = batch
            output = model(token_ids, dependency, subject_pos, object_pos)
            predict = torch.argmax(output, dim = 1)
            for j in range(len(token_ids)):
                print(' '.join([word_list[i] for i in token_ids[j]]))
                print('Sub: {};\tObj: {}'.format(
                    ' '.join([word_list[i] for i in token_ids[j, subject_pos[j, 0]: subject_pos[j, 1] + 1]]), 
                    ' '.join([word_list[i] for i in token_ids[j, object_pos[j, 0]: object_pos[j, 1] + 1]])))
                print('True: {};\nPredict: {}\n'.format(
                    relation_list[relation[j].item()], relation_list[predict[j].item()]))
            return
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default = 'train')
    parser.add_argument('-model', default = 'gcn')
    parser.add_argument('-time', default = '06151622', help = 'Test model')
    parser.add_argument('-nlayer', type = int, default = 1)
    parser.add_argument('-epoch', type = int, default = 50)
    parser.add_argument('-lr', type = float, default = 1e-4)
    parser.add_argument('-batch', type = int, default = 32)
    parser.add_argument('-path', default = './data/json/')
    parser.add_argument('-savemodel', type = bool, default = True)
    parser.add_argument('-hidden', type = int, default = 300)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'eval':
        eval(args)
    
    print(args)
        

