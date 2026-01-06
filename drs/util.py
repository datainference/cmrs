import os
import torch
import shutil 
import random 
import logging
import pandas as pd
import numpy as np
from gensim.models import FastText
from collections import OrderedDict
from sklearn.metrics import f1_score
from multiprocessing import Pool, cpu_count

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False

def save_checkpoint(state, is_best, filename, fname_log):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'../Models/best_model/model_best_{fname_log}.pth')

def class_label(atc_path):
    metadata = pd.read_pickle(atc_path)
    # Extract unique values for each level of substring length
    levels = [1, 3, 4, 5]
    level_sets = {length: sorted(set(metadata['atc_code'].str.strip().str[:length].values)) for length in levels}

    # Create mappings for each level
    mapping = {
        'label_map': {i[:5]: li for li, i in enumerate(sorted(set(metadata['atc_code'].str.strip().str[:5].values)))},
        'label_reverse': {li: i[:5] for li, i in enumerate(sorted(set(metadata['atc_code'].str.strip().str[:5].values)))},
        'label_desc': {i[0].strip()[:5]: i[1] for i in metadata.values.tolist()}
    }

    # Populate mappings for each level
    for i, length in enumerate(levels):
        level_map = {l: li for li, l in enumerate(level_sets[length])}
        level_map_reverse = {li: l for li, l in enumerate(level_sets[length])}
        mapping[f'l{i+1}_map'] = level_map
        mapping[f'l{i+1}_map_reverse'] = level_map_reverse

    # Calculate number of classes for each level
    num_classes_layer = [len(level_sets[length]) for length in levels]
    total_classes = np.sum(num_classes_layer)
    return mapping, num_classes_layer, total_classes

def debug_result(args, feature):
    int_to_vocab = {v: k for k, v in args.word2idx.items()}
    print('Input:')
    print(' '.join([int_to_vocab[x] for x in feature[0].numpy()]).replace('_UNK', ''))
    print('Output:')
    print(str([f'{args.mapping[f"l4_map_reverse"][i]}' for i in np.where(feature[1])[0]]))



def _compute_f1(threshold_y):
    threshold, y_true, y_pred = threshold_y
    y_pred_bin = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_bin, average='micro')

def find_best_threshold(y_true, y_pred, thresholds=np.arange(0.1, 0.9, 0.05)):
    order = [(threshold, y_true, y_pred) for threshold in thresholds]
    f1_scores = []
    with Pool(processes=(cpu_count()-2)) as pool:
        for score in pool.imap(_compute_f1, order):
            f1_scores.append(score)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def calculate_ddi_rate(input_med, adj_ddi_matrix, map_l4_re):
    all_cnt = 0
    ddi_cnt = 0
    list_dd_name = []
    list_dd_idx = []
    for label_set in input_med:
        list_med = np.where(label_set)[0]
        for i, med_i in enumerate(list_med):
            for med_j in list_med[i+1:]:
                all_cnt += 1
                ddi_cnt += adj_ddi_matrix[med_i, med_j]
                if (adj_ddi_matrix[med_i, med_j] == 1): 
                    list_dd_name.append([map_l4_re[med_i], map_l4_re[med_j]])
                    list_dd_idx.append([med_i, med_j])
    ddi_rate = ddi_cnt/ all_cnt if all_cnt > 0 else 0
    
    return ddi_rate, list_dd_name, list_dd_idx

def cal_ddi_rate_train(predict, adj_ddi, k=5):
    ddi_count = 0
    num_pair = 0
    _, topk = torch.topk(predict.detach(), k)
    for items in topk:
        for n_item, i_item in enumerate(items):
            for j_item in items[n_item+1:]:
                num_pair += 1
                ddi_count += adj_ddi[i_item, j_item]
    return ddi_count , num_pair
                

def printlog(massage):
    print(massage)
    logging.info(massage)

def word2matrix(args):
    model = FastText.load(args.embeddimg_path)
    wv = model.wv
    word2idx = OrderedDict({"_UNK": 0})
    embedding_size = wv.vector_size
    for k in wv.key_to_index:
        word2idx[k] = wv.key_to_index[k] + 1
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in word2idx.items():
        if key == "_UNK":
            embedding_matrix[value] = [0. for _ in range(embedding_size)]
        else:
            embedding_matrix[value] = wv[key]
    return embedding_matrix, word2idx

