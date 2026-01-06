from drs import util

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import Dataset, DataLoader

class InputFeature:
    def __init__(self, text_ints, levels):
        self.text_ints = text_ints
        self.levels = levels

def convert_data_to_features(args, dx, dy, vocab_to_int):
    text_ints = np.array([vocab_to_int.get(word, vocab_to_int['_UNK']) for word in dx], dtype=np.int32)
    text_ints = text_ints[:args.seq_length].reshape(1, -1)

    levels = [args.mapping["l4_map"][label] for label in dy]

    return InputFeature(text_ints, levels)

class TextDataset(Dataset):
    def __init__(self, args, data_x, data_y):
        self.seq_length = args.seq_length
        self.num_classes_layer = args.num_classes_layer
        self.total_classes = args.total_classes
        self.data = [
            convert_data_to_features(args, dx, dy, args.word2idx)
            for dx, dy in tqdm(zip(data_x, data_y), total=len(data_y))
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = self.data[index]
        features = self.pad_features(feature.text_ints)
        onehot_levels = self.create_onehot_labels(feature.levels,self.num_classes_layer[3])

        return (torch.tensor(features),
                torch.tensor(onehot_levels))

    def pad_features(self, text_ints):
        padded = np.zeros((self.seq_length,), dtype=int)
        length = min(self.seq_length, text_ints.shape[1])
        padded[-length:] = text_ints[0, :length]
        return padded

    def create_onehot_labels(self, labels_index, num_labels):
        onehot = torch.zeros(num_labels, dtype=int)
        onehot[labels_index] = 1
        return onehot

# Define embedding
def embedding(args):
    args.embedding_matrix, args.word2idx = util.word2matrix(args)

# Define class label
def class_label(args):
    args.mapping, args.num_classes_layer, args.total_classes = util.class_label(args.path_atc)

# Define dataset
def define_ds(args, data_path: str):
    args.seq_length = 1024
    # load data
    df = pd.read_pickle(data_path)
    # add note or not
    add_note = 'X_note' if args.note else 'X'
    # create dataset
    dataset = TextDataset(args, df[add_note], df['y'])

    return dataset

# Define dataloader
def define_dl(dataset, batch_size: int, shuffle:bool = True, drop_last:bool =False, num_workers: int = 8, pin_memory: bool = True):
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers,
                            pin_memory=pin_memory
                            )
    return dataloader
