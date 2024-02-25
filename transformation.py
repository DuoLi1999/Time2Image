import logging
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from sklearn.preprocessing import LabelEncoder
import torch
import torch.utils.data as Data
from PIL import Image
from tqdm import tqdm
import csv
import numpy as np
import os
import argparse
import ml_collections
label_encoder = LabelEncoder()

def get_config(img_size, patch_size):
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.img_size = img_size
    config.patch_size = patch_size
    assert img_size % patch_size == 0
    config.seq_length = int((img_size / patch_size)**2)
    config.shape_size = int(img_size / patch_size)
    return config

def read_row(line_number, filepath):
    """
    Get data name and corresponding num_classes.
    """
    with open(filepath, "r") as file:
        for current_line, line in enumerate(file, start=1):
            if current_line == line_number:
                name, classes = line.strip().split('\t')
                return name, int(classes)
    return None, None  

def read_tsv(dir_data, num_classes):
    data = []
    label = []
    with open(dir_data, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            float_row = list(map(float, row))  # Using map for conversion
            data.append(float_row[1:])
            label.append(float_row[0])
        speed = np.array(data)
        label = label_encoder.fit_transform(label)
        verify = sorted(set(label)) == list(range(num_classes))
        if verify:
            return speed, label
        else:
            raise ValueError('Label verification failed')  # Changed assertion to ValueError


def get_new_seq(data, size=196):
    '''
    Resize the data length.
    '''
    seq = np.nan_to_num(data, nan=0)
    target_length = size
    x_new = np.linspace(0, len(seq)-1, target_length)

    f = interp1d(np.arange(len(seq)), seq, kind='cubic')
    seq_new = f(x_new)
    normalized_series = (seq_new - seq_new.min()) / (seq_new.max() - seq_new.min())
    normalized_series = 2 * normalized_series - 1

    mu = np.mean(seq_new)
    sigma = np.sqrt(np.var(seq_new))
    Z = (seq_new - mu) / sigma
    Z -= Z.mean()
    return seq_new

def resize_batch_time_series(batch_data, size=196):
    '''
    Resize a batch of time series data to the same length.
    '''
    # Apply get_new_seq to each time series in the batch
    resized_batch = np.array([get_new_seq(seq, size=size) for seq in batch_data])
    return resized_batch

def expand_matrices(batch, config):
    '''
    
    '''
    batch = batch.reshape(-1, config.shape_size, config.shape_size)
    batch_expanded = np.repeat(np.repeat(batch, config.patch_size, axis=1), config.patch_size, axis=2)
    return batch_expanded

def get_gmat(config, m=0, n=0, std=1):

    X = np.linspace(-std, std, config.patch_size)
    Y = np.linspace(-std, std, config.patch_size)
    x, y = np.meshgrid(X, Y)

    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # var = v
    Sigma = [[1,0],
             [0,1]]
    mu = [m,n]
    z = multivariate_normal(mu, Sigma)
    pdf = np.array(z.pdf(pos))
    pdf = (pdf - np.min(pdf)) / (np.max(pdf) - np.min(pdf))
    pdf = np.tile(pdf,(config.shape_size,config.shape_size))
    return pdf


def main(args):
    config = get_config(args.image_size, args.patch_size)
    name, num_classes = read_row(args.index, args.name_class_path)
    data_dir = [os.path.join(args.data_path, name, f"{name}_TRAIN.tsv"),
                os.path.join(args.data_path, name, f"{name}_TEST.tsv")]
    data_train, label_train= read_tsv(data_dir[0], num_classes)
    data_test, label_test  = read_tsv(data_dir[1], num_classes)

    train_dir = os.path.join(args.results_dir +'-img_size' + str(args.image_size)+ '-patch_size' + str(args.patch_size), name, 'train')
    test_dir = os.path.join(args.results_dir +'-img_size' + str(args.image_size)+ '-patch_size' + str(args.patch_size), name, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for i in range(num_classes):
        os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, str(i)), exist_ok=True)

    gaussian_mat = get_gmat(config,m=0, n=0, std=args.std)
    data_train = expand_matrices(resize_batch_time_series(data_train, config.seq_length), config) * gaussian_mat * 255
    data_test = expand_matrices(resize_batch_time_series(data_test, config.seq_length), config) * gaussian_mat * 255
    
    print('**********************************************************************')
    for i in tqdm(range(len(data_train))):
        img = Image.fromarray(data_train[i].astype(np.uint8), mode='L')
        img = img.convert('RGB')
        img.save(os.path.join(train_dir, str(label_train[i]), f'{i}.JPEG'))
    print(f'Processing {name} training dataset done!')

    for i in tqdm(range(len(data_test))):
        img = Image.fromarray(data_test[i].astype(np.uint8), mode='L')
        img = img.convert('RGB')
        img.save(os.path.join(test_dir, str(label_test[i]), f'{i}.JPEG'))
    print(f'Processing {name} testing dataset done!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name-class-path", type=str, default='utils/name_class.txt')
    parser.add_argument("--data-path", type=str, default='UCRArchive_2018')
    parser.add_argument("--results-dir", type=str, default="data")
    parser.add_argument("--image-size", type=int, choices=[224, 384, 512], default=224)
    parser.add_argument("--patch-size", type=int, choices=[14, 16, 32], default=16)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--std", type=float, default=1)

    args = parser.parse_args()
    main(args)