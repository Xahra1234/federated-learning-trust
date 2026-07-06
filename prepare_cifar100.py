#!/usr/bin/env python3
"""Prepare CIFAR-100 dataset for federated learning"""
import os
import numpy as np
import pickle
from pathlib import Path

def download_cifar100(data_dir="data"):
    import urllib.request
    import tarfile
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-100-python.tar.gz")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-100...")
        urllib.request.urlretrieve(url, tar_path)
    
    extract_dir = os.path.join(data_dir, "cifar-100-python")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
    
    return extract_dir

def load_cifar100_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(batch[b'fine_labels'])
    return data, labels

def create_noniid_split(labels, num_clients, alpha):
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for c_idx, c_indices in enumerate(class_indices):
        np.random.shuffle(c_indices)
        splits = (label_distribution[c_idx] * len(c_indices)).astype(int)
        splits[-1] = len(c_indices) - splits[:-1].sum()
        
        start = 0
        for client_id, split_size in enumerate(splits):
            client_indices[client_id].extend(c_indices[start:start + split_size])
            start += split_size
    
    return [np.array(indices) for indices in client_indices]

def prepare_cifar100(data_dir="data", num_clients=10, alphas=[0.5, 1.0], seed=0):
    cifar_dir = download_cifar100(data_dir)
    
    print("Loading training data...")
    train_data, train_labels = load_cifar100_batch(os.path.join(cifar_dir, "train"))
    
    print("Loading test data...")
    test_data, test_labels = load_cifar100_batch(os.path.join(cifar_dir, "test"))
    
    val_size = 5000
    val_data, val_labels = test_data[:val_size], test_labels[:val_size]
    test_data, test_labels = test_data[val_size:], test_labels[val_size:]
    
    for alpha in alphas:
        print(f"\nCreating splits for alpha={alpha}...")
        np.random.seed(seed)
        
        output_dir = Path(data_dir) / "cifar100_noniid" / f"alpha_{alpha}" / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        client_indices = create_noniid_split(train_labels, num_clients, alpha)
        
        for client_id, indices in enumerate(client_indices):
            np.savez(output_dir / f"train_{client_id}.npz", X=train_data[indices], y=train_labels[indices])
            print(f"  Client {client_id}: {len(indices)} samples")
        
        np.savez(output_dir / "val.npz", X=val_data, y=val_labels)
        np.savez(output_dir / "test.npz", X=test_data, y=test_labels)
    
    print("\n✅ CIFAR-100 dataset preparation complete!")

if __name__ == "__main__":
    prepare_cifar100()
