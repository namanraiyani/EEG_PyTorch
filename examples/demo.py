import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from eeg_pytorch.data.DataDownloader import DataDownloader
from eeg_pytorch.data.DataLoader import EEGDataset
from eeg_pytorch.models.CNN import CNN
from eeg_pytorch.models.Transformer import Transformer
from eeg_pytorch.models.LSTM import LSTM
from eeg_pytorch.models.RNN import RNN
from eeg_pytorch.models.GCN import GCN
from eeg_pytorch.models.lib_for_GCN import coarsening, graph
from eeg_pytorch.analysis import features, visualization

def main():
    print("Starting EEG-PyTorch Framework Demo...")
    
    save_path = './eeg_data'
    downloader = DataDownloader(save_path)
    
    subject_ids = [1]
    runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    print(f"Downloading data for Subject {subject_ids}, Runs {runs}...")
    data_paths = downloader.download_physionet_motor_imagery(subject_ids=subject_ids, runs=runs)
    
    edf_files = data_paths[1]
    print(f"Downloaded files: {edf_files}")
    
    print("Loading data using EEGDataset (MNE)...")
    dataset = EEGDataset(data_path=edf_files, file_type='edf', target_shape=(64, 64))
    
    print(f"Data shape: {dataset.data.shape}")
    print(f"Labels shape: {dataset.labels.shape}")
    
    print("Visualizing Raw Data...")
    visualization.plot_raw_eeg(dataset.data[0].reshape(64, 64), fs=160, title="Sample EEG Epoch")
    
    print("Extracting Features...")
    sample_data = dataset.data[0].reshape(64, 64)
    feats = features.extract_features(sample_data, fs=160)
    print("Extracted Features:", feats.keys())
    
    print("Visualizing PSD...")
    freqs, psd = features.compute_psd(sample_data, fs=160)
    visualization.plot_psd(freqs, psd, title="Sample PSD")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = torch.from_numpy(dataset.data)
    labels = torch.from_numpy(dataset.labels)
    
    if labels.max() >= 4:
        labels = labels % 4
        
    dataset_torch = TensorDataset(data, labels)
    train_size = int(0.8 * len(dataset_torch))
    test_size = len(dataset_torch) - train_size
    train_dataset, test_dataset = random_split(dataset_torch, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    
    models = {
        'CNN': CNN(num_classes=4),
        'Transformer': Transformer(maxlen=3, embed_dim=97, num_classes=4),
        'LSTM': LSTM(n_input=64, lstm_size=128, num_classes=4),
        'RNN': RNN(n_input=64, rnn_size=128, num_classes=4)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        history = {'loss': [], 'acc': []}
        
        for epoch in range(5):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if name == 'CNN':
                    pass
                elif name in ['LSTM', 'RNN']:
                    inputs = inputs.view(-1, 64, 64)
                elif name == 'Transformer':
                    inputs = inputs[:, :291]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            epoch_loss = total_loss/len(train_loader)
            epoch_acc = 100*correct/total
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)
            
        visualization.plot_training_history(history, title=f"{name} Training History")
            
        print(f"Evaluating {name}...")
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if name == 'CNN':
                    pass
                elif name in ['LSTM', 'RNN']:
                    inputs = inputs.view(-1, 64, 64)
                elif name == 'Transformer':
                    inputs = inputs[:, :291]
                    
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        acc = 100 * correct / total
        print(f"{name} Test Accuracy: {acc:.2f}%")
        results[name] = acc
        
        visualization.plot_confusion_matrix(all_targets, all_preds, classes=[0, 1, 2, 3], title=f"{name} Confusion Matrix")
        
    print("\nTraining GCN...")
    
    N = 64
    W = np.random.rand(N, N)
    W = (W + W.T) / 2
    W[W < 0.8] = 0
    W = scipy.sparse.csr_matrix(W)
    
    graphs, perm = coarsening.coarsen(W, levels=5, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    
    F = [16, 32, 64, 128, 256, 512]
    K = [2, 2, 2, 2, 2, 2]
    p = [2, 2, 2, 2, 2, 2]
    M = [4]
    
    M_0 = L[0].shape[0]
    print(f"Original nodes: {N}, Coarsened graph nodes (layer 0): {M_0}")
    
    model_gcn = GCN(L, F, K, p, M, num_classes=4).to(device)
    optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=1e-3)
    
    history_gcn = {'loss': [], 'acc': []}
    
    for epoch in range(5):
        model_gcn.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs_np = inputs.numpy()
            inputs_reshaped = inputs_np.reshape(-1, 64, 64)
            
            inputs_nodes = inputs_reshaped.mean(axis=2)
            
            if M_0 > N:
                pad_width = ((0, 0), (0, M_0 - N))
                inputs_nodes = np.pad(inputs_nodes, pad_width, mode='constant')
            
            inputs_t = torch.from_numpy(inputs_nodes).float().to(device)
            targets = targets.to(device)
            
            optimizer_gcn.zero_grad()
            outputs = model_gcn(inputs_t)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_gcn.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        epoch_loss = total_loss/len(train_loader)
        epoch_acc = 100*correct/total
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        history_gcn['loss'].append(epoch_loss)
        history_gcn['acc'].append(epoch_acc)
        
    visualization.plot_training_history(history_gcn, title="GCN Training History")
        
    print("Evaluating GCN...")
    model_gcn.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs_np = inputs.numpy()
            inputs_nodes = inputs_np.reshape(-1, 64, 64).mean(axis=2)
            
            if M_0 > N:
                pad_width = ((0, 0), (0, M_0 - N))
                inputs_nodes = np.pad(inputs_nodes, pad_width, mode='constant')
                
            inputs_t = torch.from_numpy(inputs_nodes).float().to(device)
            targets = targets.to(device)
            
            outputs = model_gcn(inputs_t)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    acc = 100 * correct / total
    print(f"GCN Test Accuracy: {acc:.2f}%")
    results['GCN'] = acc
    
    visualization.plot_confusion_matrix(all_targets, all_preds, classes=[0, 1, 2, 3], title="GCN Confusion Matrix")

    print("\nResults Summary:")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")
        
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values())
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.savefig("model_comparison.png")
    print("Saved model_comparison.png")
    
    print("Demo Completed Successfully.")

if __name__ == "__main__":
    main()
