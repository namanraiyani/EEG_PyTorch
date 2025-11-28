import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eeg_pytorch.data.DataLoader import DatasetLoader
from eeg_pytorch.models.Transformer import Transformer

DIR = '../EEG-DL/DatasetAPI/EEG-Motor-Movement-Imagery-Dataset/'
batch_size = 128
num_epoch = 1000
learning_rate = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    if not os.path.exists(DIR + 'training_set.csv'):
        print("Dataset not found. Please provide the dataset.")
        return

    train_loader, test_loader = DatasetLoader(DIR, batch_size)
    
    # Assuming input features is 97 based on original code
    model = Transformer(maxlen=3, embed_dim=97, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss() # Binary classification as per original code
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        
        if (epoch + 1) % 10 == 0:
            test(model, test_loader)

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train()
