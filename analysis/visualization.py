import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_raw_eeg(data, fs, n_channels=5, duration=5, title="Raw EEG"):
    time = np.arange(data.shape[1]) / fs
    plt.figure(figsize=(12, 6))
    for i in range(min(n_channels, data.shape[0])):
        plt.plot(time[:int(duration*fs)], data[i, :int(duration*fs)] + i*100)
    plt.xlabel("Time (s)")
    plt.ylabel("Channels")
    plt.title(title)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_psd(freqs, psd, title="Power Spectral Density"):
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd.mean(axis=0))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_training_history(history, title="Training History"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Train Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
