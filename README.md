# EEG-PyTorch

EEG-PyTorch is a framework for analyzing and classifying Electroencephalography (EEG) data using PyTorch. It supports research into Motor Imagery and deep learning architectures for time-series data.

## Key Features

*   **Multi-Model Support**: Implementations of CNN, Transformer, LSTM, RNN, and GCN models.
*   **Automated Data Pipeline**: Tools to download and preprocess the PhysioNet Motor Imagery dataset.
*   **Advanced Analysis**: Utilities for feature extraction and Power Spectral Density (PSD) computation.
*   **Visualization**: Plotting functions for raw EEG signals, training history, and confusion matrices.

## Installation

Requires Python 3.8+. Install dependencies:

```bash
pip install torch numpy pandas matplotlib mne scipy
```

## Quick Start

Run the demo script to download sample data, train models, and generate reports:

```bash
python eeg_pytorch/examples/demo.py
```

## Project Structure

*   `eeg_pytorch/models/`: PyTorch model architectures.
*   `eeg_pytorch/data/`: Data downloading (`DataDownloader`) and loading (`EEGDataset`).
*   `eeg_pytorch/analysis/`: Signal processing (`features.py`) and plotting (`visualization.py`).
*   `eeg_pytorch/examples/`: Example scripts.

## Usage Examples

### Training a Specific Model

Train individual models using scripts in the `examples` directory:

```bash
# Train a CNN model
python eeg_pytorch/examples/main-CNN.py

# Train a Graph Convolutional Network
python eeg_pytorch/examples/main-GCN.py
```

### Custom Data Loading

Load custom EDF files using the `EEGDataset` class:

```python
from eeg_pytorch.data.DataLoader import EEGDataset

# Load data from EDF files
dataset = EEGDataset(data_path=['subject1.edf'], file_type='edf', target_shape=(64, 64))
print(f"Loaded data shape: {dataset.data.shape}")
```

### Visualization

Visualize data and model performance:

```python
from eeg_pytorch.analysis import visualization

# plot training history
visualization.plot_training_history(history, title="Model Training")

# plot confusion matrix
visualization.plot_confusion_matrix(targets, predictions, classes=[0, 1, 2, 3])
```

## Contributing

Contributions are welcome. Submit pull requests or open issues to improve the framework.
