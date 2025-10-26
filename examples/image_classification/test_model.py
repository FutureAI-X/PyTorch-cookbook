import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import NeuralNetwork
from data_loader import get_test_dataloader

test_dataloader = get_test_dataloader(batch_size=64)
    
model = NeuralNetwork()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
softmax = nn.Softmax(dim=1)

def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_probab = softmax(pred)
            print(f"pred: {pred_probab}, actual:{y}")

test_loop(test_dataloader, model)