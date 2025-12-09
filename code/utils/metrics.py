import torch

def calculate_accuracy(outputs, labels, threshold=0.5):
    """
    Computes accuracy for binary classification.
    """
    predicted = (outputs > threshold).float()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total