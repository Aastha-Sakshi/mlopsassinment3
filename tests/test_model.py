import torch

from src.models.simple_cnn import SimpleCNN
from src.training.engine import predict_probabilities


def test_simple_cnn_output_shape():
    model = SimpleCNN()
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 2)


def test_prediction_probabilities_sum_to_one():
    probabilities = predict_probabilities(SimpleCNN(), torch.zeros(1, 3, 224, 224), torch.device("cpu"))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
