import pytest
from src.generate_RTS_data import generate_data, generate_classification_data
import numpy as np

def test_data_is_saved():
    generate_data()
    data = np.load('./src/data/autoencoder/data.npy')
    assert data.shape == (10000, 1000, 1) 