import pytest
from src.generate_RTS import generate_RTS, generate_gaussian_noise
import numpy as np

def test_RTS_num_samples_100():
    
    rts = generate_RTS(
        num_samples=100,
    )

    assert len(rts) == 100

def test_RTS_num_samples_1000():
        
    rts = generate_RTS(
        num_samples=1000,
    )

    assert len(rts) == 1000

def test_RTS_seed():
    
    rts_1 = generate_RTS(
        num_samples=1000,
        seed=1,
    )
    
    rts_2 = generate_RTS(
        num_samples=1000,
        seed=1,
    )
    
    assert rts_1 == rts_2


#! I know this is not working rn
def test_RTS_num_states_2():
    pass

def test_RTS_num_states_3():
    pass

def test_RTS_transition_probs_2x2():
    
    rts = generate_RTS(
        num_samples=100_000,
    )

    state_one_count = len([x for x in rts if x > 0])
    state_zero_count = len([x for x in rts if x < 1])

    # they should roughly be equal
    assert state_one_count == pytest.approx(state_zero_count, rel=5000)

def test_RTS_transition_probs_more_complex():
    pass

def test_generate_gaussian_noise_basic():
    
    noise = generate_gaussian_noise(
        num_samples=10_000,
        mean=0,
        std=1,
    )

    mean = np.mean(noise)
    std = np.std(noise)

    assert mean == pytest.approx(0, abs=0.1)
    assert std == pytest.approx(1, abs=0.1)

def test_generate_gaussian_noise_seed():
    
    noise_1 = generate_gaussian_noise(
        num_samples=10_000,
        mean=3.1415,
        std=3,
        seed=1,
    )
    
    noise_2 = generate_gaussian_noise(
        num_samples=10_000,
        mean=3.1415,
        std=3,
        seed=1,
    )
    
    np.testing.assert_array_equal(noise_1, noise_2)
    