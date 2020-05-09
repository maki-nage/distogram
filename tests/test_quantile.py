from pytest import approx
import distogram

import numpy as np
import random


def test_quantile():
    h = distogram.Distogram(bin_count=3)
    h.bins=[
        (16, 4), (23, 3), (28, 5)
    ]

    assert distogram.quantile(h, 0.5) == approx(23.625)


def test_normal():    
    #normal = np.random.normal(0,1, 1000)
    normal = [random.normalvariate(0.0, 1.0) for _ in range(10000)]
    h = distogram.Distogram(bin_count=64)

    for i in normal:
        distogram.update(h, i)

    assert distogram.quantile(h, 0.5) == approx(np.quantile(normal, 0.5), abs=0.2)
    assert distogram.quantile(h, 0.95) == approx(np.quantile(normal, 0.95), abs=0.2)
