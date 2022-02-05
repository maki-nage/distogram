import random
import numpy as np
from pytest import approx
import distogram
import pytest


def test_frequency_density_distribution():
    normal = [random.normalvariate(0.0, 1.0) for _ in range(10000)]
    h = distogram.Distogram(bin_count=64)

    for i in normal:
        h = distogram.update(h, i)

    h = distogram.Distogram(bin_count=3)
    h = distogram.update(h, 23)
    h = distogram.update(h, 28)
    h = distogram.update(h, 16)

    hist = distogram.frequency_density_distribution(h)
    integral = 0
    for density, new, old in zip(hist[0], hist[1][1:], hist[1][:-1]):
        integral += density * (new-old)

    assert(hist == (approx([0.21428571428571427, 0.3]), [16.0, 23.0, 28.0]))
    assert(integral == approx(3.0))
