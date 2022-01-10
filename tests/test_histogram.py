import random
import numpy as np
from pytest import approx
import distogram
import pytest


def test_histogram():
    normal = [random.normalvariate(0.0, 1.0) for _ in range(10000)]
    h = distogram.Distogram(bins=64)

    for i in normal:
        h = distogram.update(h, i)

    np_values, np_edges = np.histogram(normal, 10)
    d_values, d_edges = distogram.histogram(h, 10)

    h = distogram.Distogram(bins=3)
    h = distogram.update(h, 23)
    h = distogram.update(h, 28)
    h = distogram.update(h, 16)
    assert(distogram.histogram(h, bins=3) ==
           (approx([1.0714285714285714, 0.6285714285714286, 1.3]),
            [16.0, 20.0, 24.0, 28]))
    assert(sum(distogram.histogram(h, bins=3)[0]) == approx(3.0))

    hist = distogram.frequency_density_distribution(h)
    integral = 0
    for density, new, old in zip(hist[0], hist[1][1:], hist[1][:-1]):
        integral += density * (new-old)

    assert(hist == approx(([0.21428571428571427, 0.3], [16.0, 23.0, 28.0])))
    assert(integral == approx(3.0))

    # how to compare histograms?
    #assert np_values == approx(d_values, abs=0.2)
    #assert np_edges == approx(d_edges, abs=0.2)


def test_histogram_on_too_small_distribution():
    h = distogram.Distogram(bins=64)

    for i in range(5):
        h = distogram.update(h, i)

    assert distogram.histogram(h, 10) == None


def test_format_histogram():
    bins = 4
    h = distogram.Distogram(bins=bins)

    for i in range(4):
        h = distogram.update(h, i)

    hist = distogram.histogram(h, bins=bins)
    assert(len(hist[1]) == len(hist[0]) + 1)

