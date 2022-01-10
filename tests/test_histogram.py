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

