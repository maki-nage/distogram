import random
import numpy as np
from pytest import approx
import distogram
import pytest


def test_histogram():
    normal = [random.normalvariate(0.0, 1.0) for _ in range(10000)]
    h = distogram.Distogram(bin_count=64)

    for i in normal:
        h = distogram.update(h, i)

    np_values, np_edges = np.histogram(normal, 10)
    d_edges, d_values = zip(*distogram.histogram(h, 10))

    # how to compare histograms?
    #assert np_values == approx(d_values, abs=0.2)
    #assert np_edges == approx(d_edges, abs=0.2)


def test_histogram_on_too_small_distribution():
    h = distogram.Distogram(bin_count=64)

    for i in range(5):
        h = distogram.update(h, i)

    assert distogram.histogram(h, 10) == None


def test_histogram_with_invalid_data_type():
    bin_count = 4
    h = distogram.Distogram(bin_count=bin_count)

    for i in range(5):
        h = distogram.update(h, i)

    with pytest.raises(ValueError):
        distogram.histogram(
            h, bin_count=bin_count, data_type="invalid_data_type")


def test_format_histogram():
    bin_count = 4
    h = distogram.Distogram(bin_count=bin_count)

    for i in range(4):
        h = distogram.update(h, i)

    hist = distogram.histogram(h, bin_count=bin_count)
    assert(len(hist[1]) == len(hist[0]))
    hist = distogram.histogram(h, bin_count=bin_count, data_type="numpy")
    assert(len(hist[1]) == len(hist[0]) + 1)

