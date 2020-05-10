import pytest
from pytest import approx
import random
import distogram


def test_count_at():
    h = distogram.Distogram(bin_count=3)
    print(h)

    # fill histogram
    h = distogram.update(h, 16, count=4)
    h = distogram.update(h, 23, count=3)
    h = distogram.update(h, 28, count=5)
    print(h)

    actual_result = distogram.count_at(h, 25)
    assert actual_result == approx(6.859999999)


def test_count_at_normal():
    points = 10000
    normal = [random.normalvariate(0.0, 1.0) for _ in range(points)]
    h = distogram.Distogram()

    for i in normal:
        h = distogram.update(h, i)

    assert distogram.count_at(h, 0) == approx(points/2, rel=0.025)


def test_count_at_not_enough_elements():
    h = distogram.Distogram()

    h = distogram.update(h, 1)
    h = distogram.update(h, 2)
    h = distogram.update(h, 3)

    with pytest.raises(ValueError):
        h = distogram.count_at(h, 2.5)
