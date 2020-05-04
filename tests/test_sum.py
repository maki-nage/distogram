from pytest import approx
import distogram


def test_sum():
    h = distogram.Distogram(bin_count=3)
    print(h)

    # fill histogram
    h = distogram.update(h, 16, count=4)
    h = distogram.update(h, 23, count=3)
    h = distogram.update(h, 28, count=5)
    print(h)

    actual_result = distogram.sum(h, 25)
    assert actual_result == approx(6.316326)
