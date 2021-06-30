import distogram


def test_count():
    h = distogram.Distogram(bin_count=3)
    assert distogram.count(h) == 0

    h = distogram.update(h, 16, count=4)
    assert distogram.count(h) == 4
    h = distogram.update(h, 23, count=3)
    assert distogram.count(h) == 7
    h = distogram.update(h, 28, count=5)
    assert distogram.count(h) == 12
