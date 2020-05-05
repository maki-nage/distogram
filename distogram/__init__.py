from collections import namedtuple
#from typing import Any, List, Tuple, NamedTuple

__author__ = """Romain Picard"""
__email__ = 'romain.picard@oakbits.com'
__version__ = '0.0.0'


'''
class Distogram(NamedTuple):    
    bin_count: int = 200
    bins: List[Tuple] = []
'''

#Distogram = namedtuple('Distogram', ['bin_count', 'bins'])
#Distogram.__new__.__defaults__ = (200, [],)


class Distogram(object):
    __slots__ = 'bin_count', 'bins'

    def __init__(self, bin_count=200, bins=[]):
        self.bin_count = bin_count
        self.bins = bins


def trim(h):
    bins = h.bins
    while len(bins) > h.bin_count:
        min_diff = None
        i = None
        for index, value in enumerate(h.bins):
            if index > 0:
                diff = value[0] - prev_value
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    i = index - 1
            prev_value = value[0]

        bins[i] = (
            (bins[i][0]*bins[i][1] + bins[i+1][0]*bins[i+1][1]) / (bins[i][1] + bins[i+1][1]),
            bins[i][1] + bins[i+1][1]
        )
        del bins[i+1]

    h.bins = bins
    return h


def update(h, value, count=1):
    bins = h.bins
    for index, bin in enumerate(bins):
        if bin[0] == value:
            print("same: {}, {}".format(bin[0], value))
            bin = (bin[0], bin[1]+count)
            h.bins[index] = bin
            return h

    bins.append((value, count))
    bins = sorted(bins, key=lambda i: i[0])
    h.bins = bins
    return trim(h)


def merge(h1, h2):
    h = h1
    for i in h2:
        h = update(h, i[0], i[1])
    return h


def sum(h, value):
    bins = h.bins
    i = -1
    while value > bins[i+1][0] and i < len(bins) - 1:
        i += 1

    mb = bins[i][1] + (bins[i+1][1] - bins[i][1]) / (bins[i+1][0] - bins[i][0]) * (value - bins[i][0])
    s = (bins[i][1] + mb) / 2 * (value - bins[i][0]) / (bins[i+1][0] - bins[i][0])
    for j in range(i):
        s = s + bins[j][1]

    s = s + bins[i][1] / 2
    return s


def count(h):
    '''Returns the number of items in the histogram
    '''
    sum = 0
    for i in h.bins:
        sum += i[1]

    return sum


def quantile(h, value):
    total_count = count(h)
    q_count = int(total_count * value)
    print("total: {}, q: {}".format(total_count, q_count))

    # transpose histogram to compute quantile
    pt = []
    acc = 0
    for i in h.bins:
        acc += i[1]
        pt.append(acc)

    mt = []
    prev = 0
    for i in h.bins:
        mt.append(i[0] - prev)
        prev = i[0]

    bins = [(pt[i], mt[i]) for i in range(len(mt))]
    print(bins)
    ht = Distogram(bin_count=h.bin_count, bins=bins)
    q = sum(ht, q_count+1)
    return q
