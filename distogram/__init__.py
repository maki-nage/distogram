import math
from functools import reduce

__author__ = """Romain Picard"""
__email__ = 'romain.picard@oakbits.com'
__version__ = '1.6.0'


# bins is a tuple of (cut point, count)
class Distogram(object):
    '''Compressed representation of a distribution
    '''
    __slots__ = 'bin_count', 'bins', 'min', 'max', 'diffs', 'min_diff', 'weighted_diff'

    def __init__(self, bin_count=100, weighted_diff=False):
        '''Creates a new Distogram object

        Args:
            bin_count: [Optional] the number of bins to use.

        Returns:
            A Distogram object.
        '''
        self.bin_count = bin_count
        self.bins = []
        self.min = None
        self.max = None
        self.diffs = None
        self.min_diff = None
        self.weighted_diff = weighted_diff


def _linspace(start, stop, num):
    if num == 1:
        return stop
    h = (stop - start) / float(num)
    values = [start + h * i for i in range(num+1)]
    return values


def _moment(x, counts, c, n):
    m = []
    for i in range(len(x)):
        m.append(counts[i]*(x[i]-c)**n)
    return sum(m) / sum(counts)


def _weighted_diff(h, l, r):
    diff = l[0] - r[0]
    if h.weighted_diff is True:
        diff *= math.log(0.00001 + abs(l[1] - r[1]))
    return diff


def _update_diffs(h, i):
    if h.diffs is not None:
        update_min = False
        if i > 0:
            diff = _weighted_diff(h, h.bins[i], h.bins[i-1])
            if h.diffs[i-1] == h.min_diff:
                h.diffs[i-1] = diff
                update_min = True
            else:
                h.diffs[i-1] = diff
                if h.diffs[i-1] < h.min_diff:
                    h.min_diff = h.diffs[i-1]
        if i < len(h.bins) - 1:
            diff = _weighted_diff(h, h.bins[i+1], h.bins[i])
            if h.diffs[i] == h.min_diff:
                h.diffs[i] = diff
                update_min = True
            else:
                h.diffs[i] = diff
                if h.diffs[i] < h.min_diff:
                    h.min_diff = h.diffs[i]

        if update_min is True:
            h.min_diff = min(h.diffs)


def _trim(h, index):
    bins = h.bins
    while len(bins) > h.bin_count:
        if h.diffs is not None:
            min_diff = h.min_diff
            i = h.diffs.index(min_diff)
        else:
            min_diff = None
            i = None
            for index, value in enumerate(h.bins):
                if index > 0:
                    diff = _weighted_diff(h, h.bins[index], h.bins[index-1])
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        i = index - 1

        bins[i] = (
            (bins[i][0]*bins[i][1] + bins[i+1][0]*bins[i+1][1])
            / (bins[i][1] + bins[i+1][1]),
            bins[i][1] + bins[i+1][1]
        )
        del bins[i+1]
        if h.diffs is not None:
            del h.diffs[i]
            _update_diffs(h, i)
            h.min_diff = min(h.diffs)

    h.bins = bins
    return h


def _trim_in_place(h, value, count, i):
    bins = h.bins

    bins[i] = (
        (bins[i][0]*bins[i][1] + value*count)
        / (bins[i][1] + count),
        bins[i][1] + count
    )

    h.bins = bins
    _update_diffs(h, i)
    return h


def _compute_diffs(h):
    diffs = []
    bins = h.bins
    h.min_diff = None
    for index in range(1, len(bins)):
        diff = _weighted_diff(h, h.bins[index], h.bins[index-1])
        diffs.append(diff)
        if h.min_diff is None or diff < h.min_diff:
            h.min_diff = diff

    return diffs


def _linear_index(bins, value):
    ratio = float((value - bins[0][0]) / (bins[-1][0] - bins[0][0]))
    index = round(ratio * len(bins))
    if index > 0 and index < (len(bins) - 1):
        if value > bins[index][0] and value < bins[index+1][0]:
            return index+1
        if value > bins[index-1][0] and value < bins[index][0]:
            return index

    return 0


def _bisect_left(a, x):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid][0] < x: lo = mid+1
        else: hi = mid
    return lo


def _search_in_place_index(h, new_value, index):
    bins = h.bins

    if h.diffs is None:
        h.diffs = _compute_diffs(h)

    min_diff = h.min_diff
    i_bin = None

    if index > 0:
        diff1 = _weighted_diff(h, (new_value, 1), h.bins[index-1])
        diff2 = _weighted_diff(h, h.bins[index], (new_value, 1))
        if diff1 < diff2:
            diff = diff1
            i_bin = index-1
        else:
            diff = diff2
            i_bin = index

        if diff < min_diff:
            return i_bin
        else:
            return -1
    return -1


def update(h, value, count=1):
    '''Adds a new element to the distribution.

    Args:
        h: A Distogram object.
        value: The value to add on the histogram.
        count: [Optional] The number of times that value must be added.

    Returns:
        A Distogram object where value as been processed.
    '''
    bins = h.bins
    index = 0
    if len(bins) > 0:
        if value <= bins[0][0]:
            index = 0
        elif value >= bins[-1][0]:
            index = -1
        else:
            if len(bins) >= h.bin_count:
                index = _linear_index(bins, value)
            if index == 0:
                index = _bisect_left(bins, value)

    if index > 0 and len(bins) >= h.bin_count:
        in_place_index = _search_in_place_index(h, value, index)
        if in_place_index > 0:
            h = _trim_in_place(h, value, count, in_place_index)
            return h

    if len(bins) > 0:
        bin = bins[index]
        if bin[0] == value:
            bin = (bin[0], bin[1]+count)
            h.bins[index] = bin
            return h

    if index == -1:
        bins.append((value, count))
        if h.diffs is not None:
            diff = _weighted_diff(h, h.bins[-1], h.bins[-2])
            h.diffs.append(diff)
            if diff < h.min_diff:
                h.min_diff = diff
    else:
        bins.insert(index, (value, count))
        if h.diffs is not None:
            h.diffs.insert(index, None)
            _update_diffs(h, index)

    h.bins = bins
    if h.min is None or h.min > value:
        h.min = value
    if h.max is None or h.max < value:
        h.max = value
    return _trim(h, index)


def merge(h1, h2):
    '''Merges two Distogram objects

    Args:
        h1: First Distogram.
        h2: Second Distogram.

    Returns:
        A Distogram object being the composition of h1 and h2. The number of
        bins in this Distogram is equal to the number of bins in h1.
    '''
    h = h1
    for i in h2.bins:
        h = update(h, i[0], i[1])
    return h


def count_at(h, value):
    '''Counts the number of elements present in the distribution up to value.

    Args:
        h: A Distogram object.
        value: The value up to what elements must be counted.

    Returns:
        An estimation of the real count, computed from the compressed
        representation of the distribution. Returns None if the Distogram
        object contains no element or value is outside of the distribution
        bounds.
    '''
    if len(h.bins) == 0:
        return None

    if value < h.min or value > h.max:
        return None

    bins = h.bins
    if value <= bins[0][0]:  # left
        ratio = (value - h.min) / (bins[0][0] - h.min)
        s = ratio * bins[0][0] / 2
    elif value >= bins[-1][0]:  # right
        ratio = (value - bins[-1][0]) / (h.max - bins[-1][0])
        s = ratio * (bins[-1][1]) / 2
        s += bins[-1][1] / 2
        for j in range(len(bins) - 1):
            s += bins[j][1]
    else:
        i = -1
        while value > bins[i+1][0] and i < len(bins) - 1:
            i += 1

        mb = bins[i][1] + (bins[i+1][1] - bins[i][1]) \
            / (bins[i+1][0] - bins[i][0]) * (value - bins[i][0])
        s = (bins[i][1] + mb) / 2 * (value - bins[i][0]) \
            / (bins[i+1][0] - bins[i][0])
        for j in range(i):
            s = s + bins[j][1]

        s = s + bins[i][1] / 2
    return s


def count(h):
    '''Counts the number of elements in the distribution.

    Args:
        h: A Distogram object.

    Returns:
        The number of elements in the distribution.
    '''
    return reduce(lambda acc, i: acc + i[1], h.bins, 0)


def bounds(h):
    '''Returns the min and max values of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        A tuple containing the minimum and maximum values of the distribution.
    '''
    return (h.min, h.max)


def mean(h):
    '''Returns the mean of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        An estimation of the mean of the values in the distribution.
    '''
    p = [i[0] for i in h.bins]
    m = [i[1] for i in h.bins]
    return _moment(p, m, 0, 1)


def variance(h):
    '''Returns the variance of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        An estimation of the variance of the values in the distribution.
    '''
    p = [i[0] for i in h.bins]
    m = [i[1] for i in h.bins]
    return _moment(p, m, mean(h), 2)


def stddev(h):
    '''Returns the standard deviation of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        An estimation of the standard deviation of the values in the
        distribution.
    '''
    return math.sqrt(variance(h))


def histogram(h, ucount=100):
    '''Returns a histogram of the distribution

    Args:
        h: A Distogram object.
        ucount: [Optional] The number of bins in the histogram.

    Returns:
        An estimation of the histogram of the distribution.

    Raises:
        ValueError if distribution contains less elements than the number of
        bins in the Distogram object.
    '''
    if len(h.bins) < h.bin_count:
        raise ValueError("not enough elements in distribution")

    last = 0.0
    u = []
    bounds = _linspace(h.min, h.max, num=ucount+1)
    for e in bounds[1:-1]:
        new = count_at(h, e)
        u.append((e, new-last))
        last = new
    return u


def quantile(h, value):
    '''Returns a quantile of the distribution

    Args:
        h: A Distogram object.
        value: The quantile to compute. Must be between 0 and 1

    Returns:
        An estimation of the quantile. Returns None if the Distogram
        object contains no element or value is outside of [0:1].
    '''
    if len(h.bins) == 0:
        return None

    if value < 0.0 or value > 1.0:
        return None

    total_count = count(h)
    q_count = int(total_count * value)
    bins = h.bins
    i = 0
    if q_count <= (bins[0][1] / 2):  # left values
        print("left")
        ratio = q_count / (bins[0][1] / 2)
        value = h.min + (ratio * (bins[0][0] - h.min))
    elif q_count >= (total_count - (bins[-1][1] / 2)):  # right values
        base = q_count - (total_count - (bins[-1][1] / 2))
        ratio = base / (bins[-1][1] / 2)
        value = bins[-1][0] + (ratio * (h.max - bins[-1][0]))
        print("right value for quantile base: {}, ratio: {}".format(
            base, ratio))
    else:
        mb = q_count - bins[0][1] / 2
        while mb - (bins[i][1] + bins[i+1][1]) / 2 > 0 and i < len(bins) - 1:
            mb -= (bins[i][1] + bins[i+1][1]) / 2
            i += 1

        ratio = mb / ((bins[i][1] + bins[i+1][1]) / 2)
        value = bins[i][0] + (ratio * (bins[i+1][0] - bins[i][0]))
    return value
