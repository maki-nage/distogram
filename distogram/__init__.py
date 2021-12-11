__author__ = """Romain Picard"""
__email__ = 'romain.picard@oakbits.com'
__version__ = '2.0.0'

import math
from bisect import bisect_left
from functools import reduce
from itertools import accumulate
from operator import itemgetter
from typing import List, Optional, Tuple, Union, Any
import copy

from collections import deque

EPSILON = 1e-5
Bin = Tuple[float, int]


class Item(object):
    """ Holds a value and an object for OrderedMinMaxList.
    """

    def __init__(self, value, obj=None):
        """ Creates a new Item object

        Args:
            value: [Required] A value.
            obj: [Optional] A python object associated with the value.

        Returns:
            A Item object.
        """
        self.value: float = value
        self.obj = copy.deepcopy(obj)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __str__(self):
        return f"Item({self.value})"

    def __repr__(self):
        return f"Item({self.value}): {self.obj}"


class OrderedMinMaxList(object):
    """ Maintains an ordered fixed size min or max list.
    """
    # @TODO: This implementation should be faster than a fixed size heap for 
    #        most large datasets, but a comparison would be interesting.
    # https://stackoverflow.com/questions/30443150/maintain-a-fixed-size-heap-python

    def __init__(self, min_max_type=None, max_len=10):
        """ Creates a new OrderedMinMaxList object.

        Args:
            min_max_type: [Required] Either "min" or "max".
            max_len: [Optional] Maximum length of list.

        Returns:
            A OrderedMinMaxList object.
        """
        self.queue = deque()
        self.min_max_type: str = min_max_type
        self.max_len: int = max_len
        if self.min_max_type not in ["min", "max"]:
            raise ValueError(
                f"{self.min_max_type} is not supported. "
                f"Only 'min' and 'max' are supported.")

    def merge(self, other):
        """ Merge two OrderedMinMaxList objects.

        The merged OrderedMinMaxList will keep the `max_len` min or max 
        Items in both self and other. 

        Args:
            other: [Required] An OrderedMinMaxList object.

        Returns:
            A OrderedMinMaxList object.
        """
        if self.min_max_type != other.min_max_type:
            raise ValueError(
                f"'{self.min_max_type}' != '{other.min_max_type}'"
                f"Only OrderedMinMaxList(s) must be same type")
        a_queue = copy.deepcopy(self.queue)
        b_queue = copy.deepcopy(other.queue)
        new_queue = deque()
        new_len = min(self.max_len, other.max_len)
        new_min_max_list = OrderedMinMaxList(
            self.min_max_type, max_len=new_len)

        if self.min_max_type == "min":
            while len(new_queue) < new_len:
                if len(a_queue) == 0:
                    new_queue.append(b_queue[0])
                    b_queue.popleft()
                elif len(b_queue) == 0:
                    new_queue.append(a_queue[0])
                    a_queue.popleft()
                else:
                    a = a_queue[0]
                    b = b_queue[0]
                    if a <= b:
                        new_queue.append(a)
                        a_queue.popleft()
                    else:
                        new_queue.append(b)
                        b_queue.popleft()

        elif self.min_max_type == "max":
            while len(new_queue) < new_len:
                if len(a_queue) == 0:
                    new_queue.appendleft(b_queue[-1])
                    b_queue.pop()
                elif len(b_queue) == 0:
                    new_queue.appendleft(a_queue[-1])
                    a_queue.pop()
                else:
                    a = a_queue[-1]
                    b = b_queue[-1]
                    if a >= b:
                        new_queue.appendleft(a)
                        a_queue.pop()
                    else:
                        new_queue.appendleft(b)
                        b_queue.pop()

        new_min_max_list.queue = new_queue
        return new_min_max_list

    def add(self, value, obj=None):
        """ Add a value and optional object to a OrderedMinMaxList.

        Args:
            value: [Required] value to be added.
            obj: [Optional] object associated with value.

        Returns:
            None.
        """
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{value} can not be converted into a float")
        item = Item(value, obj)
        # unchanged for min/max
        if len(self.queue) < self.max_len:
            if len(self.queue) == 0:
                self.queue.append(item)
            elif item < self.queue[0]:
                self.queue.appendleft(item)
            elif item > self.queue[-1]:
                self.queue.append(item)
            else:
                self.queue.append(item)
                self.queue = deque(sorted(list(self.queue)))
        else:
            if self.min_max_type == "min":
                if item <= self.queue[0]:
                    self.queue.appendleft(item)
                    self.queue = deque(list(self.queue)[:self.max_len])
                elif item > self.queue[-1]:
                    pass
                else:
                    self.queue.append(item)
                    self.queue = deque(sorted(list(self.queue))[:self.max_len])

            elif self.min_max_type == "max":
                if item >= self.queue[-1]:
                    self.queue.append(item)
                    self.queue = deque(list(self.queue)[-self.max_len:])
                elif item < self.queue[0]:
                    pass
                else:
                    self.queue.append(item)
                    self.queue = (
                        deque(sorted(list(self.queue))[-self.max_len:]))


# bins is a tuple of (cut point, count)
class Distogram(object):
    """ Compressed representation of a distribution.
    """
    __slots__ = (
        'bin_count', 'bins', 'min', 'max', 'diffs', 'min_diff', 
        'weighted_diff', '_min_list', '_max_list',  
        'with_min_max_list', 'min_max_list_size')

    def __init__(
        self, 
        bin_count: int = 100, 
        weighted_diff: bool = False, 
        sample_size: int = 0,
        with_min_max_list: bool = False,
        min_max_list_size: int = 10,
    ):
        """ Creates a new Distogram object

        Args:
            bin_count: [Optional] the number of bins to use.
            weighted_diff: [Optional] Whether to use weighted bin sizes.
            with_min_max_list: [Optional] Whether to maintain a list of minimum and maximum values
            min_max_list_size: [Optional] How many min/max bin samples to keep.

        Returns:
            A Distogram object.
        """
        self.bin_count: int = bin_count
        self.bins: List[Bin] = list()
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self._min_list: Optional[List[object]] = list()
        self._max_list: Optional[List[object]] = list()
        self.diffs: Optional[List[float]] = None
        self.min_diff: Optional[float] = None
        self.weighted_diff: bool = weighted_diff
        self.with_min_max_list: bool = with_min_max_list
        self.min_max_list_size: int = min_max_list_size
        if self.with_min_max_list:
            self._min_list = OrderedMinMaxList("min", self.min_max_list_size)
            self._max_list = OrderedMinMaxList("max", self.min_max_list_size)


def _linspace(start: float, stop: float, num: int) -> List[float]:
    if num == 1:
        return [stop]
    step = (stop - start) / float(num)
    values = [start + step * i for i in range(num + 1)]
    return values


def _moment(x: List[float], counts: List[float], c: float, n: int) -> float:
    m = (ci * (v - c) ** n for i, (ci, v) in enumerate(zip(counts, x)))
    return sum(m) / sum(counts)


def _weighted_diff(h: Distogram, left: Bin, right: Bin):
    diff = left[0] - right[0]
    if h.weighted_diff is True:
        diff *= math.log(EPSILON + min(left[1], right[1]))
    return diff


def _update_diffs(h: Distogram, i: int) -> None:
    if h.diffs is not None:
        update_min = False

        if i > 0:
            if h.diffs[i - 1] == h.min_diff:
                update_min = True

            h.diffs[i - 1] = _weighted_diff(h, h.bins[i], h.bins[i - 1])
            if h.diffs[i - 1] < h.min_diff:
                h.min_diff = h.diffs[i - 1]

        if i < len(h.bins) - 1:
            if h.diffs[i] == h.min_diff:
                update_min = True

            h.diffs[i] = _weighted_diff(h, h.bins[i + 1], h.bins[i])
            if h.diffs[i] < h.min_diff:
                h.min_diff = h.diffs[i]

        if update_min is True:
            h.min_diff = min(h.diffs)

    return


def _trim(h: Distogram) -> Distogram:
    while len(h.bins) > h.bin_count:
        if h.diffs is not None:
            i = h.diffs.index(h.min_diff)
        else:
            diffs = [
                (i - 1, _weighted_diff(h, b, h.bins[i - 1]))
                for i, b in enumerate(h.bins[1:], start=1)
            ]
            i, _ = min(diffs, key=itemgetter(1))

        v1, f1 = h.bins[i]
        v2, f2 = h.bins[i + 1]
        h.bins[i] = (v1 * f1 + v2 * f2) / (f1 + f2), f1 + f2
        del h.bins[i + 1]

        if h.diffs is not None:
            del h.diffs[i]
            _update_diffs(h, i)
            h.min_diff = min(h.diffs)

    return h


def _trim_in_place(h: Distogram, value: float, c: int, i: int):
    v, f = h.bins[i]
    h.bins[i] = (v * f + value * c) / (f + c), f + c
    _update_diffs(h, i)
    return h


def _compute_diffs(h: Distogram) -> List[float]:
    if h.weighted_diff is True:
        diffs = [
            (v2 - v1) * math.log(EPSILON + min(f1, f2))
            for (v1, f1), (v2, f2) in zip(h.bins[:-1], h.bins[1:])
        ]
    else:
        diffs = [v2 - v1 for (v1, _), (v2, _) in zip(h.bins[:-1], h.bins[1:])]
    h.min_diff = min(diffs)

    return diffs


def _search_in_place_index(h: Distogram, new_value: float, index: int) -> int:
    if h.diffs is None:
        h.diffs = _compute_diffs(h)

    if index > 0:
        diff1 = _weighted_diff(h, (new_value, 1), h.bins[index - 1])
        diff2 = _weighted_diff(h, h.bins[index], (new_value, 1))

        i_bin, diff = (index - 1, diff1) if diff1 < diff2 else (index, diff2)

        return i_bin if diff < h.min_diff else -1

    return -1


def update(
        h: Distogram, value: float, count: int = 1, obj=None) -> Distogram:
    """ Adds a new element to the distribution.

    Args:
        h: A Distogram object.
        value: The value to add on the histogram.
        count: [Optional] The number of times that value must be added.

    Returns:
        A Distogram object where value as been processed.

    Raises:
        ValueError if count is not strictly positive.
    """
    if count <= 0:
        raise ValueError("count must be strictly positive")

    if h.with_min_max_list:
        h._min_list.add(value, obj)
        h._max_list.add(value, obj)

    index = 0
    if len(h.bins) > 0:
        if value <= h.bins[0][0]:
            index = 0
        elif value >= h.bins[-1][0]:
            index = -1
        else:
            index = bisect_left(h.bins, (value, 1))

        vi, fi = h.bins[index]
        if vi == value:
            h.bins[index] = (vi, fi + count)
            return h

    if index > 0 and len(h.bins) >= h.bin_count:
        in_place_index = _search_in_place_index(h, value, index)
        if in_place_index > 0:
            h = _trim_in_place(h, value, count, in_place_index)
            return h

    if index == -1:
        h.bins.append((value, count))
        if h.diffs is not None:
            diff = _weighted_diff(h, h.bins[-1], h.bins[-2])
            h.diffs.append(diff)
            h.min_diff = min(h.min_diff, diff)
    else:
        h.bins.insert(index, (value, count))
        if h.diffs is not None:
            h.diffs.insert(index, 0)
            _update_diffs(h, index)

    if (h.min is None) or (h.min > value):
        h.min = value
    if (h.max is None) or (h.max < value):
        h.max = value

    return _trim(h)


def merge(h1: Distogram, h2: Distogram, preserve_inputs: bool = False) -> Distogram:
    """ Merges two Distogram objects

    Args:
        h1: First Distogram.
        h2: Second Distogram.
        copy: Do not modify h1 or h2. 

    Returns:
        A Distogram object being the composition of h1 and h2. The number of
        bins in this Distogram is equal to the number of bins in h1.
    """
    if preserve_inputs or (h1.with_min_max_list and h2.with_min_max_list):
        h1c = copy.deepcopy(h1)
        h2c = copy.deepcopy(h2)

        h = reduce(
            lambda residual, b: update(residual, *b),
            h2c.bins,
            h1c,
        )
        if h1.with_min_max_list and h2.with_min_max_list:
            h._min_list = h1._min_list.merge(h2._min_list)
            h._max_list = h1._max_list.merge(h2._max_list)
    else:
        h = reduce(
            lambda residual, b: update(residual, *b),
            h2.bins,
            h1,
        )  
        if h1.with_min_max_list and h2.with_min_max_list:
            h._min_list = h1._min_list.merge(h2._min_list)
            h._max_list = h1._max_list.merge(h2._max_list)

    return h


def count_at(h: Distogram, value: float):
    """ Counts the number of elements present in the distribution up to value.

    Args:
        h: A Distogram object.
        value: The value up to what elements must be counted.

    Returns:
        An estimation of the real count, computed from the compressed
        representation of the distribution. Returns None if the Distogram
        object contains no element or value is outside of the distribution
        bounds.
    """
    if len(h.bins) == 0:
        return None

    if value < h.min or value > h.max:
        return None

    v0, f0 = h.bins[0]
    vl, fl = h.bins[-1]
    if value <= v0:  # left
        ratio = (value - h.min) / (v0 - h.min)
        result = ratio * v0 / 2
    elif value >= vl:  # right
        ratio = (value - vl) / (h.max - vl)
        result = (1 + ratio) * fl / 2
        result += sum((f for _, f in h.bins[:-1]))
    else:
        i = sum(((value > v) for v, _ in h.bins)) - 1
        vi, fi = h.bins[i]
        vj, fj = h.bins[i + 1]

        mb = fi + (fj - fi) / (vj - vi) * (value - vi)
        result = (fi + mb) / 2 * (value - vi) / (vj - vi)
        result += sum((f for _, f in h.bins[:i]))

        result = result + fi / 2

    return result


def count(h: Distogram) -> float:
    """ Counts the number of elements in the distribution.

    Args:
        h: A Distogram object.

    Returns:
        The number of elements in the distribution.
    """
    return sum((f for _, f in h.bins))


def bounds(h: Distogram) -> Tuple[float, float]:
    """ Returns the min and max values of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        A tuple containing the minimum and maximum values of the distribution.
    """
    return h.min, h.max


def mean(h: Distogram) -> float:
    """ Returns the mean of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        An estimation of the mean of the values in the distribution.
    """
    p, m = zip(*h.bins)
    return _moment(p, m, 0, 1)


def variance(h: Distogram) -> float:
    """ Returns the variance of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        An estimation of the variance of the values in the distribution.
    """
    p, m = zip(*h.bins)
    return _moment(p, m, mean(h), 2)


def stddev(h: Distogram) -> float:
    """ Returns the standard deviation of the distribution.

    Args:
        h: A Distogram object.

    Returns:
        An estimation of the standard deviation of the values in the
        distribution.
    """
    return math.sqrt(variance(h))


def histogram(h: Distogram, bin_count: int = 100) -> List[Tuple[float, float]]:
    """ Returns a histogram of the distribution

    Args:
        h: A Distogram object.
        bin_count: [Optional] The number of bins in the histogram.

    Returns:
        An estimation of the histogram of the distribution, or None
        if there is not enough items in the distribution.
    """

    if len(h.bins) < bin_count:
        return None

    bin_bounds = _linspace(h.min, h.max, num=(bin_count + 2))
    counts = [count_at(h, e) for e in bin_bounds[1:-1]]
    u = [
        (b, new - last)
        for b, new, last in zip(bin_bounds[1:], counts[1:], counts[:-1])
    ]

    return u


def quantile(h: Distogram, value: float) -> Optional[float]:
    """ Returns a quantile of the distribution

    Args:
        h: A Distogram object.
        value: The quantile to compute. Must be between 0 and 1

    Returns:
        An estimation of the quantile. Returns None if the Distogram
        object contains no element or value is outside of [0, 1].
    """
    if len(h.bins) == 0:
        return None

    if not (0 <= value <= 1):
        return None

    total_count = count(h)
    q_count = int(total_count * value)
    v0, f0 = h.bins[0]
    vl, fl = h.bins[-1]

    if q_count <= (f0 / 2):  # left values
        fraction = q_count / (f0 / 2)
        result = h.min + (fraction * (v0 - h.min))

    elif q_count >= (total_count - (fl / 2)):  # right values
        base = q_count - (total_count - (fl / 2))
        fraction = base / (fl / 2)
        result = vl + (fraction * (h.max - vl))

    else:
        mb = q_count - f0 / 2
        mids = [
            (fi + fj) / 2 for (_, fi), (_, fj) in zip(h.bins[:-1], h.bins[1:])]
        i, _ = (
            next(filter(lambda i_f: mb < i_f[1], enumerate(accumulate(mids)))))

        (vi, _), (vj, _) = h.bins[i], h.bins[i + 1]
        fraction = (mb - sum(mids[:i])) / mids[i]
        result = vi + (fraction * (vj - vi))

    return result


def _min_max_list(
    h: Distogram, 
    with_objects: bool = False,
    min_max_type: str = None) -> Union[List[float],
                                       List[Tuple[float, Any]]]:

    if min_max_type not in ["min", "max"]:
        raise ValueError(
            f"{min_max_type} is not supported. "
            f"Only 'min' and 'max' are supported.")

    if not h.with_min_max_list:
        if with_objects:
            if min_max_type == "min":
                return [(h.min, None)]
            else:
                return [(h.max, None)]
        else:
            if min_max_type == "min":
                return [h.min]
            else:
                return [h.max]   
            
    if min_max_type == "min":
        result = list(h._min_list.queue)
    elif min_max_type == "max":
        result = list(h._max_list.queue)  
    if with_objects:
        result = [(item.value, item.obj) for item in result]
    else:
        result = [item.value for item in result]
    return result


def min_list(
    h: Distogram, 
    with_objects: bool = False) -> Union[List[float],
                                         List[Tuple[float, Any]]]:
    """ Returns list of minimum values and, optionally, associated objects

    Args:
        h: A Distogram object.
        with_objects: [Optional] If True, returns associated objects

    Returns:
        A list of minimum values or a list of tuples of minimum values
        and associated objects, if present in the Distogram. 
    """
    return _min_max_list(h, min_max_type="min", with_objects=with_objects)


def max_list(
    h: Distogram, 
    with_objects: bool = False) -> Union[List[float],
                                         List[Tuple[float, Any]]]:
    """ Returns list of maximum values and, optionally, associated objects

    Args:
        h: A Distogram object.
        with_objects: [Optional] If True, returns associated objects

    Returns:
        A list of maximum values or a list of tuples of minimum values
        and associated objects, if present in the Distogram. 
    """
    return _min_max_list(h, min_max_type="max", with_objects=with_objects)
