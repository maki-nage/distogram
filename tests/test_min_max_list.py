import pytest
import distogram

from distogram import OrderedMinMaxList
from collections import deque

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

def test_min_max_list_interface():
    h = distogram.Distogram()
    for i in range(5):
        h = distogram.update(h, i)
    assert (
        distogram._min_max_list(h, min_max_type="min", with_objects=False)
        == [0])
    assert (
        distogram._min_max_list(h, min_max_type="max", with_objects=False)
        == [4])
    assert (
        distogram._min_max_list(h, min_max_type="min", with_objects=True)
        == [(0, None)])
    assert (
        distogram._min_max_list(h, min_max_type="max", with_objects=True)
        == [(4, None)])

    h = distogram.Distogram(bin_count=3, with_min_max_list=True, min_max_list_size=2)
    for i in range(5):
        h = distogram.update(h, i, obj=i)
    assert (
        distogram._min_max_list(h, min_max_type="min", with_objects=False)
        == [0, 1])
    assert (
        distogram._min_max_list(h, min_max_type="max", with_objects=False)
        == [3, 4])
    assert (
        distogram._min_max_list(h, min_max_type="min", with_objects=True)
        == [(0, 0), (1, 1)])
    assert (
        distogram._min_max_list(h, min_max_type="max", with_objects=True)
        == [(3, 3), (4, 4)])

    assert (
        distogram._min_max_list(h, min_max_type="max", with_objects=True)
        == distogram.max_list(h, with_objects=True))
    assert (
        distogram._min_max_list(h, min_max_type="min", with_objects=True)
        == distogram.min_list(h, with_objects=True))

def compare_queue_values(item_queue, number_queue):
    for item, num in zip(item_queue, number_queue):
        assert item.value == num

                    
def test_ordered_min_list():
    # TTD run coverage test
    min_list = OrderedMinMaxList("min", 4)
    # first element
    min_list.add(0)
    compare_queue_values(min_list.queue, deque([0]))
    compare_queue_values(min_list.queue, deque([0]))
    # smaller element
    min_list.add(-1)
    compare_queue_values(min_list.queue, deque([-1, 0]))
    # bigger element
    min_list.add(1)
    compare_queue_values(min_list.queue, deque([-1, 0, 1]))
    # intermediate element
    min_list.add(0.5)
    compare_queue_values(min_list.queue, deque([-1, 0, 0.5, 1]))
    # bigger element, full queue
    min_list.add(2)
    compare_queue_values(min_list.queue, deque([-1, 0, 0.5, 1]))  
    # intermediate element, full queue
    min_list.add(-0.5)
    compare_queue_values(min_list.queue, deque([-1, -0.5, 0, 0.5])) 
    # smaller element, full queue
    min_list.add(-2)
    compare_queue_values(min_list.queue, deque([-2, -1, -0.5, 0]))   
    min_list.add(-2)
    compare_queue_values(min_list.queue, deque([-2, -2, -1, -0.5]))  
     

def test_ordered_max_list():
    # TTD run coverage test
    max_list = OrderedMinMaxList("max", 4)
    # first element
    max_list.add(0)
    compare_queue_values(max_list.queue, deque([0]))
    # smaller element
    max_list.add(-1)
    compare_queue_values(max_list.queue, deque([-1, 0]))
    # bigger element
    max_list.add(1)
    compare_queue_values(max_list.queue, deque([-1, 0, 1]))
    # intermediate element
    max_list.add(0.5)
    compare_queue_values(max_list.queue, deque([-1, 0, 0.5, 1]))
    # bigger element, full queue
    max_list.add(2)
    compare_queue_values(max_list.queue, deque([0, 0.5, 1, 2]))  
    # intermediate element, full queue
    max_list.add(0.25)
    compare_queue_values(max_list.queue, deque([0.25, 0.5, 1, 2]))  
    # smaller element, full queue
    max_list.add(-2)
    compare_queue_values(max_list.queue, deque([0.25, 0.5, 1, 2]))  
    max_list.add(2)
    compare_queue_values(max_list.queue, deque([0.5, 1, 2, 2]))  


def test_merge_min_max_list_error():
    min_list = OrderedMinMaxList("min", 4)
    max_list = OrderedMinMaxList("max", 4)

    with pytest.raises(ValueError):
        min_list.merge(max_list)


def test_merge_max_list():

    max_list = OrderedMinMaxList("max", 2)
    max_list_2 = OrderedMinMaxList("max", 2)
    max_list.add(8)
    max_list.add(10)
    max_list_2.add(7)
    max_list_2.add(9)
    merged_list = max_list.merge(max_list_2)
    compare_queue_values(merged_list.queue, deque([9, 10]))

    max_list = OrderedMinMaxList("max", 3)
    max_list_2 = OrderedMinMaxList("max", 2)
    max_list.add(6)
    max_list.add(8)
    max_list.add(10)
    max_list_2.add(5)
    max_list_2.add(7)
    max_list_2.add(9)
    merged_list = max_list.merge(max_list_2)
    compare_queue_values(merged_list.queue, deque([9, 10]))

    max_list = OrderedMinMaxList("max", 2)
    max_list_2 = OrderedMinMaxList("max", 2)
    max_list.add(1)
    max_list.add(2)
    max_list_2.add(3)
    max_list_2.add(4)
    merged_list = max_list.merge(max_list_2)
    compare_queue_values(merged_list.queue, deque([3, 4]))

    max_list = OrderedMinMaxList("max", 2)
    max_list_2 = OrderedMinMaxList("max", 2)
    max_list.add(5)
    max_list.add(4)
    max_list_2.add(3)
    max_list_2.add(2)
    merged_list = max_list.merge(max_list_2)
    compare_queue_values(merged_list.queue, deque([4, 5]))


def test_merge_min_list():

    min_list = OrderedMinMaxList("min", 2)
    min_list_2 = OrderedMinMaxList("min", 2)
    min_list.add(8)
    min_list.add(10)
    min_list_2.add(7)
    min_list_2.add(9)
    merged_list = min_list.merge(min_list_2)
    compare_queue_values(merged_list.queue, deque([7, 8]))

    min_list = OrderedMinMaxList("min", 3)
    min_list_2 = OrderedMinMaxList("min", 2)
    min_list.add(6)
    min_list.add(8)
    min_list.add(10)
    min_list_2.add(5)
    min_list_2.add(7)
    min_list_2.add(9)
    merged_list = min_list.merge(min_list_2)
    compare_queue_values(merged_list.queue, deque([5, 6]))

    min_list = OrderedMinMaxList("min", 2)
    min_list_2 = OrderedMinMaxList("min", 2)
    min_list.add(1)
    min_list.add(2)
    min_list_2.add(3)
    min_list_2.add(4)
    merged_list = min_list.merge(min_list_2)
    compare_queue_values(merged_list.queue, deque([1, 2]))

    min_list = OrderedMinMaxList("min", 2)
    min_list_2 = OrderedMinMaxList("min", 2)
    min_list.add(5)
    min_list.add(4)
    min_list_2.add(3)
    min_list_2.add(2)
    merged_list = min_list.merge(min_list_2)
    compare_queue_values(merged_list.queue, deque([2, 3]))


def test_merge_min_max_distogram():
    h1 = distogram.Distogram(bin_count=3, with_min_max_list=True, min_max_list_size=2)
    h2 = distogram.Distogram(bin_count=3, with_min_max_list=True, min_max_list_size=2)
    for i in range(20):
        h1 = distogram.update(h1, i)
    for i in range(10, 30):
        h2 = distogram.update(h2, i)
    compare_queue_values(h1._min_list.queue, deque([0, 1]))
    compare_queue_values(h2._min_list.queue, deque([10, 11]))
    compare_queue_values(h1._max_list.queue, deque([18, 19]))
    compare_queue_values(h2._max_list.queue, deque([28, 29]))
    h = distogram.merge(h1, h2, preserve_inputs=True)
    compare_queue_values(h._min_list.queue, deque([0, 1]))
    compare_queue_values(h._max_list.queue, deque([28, 29]))
    compare_queue_values(h1._min_list.queue, deque([0, 1]))
    compare_queue_values(h2._min_list.queue, deque([10, 11]))
    compare_queue_values(h1._max_list.queue, deque([18, 19]))
    compare_queue_values(h2._max_list.queue, deque([28, 29]))


def test_merge_min_max_distogram_with_objects():
    h1 = distogram.Distogram(bin_count=3, with_min_max_list=True, min_max_list_size=2)
    h2 = distogram.Distogram(bin_count=3, with_min_max_list=True, min_max_list_size=2)
    for i in range(20):
        h1 = distogram.update(h1, i, obj=i)
    for i in range(10, 30):
        h2 = distogram.update(h2, i, obj=i)
    assert h1._min_list.queue[0] == distogram.Item(0, 0)
    assert h2._min_list.queue[0] == distogram.Item(10, 10)
    assert h1._max_list.queue[-1] == distogram.Item(19, 19)
    assert h2._max_list.queue[-1] == distogram.Item(29, 29)        
    # h = distogram.merge(h1, h2, preserve_inputs=True)
    # assert h._min_list.queue[0] == distogram.Item(0, 0)
    # assert h._max_list.queue[-1] == distogram.Item(29, 29)
    assert h1._min_list.queue[0] == distogram.Item(0, 0)
    assert h2._min_list.queue[0] == distogram.Item(10, 10)
    assert h1._max_list.queue[-1] == distogram.Item(19, 19)
    assert h2._max_list.queue[-1] == distogram.Item(29, 29)

def test_merge_min_max_distogram_with_objects_longer():
    size = 10
    distribution = list(range(0, size))
    ph = []
    split_count = 2
    step_size = (size // split_count)
    for i in range(split_count):
        h = distogram.Distogram(with_min_max_list=True, min_max_list_size=3)
        for index, i in enumerate(distribution[i * step_size: i * step_size + step_size]):
            h = distogram.update(h, i, obj=i)
        ph.append(h)

    h = ph[0]

    for i in range(1, split_count):
        h = distogram.merge(h, ph[i])

    assert distogram.min_list(h, with_objects=True) == [(0.0, 0), (1.0, 1), (2.0, 2)]
    assert distogram.max_list(h, with_objects=True) == [(7.0, 7), (8.0, 8), (9.0, 9)]
