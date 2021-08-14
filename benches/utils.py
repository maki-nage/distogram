import os
from functools import reduce
from typing import Dict

import numpy
from matplotlib import pyplot

import distogram

# A Dictionary to store runtimes. The structure is intended to be:
#     { num_points: { bin_count: time_takes } }
TimesDict = Dict[int, Dict[int, float]]

COL_NAMES = ['num_points', 'bin_count', 'old_time', 'new_time']


def create_values(mean, stddev, num_points) -> numpy.ndarray:
    return numpy.random.normal(loc=mean, scale=stddev, size=num_points)


def create_distogram(bin_count: int, values: numpy.ndarray):
    return reduce(
        lambda res, val: distogram.update(res, float(val)),
        values.flat,
        distogram.Distogram(bin_count)
    )


def plot_times_dict(times_dict: TimesDict, title: str):
    pyplot.close('all')

    pyplot.figure()

    if len(times_dict) == 1:
        num_points = next(iter(times_dict.keys()))
        x = list(sorted(times_dict[num_points].keys()))
        ys = [times_dict[num_points][k] for k in x]

        pyplot.plot(x, ys, label='time taken')
        pyplot.title(f'time vs bin-count for {title}')
        pyplot.xlabel('bin-count')

    else:
        x = list(sorted(times_dict.keys()))

        for bin_count in sorted(times_dict[x[0]].keys()):
            y = [times_dict[num_points][bin_count] for num_points in x]
            pyplot.plot(x, y, label=f'{bin_count} bins')

        pyplot.title(f'time vs num-points for {title}')
        pyplot.xlabel('num-points')

    pyplot.ylabel('time (s)')
    pyplot.legend()

    os.makedirs('plots', exist_ok=True)
    pyplot.savefig(f'plots/{title}.png', dpi=300)

    return
