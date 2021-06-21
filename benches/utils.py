import os
from functools import reduce
from typing import Dict
from typing import Tuple

import numpy
from matplotlib import pyplot

import distogram
import old_distogram

# A Dictionary to store runtimes. The structure is intended to be:
#     { num_points: { bin_count: (old_time, new_time) } }
TimesDict = Dict[int, Dict[int, Tuple[float, float]]]

COL_NAMES = ['num_points', 'bin_count', 'old_time', 'new_time']


def create_values(mean, stddev, num_points) -> numpy.ndarray:
    return numpy.random.normal(loc=mean, scale=stddev, size=num_points)


def create_distogram(bin_count: int, values: numpy.ndarray):
    return reduce(
        lambda res, val: distogram.update(res, float(val)),
        values.flat,
        distogram.Distogram(bin_count)
    )


def create_old_distogram(bin_count: int, values: numpy.ndarray):
    return reduce(
        lambda res, val: old_distogram.update(res, float(val)),
        values.flat,
        old_distogram.Distogram(bin_count)
    )


def plot_times_dict(times_dict: TimesDict, title: str):
    pyplot.close('all')

    fig, (ax1, ax2) = pyplot.subplots(2, 1, sharex='all')

    if len(times_dict) == 1:
        num_points = next(iter(times_dict.keys()))
        x = list(sorted(times_dict[num_points].keys()))
        ys = [times_dict[num_points][k] for k in x]
        y1 = [v for v, _ in ys]
        y2 = [v for _, v in ys]
        fractions = [(u - v) / u for u, v in ys]

        ax1.plot(x, y1, label='old time')
        ax1.plot(x, y2, label='new time')
        ax1.legend()
        ax1.set_title(f'time vs bin-count for {title}')

        ax2.plot(x, fractions, label='fractions')
        ax2.set_ylim([0, max(fractions) * 1.2])

        ax2.set_xlabel('bin-count')

    else:
        x = list(sorted(times_dict.keys()))
        max_fraction = 0

        for bin_count in sorted(times_dict[x[0]].keys()):
            y = [times_dict[num_points][bin_count] for num_points in x]
            y1 = [v for v, _ in y]
            y2 = [v for _, v in y]
            fractions = [(u - v) / u for u, v in y]
            max_fraction = max(max_fraction, max(fractions))

            ax1.plot(x, y1, label=f'old, {bin_count} bins')
            ax1.plot(x, y2, label=f'new, {bin_count} bins')

            ax2.plot(x, fractions, label=f'fractions, {bin_count} bins')

        ax2.legend()
        ax2.set_ylim([0, max_fraction * 1.2])
        ax2.set_xlabel('num-points')

    ax1.set_ylabel('time (s)')
    ax2.set_ylabel('(old - new) / old')

    ax1.legend()

    os.makedirs('plots', exist_ok=True)
    pyplot.savefig(f'plots/{title}.png')

    return
