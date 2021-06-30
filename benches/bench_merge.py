import time
from functools import reduce

import distogram
import utils


def bench_merge():
    num_samples = 10
    num_points = 100_000
    values_list = [
        utils.create_values(mean, 0.3, num_points)
        for mean in range(num_samples)
    ]

    times_dict: utils.TimesDict = {num_points: dict()}

    for n in range(6):
        bin_count = 32 * (2 ** n)

        histograms = [
            utils.create_distogram(bin_count, values)
            for values in values_list
        ]
        start = time.time()
        _ = reduce(
            lambda res, val: distogram.merge(res, val),
            histograms,
            distogram.Distogram(bin_count=bin_count)
        )
        time_taken = (time.time() - start) / num_samples

        times_dict[num_points][bin_count] = time_taken

    utils.plot_times_dict(
        times_dict,
        title='merge',
    )
    return


if __name__ == '__main__':
    bench_merge()
