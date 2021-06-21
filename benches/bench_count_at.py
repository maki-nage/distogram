import time

import numpy

import distogram
import old_distogram
import utils


def bench_count_at():
    num_samples = 100
    num_points = 100_000
    values = numpy.random.normal(size=num_points)

    times_dict: utils.TimesDict = {num_points: dict()}

    for n in range(6):
        bin_count = 32 * (2 ** n)
        h = utils.create_distogram(bin_count, values)

        start = time.time()
        [old_distogram.count_at(h, 0) for _ in range(num_samples)]
        old_time = (time.time() - start) / num_samples

        start = time.time()
        [distogram.count_at(h, 0) for _ in range(num_samples)]
        new_time = (time.time() - start) / num_samples

        times_dict[num_points][bin_count] = old_time, new_time

    utils.plot_times_dict(
        times_dict,
        title='count-at',
    )
    return


if __name__ == '__main__':
    bench_count_at()
