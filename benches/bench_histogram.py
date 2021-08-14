import time

import numpy

import distogram
import utils


def bench_histogram():
    num_samples = 100
    num_points = 100_000
    values = numpy.random.normal(size=num_points)

    times_dict: utils.TimesDict = {num_points: dict()}

    for n in range(6):
        bin_count = 32 * (2 ** n)
        h = utils.create_distogram(bin_count, values)

        start = time.time()
        [distogram.histogram(h, ucount=bin_count) for _ in range(num_samples)]
        time_taken = (time.time() - start) / num_samples

        times_dict[num_points][bin_count] = time_taken

    utils.plot_times_dict(
        times_dict,
        title='histogram',
    )
    return


if __name__ == '__main__':
    bench_histogram()
