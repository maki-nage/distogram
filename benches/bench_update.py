import time

import numpy

import utils


# This one takes the longest to run
def bench_update():
    num_samples = 1
    num_points_list = [100_000, 250_000, 500_000, 1_000_000, 2_000_000]

    times_dict: utils.TimesDict = {
        num_points: dict()
        for num_points in num_points_list
    }

    for num_points in num_points_list:
        values = numpy.random.normal(size=num_points)

        for n in range(6):
            bin_count = 32 * (2 ** n)

            start = time.time()
            [utils.create_old_distogram(bin_count, values) for _ in range(num_samples)]
            old_time = (time.time() - start) / num_samples

            start = time.time()
            [utils.create_distogram(bin_count, values) for _ in range(num_samples)]
            new_time = (time.time() - start) / num_samples

            times_dict[num_points][bin_count] = old_time, new_time

    utils.plot_times_dict(
        times_dict,
        title='update',
    )
    return


if __name__ == '__main__':
    bench_update()
