import sys
import time
import numpy as np
import distogram


print("generating distribution...")
utterance_count = 10000000
distribution = np.random.normal(size=utterance_count)
#distribution = np.random.uniform(size=utterance_count)
if len(sys.argv) >= 2 and sys.argv[1] == '--enable-np':
    print('using numpy types')
else:
    print('using python types')
    distribution = distribution.tolist()

print("distribution generated")
print("running update bench (5 times)...")
total_time = 0

for i in range(5):
    start_time = time.time()

    h = distogram.Distogram()
    for i in distribution:
        h = distogram.update(h, i)

    end_time = time.time()
    total_time += end_time - start_time

total_time /= 5
print("ran update bench in {} seconds (mean of 5 runs)".format(total_time))
print("req/s: {}".format(utterance_count // total_time))
