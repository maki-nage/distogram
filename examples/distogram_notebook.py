# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: venv_gcp-beam-stats
#     language: python
#     name: venv_gcp-beam-stats
# ---

# +
import math
import random
import distogram

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# -

# # Create a distribution

size=10000
distribution = np.random.normal(size=size)
#distribution = np.random.uniform(size=size)
#distribution = np.random.laplace(size=size)
#distribution = np.random.exponential(size=size)
#distribution = np.random.triangular(0, 1, 5, size=size)

# # Compute statistics

# Create and feed distogram from distribution
h = distogram.Distogram()
for i in distribution:
    h = distogram.update(h, i)

# +
# compute estimated metrics
nmin, nmax = distogram.bounds(h)
print("count: {}".format(distogram.count(h)))
print("mean: {}".format(distogram.mean(h)))
print("stddev: {}".format(distogram.stddev(h)))
print("min: {}".format(nmin))
print("5%: {}".format(distogram.quantile(h, 0.05)))
print("25%: {}".format(distogram.quantile(h, 0.25)))
print("50%: {}".format(distogram.quantile(h, 0.50)))
print("75%: {}".format(distogram.quantile(h, 0.75)))
print("95%: {}".format(distogram.quantile(h, 0.95)))
print("max: {}".format(nmax))

# Compare with real metrics from pandas
df_distribution = pd.DataFrame(distribution)
display(df_distribution.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

# +
# Compute estimated histogram of the distribution 
hist = distogram.histogram(h, bin_count=100)
df_hist = pd.DataFrame(hist[1][0:-1], columns=["bin"])
df_hist["count"] = hist[0]
fig = px.bar(df_hist, x="bin", y="count", title="distogram")
fig.update_layout(height=300)
fig.show()

# compare with real histogram of the distribution
np_hist = np.histogram(distribution, bins=100)
df_phist = pd.DataFrame(np_hist[1][0:-1], columns=["bin"])
df_phist["count"] = np_hist[0]
fig2 = px.bar(df_phist, x="bin", y="count", title="numpy histogram")
fig2.update_layout(height=300)
fig2.show()


# -

# # Simulate a distributed usage

ph = []
split_count = 5
step_size = (size//split_count)
for i in range(split_count):
    h = distogram.Distogram()
    for i in distribution[i*step_size: i*step_size+step_size]:
        h = distogram.update(h, i)
    ph.append(h)

# # Merge partial distributions

h = ph[0]
for i in range(1, split_count):
    h = distogram.merge(h, ph[i])

# +
# Compute estimated histogram of the distribution 
hist = distogram.histogram(h)
df_hist = pd.DataFrame(hist[1][0:-1], columns=["bin"])
df_hist["count"] = hist[0]
fig = px.bar(df_hist, x="bin", y="count", title="distogram")
fig.update_layout(height=300)
fig.show()

# compare with real histogram of the distribution
np_hist = np.histogram(distribution, bins=100)
df_phist = pd.DataFrame(np_hist[1][0:-1], columns=["bin"])
df_phist["count"] = np_hist[0]
fig2 = px.bar(df_phist, x="bin", y="count", title="numpy histogram")
fig2.update_layout(height=300)
fig2.show()


# -

# # Frequency density distribution examples

# +
# helper functions for plotting

def histogram_bar_plot_data(np_hist, columns):
    counts, bins2 = np_hist
    if len(bins2) != len(counts) + 1:
        raise ValueError("histogram data is invalid. len(bins) != len(counts) + 1")
    bins2 = [(old + new)/2 for old, new in zip(bins2[:-1],bins2[1:])]
    data = [[bin1, count] for count, bin1 in zip(counts,bins2)]
    df = pd.DataFrame(data, columns=columns)
    return df

def histogram_step_plot_data(np_hist, columns):
    counts, bins2 = np_hist
    if len(bins2) != len(counts) + 1:
        raise ValueError("histogram data is invalid. len(bins) != len(counts) + 1")
    data = [[bins2[0], 0]]
    data.extend([[bin1, count] for count, bin1 in zip(counts,bins2)])
    data.append([bins2[-1],0])
    df = pd.DataFrame(data, columns=columns)
    return df


# -

# ##  Frequency density results are similar for most distributions

# +
# Create distribution
size=10000
distribution = np.random.normal(size=size)
#distribution = np.random.uniform(size=size)
#distribution = np.random.laplace(size=size)
#distribution = np.random.exponential(size=size)
#distribution = np.random.triangular(0, 1, 5, size=size)

# Create distogram
h = distogram.Distogram()
for i in distribution:
    h = distogram.update(h, i)

# Compute estimated histogram of the distribution 
hist = distogram.histogram(h)
df_hist = histogram_step_plot_data(hist, ["bin", "count"])
fig = px.line(
    df_hist,
    x="bin",
    y="count",
    log_y=True,
    title="distogram step plot"
)
fig.update_traces(mode="lines", line_shape="hv")
fig.update_layout(height=300)
fig.show()
print(f"Sum of bins: {df_hist['count'].sum()}")

# Compute estimated frequency density distribution of the distribution 
hist = distogram.frequency_density_distribution(h)
df_hist = histogram_step_plot_data(hist, ["bin", "count/bin width"])
fig = px.line(
    df_hist,
    x="bin",
    y="count/bin width",
    log_y=True,
    title="frequency density distribution"
)
fig.update_traces(mode="lines", line_shape="hv")
fig.update_layout(height=300)
fig.show()
integral = 0
for density, new, old in zip(hist[0], hist[1][1:], hist[1][:-1]):
    integral += density * (new-old)
print(f"Sum of bins: {integral}")

# Compute numpy histogram of the distribution
hist = np.histogram(distribution, bins=100)
df_hist = histogram_step_plot_data(hist, ["bin", "count"])
fig = px.line(
    df_hist,
    x="bin",
    y="count",
    log_y=True,
    title="numpy histogram step plot"
)
fig.update_traces(mode="lines", line_shape="hv")
fig.update_layout(height=300)
fig.show()
print(f"Sum of bins: {df_hist['count'].sum()}")

# -

# ##  Frequency density results retain more detail when there are extreme outliers

# +
# Create distribution
size=10000
distribution = np.random.normal(size=size)
#distribution = np.random.uniform(size=size)
#distribution = np.random.laplace(size=size)
#distribution = np.random.exponential(size=size)
#distribution = np.random.triangular(0, 1, 5, size=size)

# Add outlier
distribution = np.append(distribution, [20*max(distribution)])

# Create distogram
h = distogram.Distogram()
for i in distribution:
    h = distogram.update(h, i)
    
# Compute estimated histogram of the distribution 
hist = distogram.histogram(h)
df_hist = histogram_step_plot_data(hist, ["bin", "count"])
fig = px.line(
    df_hist,
    x="bin",
    y="count",
    log_y=True,
    title="distogram step plot"
)
fig.update_traces(mode="lines", line_shape="hv")
fig.update_layout(height=300)
fig.show()
print(f"Sum of bins: {df_hist['count'].sum()}")

# Compute estimated frequency density distribution of the distribution 
hist = distogram.frequency_density_distribution(h)
df_hist = histogram_step_plot_data(hist, ["bin", "count/bin width"])
fig = px.line(
    df_hist,
    x="bin",
    y="count/bin width",
    log_y=True,
    title="frequency density distribution"
)
fig.update_traces(mode="lines", line_shape="hv")
fig.update_layout(height=300)
fig.show()
integral = 0
for density, new, old in zip(hist[0], hist[1][1:], hist[1][:-1]):
    integral += density * (new-old)
print(f"Sum of bins: {integral}")

# Compute numpy histogram of the distribution
hist = distogram.histogram(h)
df_hist = histogram_step_plot_data(hist, ["bin", "count"])
fig = px.line(
    df_hist,
    x="bin",
    y="count",
    log_y=True,
    title="numpy histogram step plot"
)
fig.update_traces(mode="lines", line_shape="hv")
fig.update_layout(height=300)
fig.show()
print(f"Sum of bins: {df_hist['count'].sum()}")
# -

# ##  Other numpy histogram methods require many more bins for fidelity similar to frequency distribution results

# +
# Create distribution
size=10000
distribution = np.random.normal(size=size)
# distribution = np.random.uniform(size=size)
# distribution = np.random.laplace(size=size)
# distribution = np.random.exponential(size=size)
# distribution = np.random.triangular(0, 1, 5, size=size)

# Add outlier
distribution = np.append(distribution, [20*max(distribution)])

# Create distogram
h = distogram.Distogram()
for i in distribution:
    h = distogram.update(h, i)

# Compute estimated frequency density distribution of the distribution 
hist = distogram.frequency_density_distribution(h)
df_hist = histogram_step_plot_data(hist, ["bin", "count/bin width"])
fig = px.line(
    df_hist,
    x="bin",
    y="count/bin width",
    log_x=True,
    log_y=True,
    title="frequency density distribution"
)
fig.update_traces(mode="lines", line_shape="hv")
fig.update_layout(height=300)
fig.show()
integral = 0
for density, new, old in zip(hist[0], hist[1][1:], hist[1][:-1]):
    integral += density * (new-old)
print(f"Number of bins: {len(df_hist)}")

# Compare to other numpy histogram methods
method_list = [
    'auto',
    # Maximum of the ‘sturges’ and ‘fd’ estimators. Provides good all around performance.
    # 'fd', #(Freedman Diaconis Estimator)
    # Robust (resilient to outliers) estimator that takes into account data variability and data size.
    # 'doane',
    # An improved version of Sturges’ estimator that works better with non-normal datasets.
    # 'scott',
    # Less robust estimator that that takes into account data variability and data size.
    # 'stone',
    # Estimator based on leave-one-out cross-validation estimate of the integrated squared error. Can be regarded as a generalization of Scott’s rule.
    'rice',
    # Estimator does not take variability into account, only data size. Commonly overestimates number of bins required.
    # 'sturges',
    # R’s default method, only accounts for data size. Only optimal for gaussian data and underestimates number of bins for large non-gaussian datasets.
    # 'sqrt'
    # Square root (of data size) estimator, used by Excel and other programs for its speed and simplicity.
]

for method in method_list:
    hist = np.histogram(distribution, bins=method)
    df_hist = histogram_step_plot_data(hist, ["bin", "count"])
    fig = px.line(
        df_hist,
        x="bin",
        y="count",
        log_x=True,
        log_y=True,
        title=f"numpy histogram step plot, method = '{method}'"
    )
    fig.update_traces(mode="lines", line_shape="hv")
    fig.update_layout(height=300)
    fig.show()
    print(f"Number of bins: {len(df_hist)}")
 
