==========
DistoGram
==========


.. image:: https://badge.fury.io/py/distogram.svg
    :target: https://badge.fury.io/py/distogram

.. image:: https://github.com/maki-nage/distogram/workflows/Python%20package/badge.svg
    :target: https://github.com/maki-nage/distogram/actions?query=workflow%3A%22Python+package%22
    :alt: Github WorkFlows

.. image:: https://readthedocs.org/projects/distogram/badge/?version=latest
    :target: https://distogram.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


DistoGram is a library that allows to compute histogram on streaming data, in
distributed environments. The implementation follows the algorithms described in
Ben-Haim's `Streaming Parallel Decision Trees
<http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf>`__

Get Started
============

.. code:: python

    import distogram

    h = distogram.Distogram()
    h = distogram.sum(h, 42)


Install
========

DistoGram is available on PyPi and can be installed with pip:

.. code:: console

    pip install distogram


