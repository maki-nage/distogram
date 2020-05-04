Get Started
============

DistoGram is a library that allows to compute histogram on streaming data, in
distributed environments. The implementation follows the algorithms described in
Ben-Haim's `Streaming Parallel Decision Trees
<http://jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf>`__


.. code:: python

    import distogram

    h = distogram.Distogram()
    h = distogram.sum(h, 42)
