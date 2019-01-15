dynlearn: dynamical system active learning
-----------------------------------------------

A dynamical system can be driven in different directions depending on
a forcing external input to the system. The aim is to find optimal inputs
so that the system produces certain species to a specified target level.
In an iterative active learning cycle the inputs are successively
optimised towards that target with experiments performed using the
suggested inputs. Since the dynamics of the system is
unknown it is approximated by a Gaussian process regression.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Demo: optimise gene expression
---------------------------------
.. automodule:: dynlearn.demo
   :members:

.. image:: images/Nanog_target_50.png
   :width: 600

Simulation
----------
.. automodule:: dynlearn.simulation
   :members:

Gaussian process
--------------------------
.. automodule:: dynlearn.gp_kernels
   :members:

Gaussian process state space model
----------------------------------------------
.. automodule:: dynlearn.learn
   :members:

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
