Example
============

+++++++++++++++
Basic usage
+++++++++++++++


First, you have to import the necessary modules from the package and then you can generate a random Hermitian 
matrix using :meth:`~QPCA.preprocessingUtilities.generate_matrix` method provided in the package. You can 
set the matrix dimension and a seed for the reproducibility of the execution.

..  code-block:: python

    /**
     * 
     */
      from QPCA.decomposition.Qpca import QPCA
      import numpy as np
      import matplotlib.pyplot as plt
      import random
      import pandas as pd
      from QPCA.preprocessingUtilities.preprocessing import generate_matrix

      matrix_dimension=2
      seed=19
      input_matrix=generate_matrix(matrix_dimension=matrix_dimension,seed=seed)

..  code-block:: python

   >>> print(input_matrix)
   
   [[0.59 0.13]
   [0.13 0.08]]

   eigenvalue: 0.04912229458049476 - eigenvector: [-0.233  0.973]
   eigenvalue: 0.6199503657038241 - eigenvector: [0.973 0.233]

Once you have your input matrix, you can fit your QPCA model, specifying the number of resolution qubit 
that you need for the phase estimation process. Remember that a higher resolution generally means better 
accuracy results but lower performance. Remember that the input matrix will be normalized by its trace, therefore the eigenvalues could change.

..  code-block:: python

      resolution=8
      qpca=QPCA().fit(input_matrix,resolution=resolution,plot_qram=True,plot_pe_circuit=True)
      
..  code-block:: python

   >>> print(np.linalg.eig(qpca.input_matrix))
  
   (array([0.92658152, 0.07341848]),
   array([[ 0.9725247 , -0.23279972],
          [ 0.23279972,  0.9725247 ]]))

If you set the boolean flag plot_qram and plot_pe_circuit to True as in the example before, you are able to see
two plots like the ones below.

Specifically, this plot shows the circuit that implements the encoding of the input matrix in the quantum registers. By default, an optimized version 
of the encoding circuit is implemented using StatePreparation class of Qiskit.

.. image:: Images/optimized_qram.png

If you specify :obj:`~QPCA.decomposition.QPCA.optimized_qram` as False in the :meth:`~QPCA.decomposition.QPCA.fit` method, a custom version 
of the encoding circuit is implemented.
As you can see, the number of qubit required to store the matrix is in the order of log(n*m), where n and m 
are the number of rows and columns of the input matrix.

.. image:: Images/qram.png

The other plot shows the general circuit made of the encoding part plus the phase estimation operator.
Notice that the number of qubits used for the phase estimation in this case are 9: 8 specified by the resolution
parameter to encode the eigenvalues and 1 to encode the eigenvectors. In general, you will have the qubits specified
in the resolution parameter plus half of the qubits used for the matrix encoding.

.. image:: Images/pe.png

The core part of this library is the eigenvector reconstruction that you can perform using :meth:`~QPCA.decomposition.QPCA.eigenvectors_reconstruction`. You can
specify, as input parameters, :obj:`~QPCA.decomposition.QPCA.n_shots` which is the number of measure that you
want to perform in the state vector tomography, :obj:`~QPCA.decomposition.QPCA.n_repetitions` which is the 
number of times that you want to repeat the tomography process, and :obj:`~QPCA.decomposition.QPCA.plot_peaks`
if you want to plot the output of the phase estimation which represent the most valuable approximated eigenvalues.

..  code-block:: python

      eig=qpca.eigenvectors_reconstruction(n_shots=1000000,n_repetitions=1,plot_peaks=True)

..  code-block:: python

   >>> print(eig)
   
   array([0.92578125, 0.07421875]),
   array([[ 0.97257301, -0.22836194],
        [ 0.23277106,  0.97266614]])

With the boolean flag :obj:`~QPCA.decomposition.QPCA.plot_peaks` set to True, you can visualize a plot like the 
one below, where you can see the peaks that represent the eigenvalues that phase estimation approximates with high probability.
As you can see, here the two peaks are 0.92 and 0.07 which are the two eigenvalues that you are able to 
estimate with the resolution and the number of shots that you provide.

.. image:: Images/peaks.png

Finally, you can reconstruct the original input matrix using :meth:`~QPCA.decomposition.QPCA.quantum_input_matrix_reconstruction`. 

..  code-block:: python

      rec_input_matrix=qpca.quantum_input_matrix_reconstruction()


..  code-block:: python

   >>> print(rec_input_matrix)
   
   array([[0.5884931 , 0.12919742],
         [0.12919742, 0.08054153]])

+++++++++++++++
Threshold optimization 
+++++++++++++++

In the :meth:`~QPCA.decomposition.QPCA.quantum_input_matrix_reconstruction` method, you can specify the :obj:`~QPCA.decomposition.QPCA.eigenvalue_threshold` parameter
to cut off the estimated eigenvalues that are smaller than the specified value.

..  code-block:: python

      eig=qpca.eigenvectors_reconstruction(n_shots=1000000, n_repetitions=1, plot_peaks=True, eigenvalue_threshold=0.1)

As you can see below, by specifying a threshold of 0.1, you cut off the last eigenvalue and you keep only the greatest one.

..  code-block:: python

   >>> print(eig)

   array([0.92578125]),
   array([[0.9725207],
        [0.2333083]])

.. image:: Images/threshold.png

This type of threshold can be useful to cut out the smallest eigenvalues that are the most problematic to estimate
and whose associated eigenvectors are those with the highest reconstruction error.

+++++++++++++++
Absolute tolerance 
+++++++++++++++

The absolute tolerance is a kind of threshold that allows to discard the noisy eigenvalues (and consequently the respective eigenvectors) that could arise when the number of resolution qubits
and/or the number of measurements performed in the tomography are not high enough.

Let's see the following example.
To better visualize the problem, a 4x4 matrix is considered with 6 qubits of resolution and 1000000 shots performed to reconstruct the eigenvectors.

..  code-block:: python

      resolution=6
      matrix_dimension=4
      input_matrix=generate_matrix(matrix_dimension=matrix_dimension,seed=seed)

..  code-block:: python

   >>> print(input_matrix)
   
      [[0.63 0.55 0.5  0.89]
      [0.55 1.41 1.1  1.3 ]
      [0.5  1.1  1.08 1.47]
      [0.89 1.3  1.47 2.36]]

      eigenvalue: 0.01593042549125613 - eigenvector: [ 0.23  -0.357  0.812 -0.4  ]
      eigenvalue: 0.2943707848528235 - eigenvector: [ 0.882  0.207 -0.303 -0.295]
      eigenvalue: 0.5238941243476808 - eigenvector: [-0.304  0.772  0.163 -0.534]
      eigenvalue: 4.647071393343875 - eigenvector: [-0.277 -0.483 -0.471 -0.685]

..  code-block:: python

      qpca=QPCA().fit(input_matrix,resolution=resolution,plot_qram=True,plot_pe_circuit=True)

..  code-block:: python

   >>> print(np.linalg.eig(qpca.input_matrix))
  
   (array([0.84780975, 0.09557902, 0.05370488, 0.00290634]),
   array([[-0.27669967, -0.30381059,  0.88229208,  0.2295585 ],
         [-0.48274483,  0.77216236,  0.20746302, -0.35732594],
         [-0.47083591,  0.16297368, -0.30291613,  0.81240073],
         [-0.68462272, -0.53376399, -0.29455322, -0.39953239]]))

..  code-block:: python

      eig=qpca.eigenvectors_reconstruction(n_shots=1000000,n_repetitions=1,plot_peaks=True)

As you can see below, there is an eigenvalue (0.265625 in this case) which doesn't match any of the original eigenvalues. Indeed, even the peaks plot doesn't show a peak around 0.26. Therefore,
this is a fluctuation or a noisy eigenvalue that is due to the classical postprocessing since the classical eigenvalues extractor algorithm searches for at most 4 eigenvalues (this is because 4 is the initial matrix dimension).

But as you can see, the QPCA algorithm, with the configuration specified at the beginning, found 3 peaks or "correct" eigenvalues. The fourth, that corresponds to the smallest original eigenvalue,
is something added by the postprocessing.

..  code-block:: python

   >>> print(eig)
   
   array([0.84375 , 0.09375 , 0.046875, 0.265625]),
   array([[ 0.27680417, -0.27660952,  0.80547637,  0.29495506],
        [ 0.48230629,  0.67785086,  0.10991118,  0.48594109],
        [ 0.4694504 ,  0.13039498, -0.38167556,  0.41712944],
        [ 0.68528434, -0.52157991, -0.38353287,  0.69620819]])

.. image:: Images/absolute_tolerance1.png

To tackle this problem, you can both increase the number of qubits of resolution and/or the number of shots. But if these numbers are already big enough
and you can't increase them for performance reasons, you can specify the :obj:`~QPCA.decomposition.QPCA.abs_tolerance` parameter setting a specific tolerance.

As you can see, by setting this parameter to 0.001, you can remove the noisy eigenvalue and return all the correct estimated eigenvalues/eigenvectors. If you also want to correctly estimate 
the smallest eigenvalue, you probably need to increase the number of qubits of resolution.

..  code-block:: python

      eig=qpca.eigenvectors_reconstruction(n_shots=1000000,n_repetitions=1,plot_peaks=True,abs_tolerance=1e-03)

..  code-block:: python

   >>> print(eig)
   
   array([0.84375 , 0.09375 , 0.046875]),
   array([[ 0.27703224, -0.26507805,  0.81246826],
        [ 0.48292227,  0.68519289,  0.11092602],
        [ 0.46941485,  0.12992164,  0.36394397],
        [ 0.68487267, -0.5051309 ,  0.37952075]])

Basically, the peaks are extracted by looking at their average vertical distance from their neighbors means. Therefore, specifying an absolute tolerance means specifying the average vertical height 
below which a peak is no longer considered a peak but is seen as a fluctuation or noise.
So, how to chose the absolute tolerance parameter? If you don't specify, it takes a default value of 1/n_shots. This is because the average vertical distance from the neighbors is in some sense related
to the number of shots performed in the tomography. But due to the statistical variance in measuring, this is not always the case. So the best thing to do if an unexpected eigenvalue occurs is to try
increasing the tolerance by an order of magnitude with respect to 1/n_shots (clearly the best solution would be to increase the resolution, where possible).
