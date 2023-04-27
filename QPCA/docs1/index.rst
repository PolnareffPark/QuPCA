.. Qpca documentation master file, created by
   sphinx-quickstart on Thu Mar 30 16:07:55 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Qpca's documentation!
================================
This is the documentation of the Qpca package implemented using Qiskit SDK. 
The package is organized in 4 modules:

* :mod:`~QPCA.decomposition` module: here there is :class:`~QPCA.decomposition.QPCA` Qpca class implementation. One of the main method is :meth:`~QPCA.decomposition.QPCA.fit` to fit the Qpca model. In this context, fitting the model means building the circuit which encode an arbitrary input matrix that you provide as input and to perform phase estimation to encode the eigenvalues of the input matrix in the quantum registers. With :meth:`~QPCA.decomposition.QPCA.eigenvectors_reconstruction` method you can reconstruct the original eigenvectors (and eigenvalues) by exploiting the :meth:`~QPCA.quantumUtilities.state_vector_tomography` method which implements state vector tomography algorithm. Finally, once you have reconstructed the eigenvectors and eigenvalues, you can also reconstruct the original input matrix using :meth:`~QPCA.decomposition.QPCA.quantum_input_matrix_reconstruction`. If the output is not satisfying, you can think to increase the number of shots or the number of qubits for the phase estimation resolution.

* :mod:`~QPCA.quantumUtilities` module: here there are the main quantum routines used by the Qpca algorithm. The main function is :meth:`~QPCA.quantumUtilities.state_vector_tomography` that is used by Qpca to reconstruct the eigenvectors but can also be used as a stand alone function to reconstruct the statevector of an arbitrary quantum circuit (composed only of real amplitudes).

* :mod:`~QPCA.preprocessingUtilities` module: in this module, you find the :meth:`~QPCA.preprocessingUtilities.generate_matrix` method which is useful to generate an arbitrary random Hermitian input matrix for the Qpca algorithm.

* :mod:`~QPCA.postprocessingUtilities` module: in this module, you find the :meth:`~QPCA.postprocessingUtilities.general_postprocessing` method which is used by Qpca algorithm to recontruct the eigenvectors/eigenvalues using the information provided by quantum state tomography.

* :mod:`~QPCA.benchmark` module: in this module, you find the methods to benchmark the execution of Qpca algorithm. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   modules
   Example_modules
   Example_tomography_modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
