import numpy as np
import itertools
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import random
import warnings
from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.circuit.library import PhaseEstimation
from qiskit import Aer, transpile, execute
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import math
from .._quantumUtilities.quantum_utilities import thetas_computation,from_binary_tree_to_qcircuit,state_vector_tomography#,q_ram_pHe_quantum_circuit_generation
from .._postprocessingUtilities.postprocessing_eig_reconstruction import general_postprocessing
from .._benchmark.benchmark import _eigenvectors_benchmarking,_eigenvalues_benchmarking,_error_benchmark,_error_benchmark_from_scratch
from scipy.spatial import distance
#warnings.filterwarnings("ignore")

class QPCA():
    """Quantum Principal component analysis (QPCA).
    Implementation of the QPCA algorithm proposed in "A Low Complexity Quantum Principal Component Analysis Algorithm" paper.

    Parameters
    ----------
    input_matrix : array-like of shape (n_samples, n_features)
            Input hermitian matrix on which you want to apply QPCA, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        
    resolution : int value
            Number of qubits used for the phase estimation process to encode the eigenvalues.
    
    Attributes
    ----------
    
    input_matrix_trace : NumPy array-like of shape (n_features,)
                        The trace of the input matrix.
    
    input_matrix : array-like of shape (n_samples, n_features)
                        Input hermitian matrix on which you want to apply QPCA divided by its trace, where `n_samples` is the number of samples
                        and `n_features` is the number of features.
    
    qram_circuit : QuantumCircuit 
                        The quantum circuit that encodes the input matrix.
                    
    total_circuit : QuantumCircuit
                        The quantum circuit that performs the encoding of the input matrix and Phase Estimation. The number of qubits will be log(n_samples*n_features)+resolution.
                
    n_shots : int value
                        Number of measures performed in the tomography process.
    
    reconstructed_eigenvalue_eigenvector_tuple : array-like
                        Array of tuples of the form [(e_1,v_1),(e_2,v_2),..] where e_s are the eigenvalues and v_s are the reconstructed eigenvectors.
    
    original_eigenValues : array-like
                        Original eigenvalues of the input matrix.
                
    original_eigenVectors : array-like
                        Orignal eigenvectors of the input matrix.
    
    Notes
    ----------
    It is important to consider that the input matrix is divided by its trace so as to have the eigenvalues between 0 and 1.
    
    """
    
    
    def __init__(self,input_matrix,resolution):
        #input_matrix=input_matrix/np.trace(input_matrix)
        self.input_matrix_trace=np.trace(input_matrix)
        self.input_matrix=input_matrix/np.trace(input_matrix)
        self.resolution=resolution
        eigenValues,eigenVectors=np.linalg.eig(self.input_matrix)
        idx = eigenValues.argsort()[::-1]   
        self.original_eigenValues = eigenValues[idx]
        self.original_eigenVectors = eigenVectors[:,idx]
            

    def generate_qram_circuit(self):
        
        """
        Generate qram circuit.

        Parameters
        ----------

        Returns
        ----------
        qc: QuantumCircuit 
                    The quantum circuit that encodes the input matrix.
                    
        Notes
        ----------
        This method implements the quantum circuit generation to encode a generic input matrix. It is important to note the spatial complexity of the circuit that is in the order of
        log2(n_samples, n_features).
        """
        
        input_matrix=self.input_matrix
        
        thetas, all_combinations=thetas_computation(input_matrix=input_matrix)
        
        qc=from_binary_tree_to_qcircuit(input_matrix,thetas, all_combinations)
        
        self.qram_circuit=qc
        
        return qc

    def generate_phase_estimation_circuit(self):
        """
        Generate phase estimation circuit with a number of qubits provided as resolution parameter in the constructor.

        Parameters
        ----------

        Returns
        ----------
        q_circuit: QuantumCircuit. 
                The quantum circuit that performs the encoding of the input matrix and Phase Estimation.
                    
        Notes
        ----------

        """
        
        u_circuit = NumPyMatrix(self.input_matrix, evolution_time=2*np.pi)#/2**resolution
        pe = PhaseEstimation(self.resolution, u_circuit, name = "PE")
        tot_qubit = pe.qregs[0].size+self.qram_circuit.qregs[0].size
        qr_total = QuantumRegister(tot_qubit, 'total')
        q_circuit = QuantumCircuit(qr_total , name='matrix')
        q_circuit.append(self.qram_circuit.to_gate(), qr_total[pe.qregs[0].size:])
        q_circuit.append(pe.to_gate(), qr_total[0:pe.num_qubits])
        self.total_circuit=q_circuit
        return q_circuit
        


    def eigenvectors_reconstruction(self,n_shots=50000,n_repetitions=1,plot_peaks=False):
        
        """ Method that reconstructs the eigenvalues/eigenvectors once performed Phase Estimation. 

        Parameters
        ----------
        n_shots: int value, default=50000.
                Number of measures performed in the tomography process.
                
        n_repetitions: int value, default=1.
                Number of times that state vector tomography will be executed. If > 1, the final result will be the average result
                of all the execution of the tomography.
                
        plot_peaks: bool value, defualt=False
                        If True, it returns a plot of the peaks which correspond to the eigenvalues finded by the phase estimation procedure.
        
        Returns
        ----------
        eigenvalue_eigenvector_tuple: array-like. 
                List of tuples containing as first value the reconstructed eigenvalue and as second value the reconstructed eigenvector.
                    
        Notes
        ----------
        To classically reconstruct the eigenvectors, state vector tomography function is performed (implemented from algorithm 4.1 of "A Quantum Interior Point Method for LPs and SDPs" paper). In this
        way, the statevector of the quantum state is reconstructed and a postprocessing method is executed to get the eigenvectors from the reconstructed statevector.
        """
        
        def wrapper_state_vector_tomography(quantum_circuit,n_shots):
            
            assert n_repetitions>0, "n_repetitions must be greater than 0."
            self.n_shots=n_shots

            if n_repetitions==1:
                tomo_dict=state_vector_tomography(quantum_circuit,n_shots)
                statevector_dictionary=tomo_dict
            else:
                tomo_dict=[state_vector_tomography(quantum_circuit,n_shots) for j in range(n_repetitions)]
                keys=list(tomo_dict[0].keys())
                new_tomo_dict={}
                for k in keys:
                    tmp=[]
                    for d in tomo_dict:
                        tmp.append(d[k])
                    mean=np.mean(tmp)
                    new_tomo_dict.update({k:mean})
                    statevector_dictionary=new_tomo_dict

            return statevector_dictionary
        
        statevector_dictionary=wrapper_state_vector_tomography(quantum_circuit=self.total_circuit,n_shots=n_shots)
        eigenvalue_eigenvector_tuple=general_postprocessing(statevector_dictionary=statevector_dictionary,resolution=self.resolution,n_shots=self.n_shots,plot_peaks=plot_peaks)
        self.reconstructed_eigenvalue_eigenvector_tuple=eigenvalue_eigenvector_tuple
        return eigenvalue_eigenvector_tuple
    
    def quantum_input_matrix_reconstruction(self):
        
        """ Method to reconstruct the input matrix.

        Parameters
        ----------

        Returns
        ----------
        reconstructed_input_matrix: array-like of shape (n_samples, n_features)
                Reconstructed input matrix.
                    
        Notes
        ----------
        Using the reconstructed eigenvectors and eigenvalues from QPCA, we can reconstruct the original input matrix using the reverse procedure of SVD.
        """
        
        reconstructed_eigenvalues=np.array([])
        reconstructed_eigenvectors=np.array([])
        for t in self.reconstructed_eigenvalue_eigenvector_tuple:
            reconstructed_eigenvalues=np.append(reconstructed_eigenvalues,t[0])
            reconstructed_eigenvectors=np.append(reconstructed_eigenvectors,t[1])
        try:
            reconstructed_eigenvectors=reconstructed_eigenvectors.reshape(len(reconstructed_eigenvalues),len(reconstructed_eigenvalues),order='F')
        except:
            raise Exception('Ops! QPCA was not able to reconstruct all the eigenvectors! Please check that you are not considering eigenvalues equal to zero.')
        k = reconstructed_eigenvalues.argsort()[::-1]   
        reconstructed_eigenvalues = reconstructed_eigenvalues[k]
        reconstructed_eigenvectors = reconstructed_eigenvectors[:,k]
        reconstructed_eigenvalues*=self.input_matrix_trace
        reconstructed_input_matrix = reconstructed_eigenvectors @ np.diag(reconstructed_eigenvalues) @ reconstructed_eigenvectors.T
        return reconstructed_input_matrix
    
    
    