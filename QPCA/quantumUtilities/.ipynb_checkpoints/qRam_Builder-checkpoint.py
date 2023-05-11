from ..quantumUtilities.quantum_utilities import thetas_computation,from_binary_tree_to_qcircuit

class QramBuilder():
    
    @classmethod
    def generate_qram_circuit(class_,input_matrix):
        """
        Generate qram circuit.

        Parameters
        ----------
        
        input_matrix: array-like of shape (n_samples, n_features)
                        Input hermitian matrix on which you want to apply QPCA divided by its trace, where `n_samples` is the number of samples
                        and `n_features` is the number of features.

        Returns
        ----------
        qc: QuantumCircuit 
                    The quantum circuit that encodes the input matrix.
                    
        Notes
        ----------
        This method implements the quantum circuit generation to encode a generic input matrix. It is important to note the spatial complexity of the circuit that is in the order of
        log2(n_samples, n_features).
        """
        
        thetas, all_combinations=thetas_computation(input_matrix=input_matrix)
        
        qc=from_binary_tree_to_qcircuit(input_matrix,thetas, all_combinations)
        
        return qc

    