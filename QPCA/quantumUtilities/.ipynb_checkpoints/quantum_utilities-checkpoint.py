import numpy as np
import itertools
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def thetas_computation(input_matrix,debug=False):
    
    """ Thetas computation-Preprocessing phase.

    Parameters
    ----------

    input_matrix: array-like of shape (n_samples, n_features)
                    Input hermitian matrix that has to be encoded in quantum circuit to perform QPCA.
                    
    debug: bool, default=False.
                    If True, print the amplitudes for each state. If False, nothing happens.

    Returns
    -------
    thetas: array-like. 
                    List of all the computed thetas for the unitary rotations R_y to encode the input matrix.
                    
    all_combinations: array-like.
                    List of all the possibile combinations of bitstrings up to length log2(p*q), with p number of rows and q number of columns of the input matrix.

    Notes
    -----
    This method is part of the preprocessing stage to compute all the thetas (angles) necessary to perform the unitary rotations in the circuit to store the input matrix.
    """
    
    lst_combination = []

    sum_squares = (input_matrix ** 2).sum()
    input_probabilities = (input_matrix ** 2 / sum_squares).flatten()

    for k in range(1, int(np.ceil(np.log2(len(input_matrix) ** 2))) + 1):
        lst_combination.append(list(map(list, itertools.product([0, 1], repeat=k))))
    container = []
    for lst in lst_combination:
        container.append([''.join([''.join(str(j)) for j in i]) for i in lst])
    all_combinations = [item for c in container for item in c]

    general_bitstring = [''.join([''.join(str(j)) for j in i]) for i in list(
        map(list, itertools.product([0, 1], repeat=int(np.ceil(np.log2(len(input_matrix) ** 2))))))][
                        :len(input_probabilities)]

    # Nodes contains all the values of the tree (except for the root)
    
    nodes = []
    for st in all_combinations:
        starts = [general_bitstring.index(l) for l in general_bitstring if l.startswith(st)]
        if debug == True:
            print(st, '->', np.sqrt(input_probabilities[starts].sum()))
        nodes.append(np.sqrt(input_probabilities[starts].sum()))

    # add root tree
    
    nodes.insert(0, 1)
    thetas = []

    idx = 0
    
    #compute all the thetas value for each node
    
    for i in range(1, len(nodes), 2):

        right_node = i
        left_node = right_node + 1
        if nodes[idx] != 0:
            thetas.append(2 * np.arccos(nodes[right_node] / nodes[idx]))
            thetas.append(2 * np.arcsin(nodes[left_node] / nodes[idx]))
        else:
            thetas.append(0)
            thetas.append(0)

        idx += 1

    return thetas, all_combinations
    
def from_binary_tree_to_qcircuit(input_matrix, thetas, all_combinations):
    
    """ Preprocessing quantum circuit generation.

    Parameters
    ----------

    input_matrix: array-like of shape (n_samples, n_features)
                    Input hermitian matrix that has to be encoded in quantum circuit to perform QPCA.
                    
    thetas: array-like
                    List of all the computed thetas for the unitary rotations R_ys to encode the input matrix.
    
    all_combinations: array-like
                    List of all the possibile combinations of bitstrings up to length log2(n_samples*n_features)

    Returns
    -------
    qc: QuantumCircuit. 
                    The quantum circuit that encodes the input matrix.
                    
    Notes
    -----
    This method generalize the implementation of a quantum circuit to encode a generic input matrix. It is important to note the spatial complexity of the circuit that is in the order of
    log2(n_samples*n_features).
    """
    
    right_nodes_indexes = list(range(0, len(thetas), 2))
    rotations_list = list(zip(np.array(all_combinations)[right_nodes_indexes], np.array(thetas)[right_nodes_indexes]))

    qc = QuantumCircuit(int(np.ceil(np.log2(len(input_matrix) ** 2))))

    for r_l in rotations_list:
        target_qubit = len(r_l[0]) - 1

        # First case of R_0
        if target_qubit == 0:
            qc.ry(theta=r_l[1], qubit=target_qubit)
            continue
        not_gate = []
        
        #if the first qubit is 0, we put the corresponding rotation into not_gate list to remember that we have to insert an X gate before and after the control of that rotation 
        
        for qb in range(target_qubit):
            if r_l[0][qb] == '0':
                not_gate.append(qb)
        c_t_qubits = list(range(len(r_l[0])))
        n_controls = len(range(target_qubit))
        
        if len(not_gate) > 0:
            qc.x(not_gate)
            c_ry = RYGate(r_l[1]).control(n_controls)
            qc.append(c_ry, c_t_qubits)
            qc.x(not_gate)
        else:
            c_ry = RYGate(r_l[1]).control(n_controls)
            qc.append(c_ry, c_t_qubits)
    return qc


