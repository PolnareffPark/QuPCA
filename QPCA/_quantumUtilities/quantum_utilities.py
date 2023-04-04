import numpy as np
import itertools
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.circuit.library import PhaseEstimation
from qiskit import Aer, transpile, execute
import matplotlib.pyplot as plt
from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation

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


def state_vector_tomography(quantum_circuit,n_shots,qubits_to_be_measured=None,backend=None,drawing_circuit=False):
    """ 
        State vector tomography to estimate real vectors.

        Parameters
        ----------

        quantum_circuit: QuantumCircuit 
                    The quantum circuit to be reconstructed. 

        n_shots: int value
                    Number of measures performed in the tomography process.
                    
        qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None.
                    Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).
        
        backend: Qiskit backend, default value=None.
                    The Qiskit backend used to execute the circuit. If None, the qasm simulator is used by default.
        
        drawing_circuit: bool value, default=False.
                    If True, a drawing of the tomography circuit is displayed. Otherwise, only the reconstructed statevector is returned.

        Returns
        -------
        statevector_dictionary: dict-like. 
                    The reconstructed statevector of the input quantum circuit.

        Notes
        -----
        This method reconstruct the real statevector of the input quantum circuit. It is an implementation of the Algorithm 4.1 in "A Quantum Interior Point Method for LPs and SDPs" paper, and it is composed of 
        two parts: amplitudes estimation and sign estimation.
    """
    
    def computing_amplitudes(q_size,c_size,qubits_to_be_measured):
        
        """ This is the first step of the state vector tomography algorithm described in Algorithm 4.1 in "A Quantum Interior Point Method for LPs and SDPs" paper. It is
            useful to reconstruct each statevector components.

            Parameters
            ----------

            q_size: int value
                        The size of the quantum register under consideration.

            c_size: int value
                        The size of the classical register under consideration.

            qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None
                        Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).

            Returns
            -------
            probabilities: array-like
                        The reconstructed probabilities statevector (without sign reconstruction).

            Notes
            -----
            This method is used to reconstruct the amplitudes of the values of the statevector that is under consideration. In addition to this function, state vector tomography also includes a sign estimation function.
            """
        #TODO: check if put q_size or c_size
        probabilities=np.zeros(2**c_size)
        quantum_regs_1=QuantumRegister(q_size)
        classical_regs_1 = ClassicalRegister(c_size, 'classical')
        tomography_circuit_1 = QuantumCircuit(quantum_regs_1,classical_regs_1)
        tomography_circuit_1.append(quantum_circuit,quantum_regs_1)
        tomography_circuit_1.measure(quantum_regs_1[qubits_to_be_measured],classical_regs_1)        
        job = backend.run(transpile(tomography_circuit_1, backend=backend), shots=n_shots)
        counts = job.result().get_counts()
        
        for i in counts:
            counts[i]/=n_shots
            probabilities[int(i,2)]=counts[i]
        
        return probabilities
    
    def sign_estimation(probabilities,q_size,c_size,qubits_to_be_measured):
        
        """ This is the second and last step of the state vector tomography algorithm described in Algorithm 4.1 in "A Quantum Interior Point Method for LPs and SDPs" paper. It is
            useful to reconstruct the sign of each statevector's components.

            Parameters
            ----------
            
            probabilities: array-like 
                        The reconstructed probabilities statevector (without sign reconstruction) obtained from computing_amplitudes function.
            
            q_size: int value 
                        The size of the quantum register under consideration.

            c_size: int value 
                        The size of the classical register under consideration.

            qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None
                        Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).

            Returns
            -------

            statevector_dictionary: dict-like
                        Dictionary where the keys represent the eigenvalues/eigenvectors encoded in the qubits and the values represent the reconstructed statevector's values (with sign).

            Notes
            -----
            This method is used to reconstruct the correct sign of the statevector's values under consideration.
            """
        
        
        qr_total_xi = QuantumRegister(q_size, 'xi')
        n_classical_register=c_size+1
        classical_registers=ClassicalRegister(n_classical_register,'classical')
        qr_control = QuantumRegister(1, 'control_qubit')
        op_U=quantum_circuit.to_gate(label='op_U').control()
        op_V = StatePreparation(np.sqrt(probabilities),label='op_V').control()

        tomography_circuit_2 = QuantumCircuit(qr_total_xi,qr_control, classical_registers,name='matrix')
        tomography_circuit_2.h(qr_control)
        tomography_circuit_2.x(qr_control)
        tomography_circuit_2.append(op_U, qr_control[:]+qr_total_xi[:])
        tomography_circuit_2.x(qr_control)
        tomography_circuit_2.append(op_V, qr_control[:]+qr_total_xi[qubits_to_be_measured])
        tomography_circuit_2.h(qr_control)
        tomography_circuit_2.measure(qr_total_xi[qubits_to_be_measured],classical_registers[0:n_classical_register-1])
        tomography_circuit_2.measure(qr_control,classical_registers[n_classical_register-1])
    
        if drawing_circuit:
            display(tomography_circuit_2.draw('mpl'))

        job = backend.run(transpile(tomography_circuit_2, backend=backend), shots=n_shots)
        counts_for_sign = job.result().get_counts()
        tmp=np.zeros(2**c_size)
        for c in counts_for_sign:
            if c[0]=='0':
                tmp[int(c[1:],2)]=counts_for_sign[c]
        sign_dictionary={}
        sign=0
        for e, (count, prob) in enumerate(zip(tmp, probabilities)):
            if count>0.4*prob*n_shots:
                sign=1
            else:
                sign=-1
            if prob==0:
                sign=1
            sign_dictionary.update({bin(e)[2:].zfill(c_size):sign})

        statevector_dictionary={}
        for e,key in enumerate(sign_dictionary):
            statevector_dictionary[key]=sign_dictionary[key]*np.sqrt(probabilities[e])
        return statevector_dictionary
    
    if backend==None:
            backend = Aer.get_backend("qasm_simulator")
            
    q_size=quantum_circuit.qregs[0].size
    if qubits_to_be_measured==None:
        c_size=q_size
        qubits_to_be_measured=list(range(q_size))
    elif isinstance(qubits_to_be_measured,int):
        c_size=1
    else:
        tmp_array=np.array(list(range(q_size)))
        c_size=len(tmp_array[qubits_to_be_measured])
    
    probabilities=computing_amplitudes(q_size,c_size,qubits_to_be_measured)
    statevector_dictionary=sign_estimation(probabilities,q_size,c_size,qubits_to_be_measured)

    return statevector_dictionary