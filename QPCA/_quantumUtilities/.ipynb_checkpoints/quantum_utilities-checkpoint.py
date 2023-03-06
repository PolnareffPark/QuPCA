import numpy as np
import itertools
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.circuit.library import PhaseEstimation
from qiskit import Aer, transpile, execute
import matplotlib.pyplot as plt

def thetas_computation(input_matrix,debug=False):
    
    """ Thetas computation-Preprocessing phase.

    Parameters
    ----------

    input_matrix: array-like.
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
    
def from_binary_tree_to_qcircuit(input_matrix,thetas, all_combinations):
    
    """ Preprocessing quantum circuit generation.

    Parameters
    ----------

    input_matrix: array-like.
                    Input hermitian matrix that has to be encoded in quantum circuit to perform QPCA.
                    
    thetas: array-like.
                    List of all the computed thetas for the unitary rotations R_y to encode the input matrix.
    
    all_combinations: array-like.
                    List of all the possibile combinations of bitstrings up to length log2(p*q), with p number of rows and q number of columns of the input matrix.

    Returns
    -------
    qc: QuantumCircuit. 
                    The quantum circuit that encodes the input matrix.
                    
    Notes
    -----
    This method generalize the implementation of a quantum circuit to encode a generic input matrix. It is important to note the spatial complexity of the circuit that is in the order of
    log2(p*q), with p number of rows and q number of columns of the input matrix.
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


def q_ram_pHe_quantum_circuit_generation(pe_circuit,qram_circuit):
    """
        Parameters
        ----------

        pe_circuit: QuantumCircuit. 
                    The quantum circuit that performs Phase Estimation.

        qram_circuit: QuantumCircuit. 
                    The quantum circuit that encodes the input matrix.


        Returns
        -------
        total_circuit_1: QuantumCircuit. 
                    The quantum circuit composed of the qram plus the phase estimation circuit.

        Notes
        -----
    """
    
    tot_qubit = pe_circuit.qregs[0].size+qram_circuit.qregs[0].size
    qr_total = QuantumRegister(tot_qubit, 'total')

    total_circuit_1 = QuantumCircuit(qr_total , name='matrix')

    total_circuit_1.append(qram_circuit.to_gate(), qr_total[pe_circuit.qregs[0].size:])
    total_circuit_1.append(pe_circuit.to_gate(), qr_total[0:pe_circuit.num_qubits])

    return total_circuit_1


def state_vector_tomography(quantum_circuit,n_shots,qubits_to_be_measured=None):
    """ 
        State vector tomography to estimate real vectors.

            Parameters
            ----------

            quantum_circuit: QuantumCircuit. 
                        The quantum circuit to be reconstructed. 

            n_shots: int value.
                        Number of shots to execute the quantum circuit.
                        
            qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None.
                        Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).

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

            q_size: int value. 
                        The size of the quantum register under consideration.

            c_size: int value. 
                        The size of the classical register under consideration.

            qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None.
                        Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).

            Returns
            -------
            statevector: array-like. 
                        The reconstructed statevector (without sign reconstruction).

            Notes
            -----
            This method is used to reconstruct the amplitudes of the values of the statevector that is under consideration. In addition to this function, state vector tomography also includes a sign estimation function.
            """
    
        quantum_regs_1=QuantumRegister(q_size)
            
        classical_regs_1 = ClassicalRegister(c_size, 'classical')

        new_total_circuit_1 = QuantumCircuit(quantum_regs_1,classical_regs_1)
        new_total_circuit_1.append(quantum_circuit,quantum_regs_1)
        new_total_circuit_1.measure(quantum_regs_1[qubits_to_be_measured],classical_regs_1)

        backend_total = Aer.get_backend("qasm_simulator")
        job = backend_total.run(transpile(new_total_circuit_1, backend=backend_total), shots=n_shots)
        counts = job.result().get_counts()
    
        for i in counts:
            counts[i]/=n_shots
        
        #TODO: check if put q_size or c_size
        statevector=np.zeros(2**c_size)
        for i in counts:
            statevector[int(i,2)]=counts[i]

        #self.statevector=statevector

        return statevector
    
    def sign_estimation(statevector,q_size,c_size,qubits_to_be_measured):
        
        """ This is the second and last step of the state vector tomography algorithm described in Algorithm 4.1 in "A Quantum Interior Point Method for LPs and SDPs" paper. It is
            useful to reconstruct the sign of each statevector's components.

            Parameters
            ----------
            
            statevector: array-like. 
                        The reconstructed statevector (without sign reconstruction) obtained from computing_amplitudes function.
            
            q_size: int value. 
                        The size of the quantum register under consideration.

            c_size: int value. 
                        The size of the classical register under consideration.

            qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None.
                        Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).

            Returns
            -------

            statevector_dictionary: dict-like.
                        Dictionary where the keys represent the eigenvalues/eigenvectors encoded in the qubits and the values represent the reconstructed statevector's values (with sign).

            Notes
            -----
            This method is used to reconstruct the correct sign of the statevector's values under consideration.
            """
        
        qr_total_xi = QuantumRegister(q_size, 'xi')
        qr_total_pi = QuantumRegister(q_size, 'pi')
        
        qr_control = QuantumRegister(1, 'control_qubit')
        
        n_classical_register=c_size+1
        
        classical = ClassicalRegister(n_classical_register, 'measure')

        total_circuit_2 = QuantumCircuit(qr_total_xi,qr_total_pi,qr_control ,classical, name='matrix')
        
        total_circuit_2.append(quantum_circuit,qr_total_xi)
        #index qr_total_pi[qubits_to_be_measured]
        total_circuit_2.initialize(np.sqrt(statevector),qr_total_pi[qubits_to_be_measured])
        total_circuit_2.h(qr_control)
        # qubits_to_be_measured-> range(q_size)
        for i in qubits_to_be_measured:
    
            total_circuit_2.cswap(control_qubit=qr_control, target_qubit1=qr_total_xi[i],target_qubit2=qr_total_pi[i])

        total_circuit_2.h(qr_control)
        total_circuit_2.measure(qr_total_xi[qubits_to_be_measured],classical[0:n_classical_register-1])
        total_circuit_2.measure(qr_control,classical[n_classical_register-1])
     
        backend_total = Aer.get_backend("qasm_simulator")
        job = backend_total.run(transpile(total_circuit_2, backend=backend_total), shots=n_shots)
        counts_for_sign = job.result().get_counts()

        #Take only counts with control qubits equal to 0
        tmp=np.zeros(2**c_size)
        for c in counts_for_sign:
            if c[0]=='0':
                tmp[int(c[1:],2)]=counts_for_sign[c]

        #Sign estimation
        sign_dictionary={}
        sign=0
        for e, (count, prob) in enumerate(zip(tmp, statevector)):
            if count>0.4*prob*n_shots:
                sign=1
            else:
                sign=-1
            if prob==0:
                sign=0
            sign_dictionary.update({bin(e)[2:].zfill(c_size):sign})

        statevector_dictionary={}
        for e,key in enumerate(sign_dictionary):
            statevector_dictionary[key]=sign_dictionary[key]*np.sqrt(statevector[e])
        #self.statevector_dictionary=statevector_dictionary
        return statevector_dictionary
    
    
    q_size=quantum_circuit.qregs[0].size
    if qubits_to_be_measured==None:
        c_size=q_size
        qubits_to_be_measured=list(range(q_size))
    elif isinstance(qubits_to_be_measured,int):
        c_size=1
    else:
        tmp_array=np.array(list(range(q_size)))
        c_size=len(tmp_array[qubits_to_be_measured])
    
    statevector=computing_amplitudes(q_size,c_size,qubits_to_be_measured)
    #print('statevector-wno sign',statevector)
    statevector_dictionary=sign_estimation(statevector,q_size,c_size,qubits_to_be_measured)
    
    #print('statevector',statevector_dictionary)
    return statevector_dictionary