import numpy as np
import itertools
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.circuit.library import PhaseEstimation
from qiskit import Aer, transpile, execute
import matplotlib.pyplot as plt

def generate_matrix(matrix_dimension,eigenvalues_list=None,replicate_paper=True):
    if replicate_paper == False:
        # Set dimension for the matrix
        random_matrix=np.random.rand(matrix_dimension, matrix_dimension) 
        hermitian_matrix=np.dot(random_matrix, random_matrix.T)

        # choose eigenvalues for the matrix
        if eigenvalues_list:
            eig, e_v = np.linalg.eig(hermitian_matrix)
            eigenvalues_list=sorted(eigenvalues_list,reverse=True)
            b = np.array(eigenvalues_list)

            example_matrix = e_v @ np.diag(b) @ e_v.T
        else:
            example_matrix=hermitian_matrix

    else:
        if matrix_dimension==2:
            example_matrix = np.array([[1.5, 0.5],[0.5, 1.5]])
        elif matrix_dimension==4:
            example_matrix = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]])
        else:
            raise Exception('Only matrix 2x2 or 4x4 are allowed to replicate the reference paper')
    
    
    print(f'Matrix:\n {example_matrix.round(2)}\n')
    for eigenval, eigenvec in zip(np.linalg.eig(example_matrix)[0][::-1], np.rot90(np.linalg.eig(example_matrix)[1])):
        print(f'eigenvalue: {eigenval:.0f} - eigenvector: {eigenvec.round(3)}')

    return example_matrix




# Tommaso's great code for computing the thetas
def thetas_computation(example_matrix, debug=False):
    lst_combination = []

    sum_squares = (example_matrix ** 2).sum()
    input_probabilities = (example_matrix ** 2 / sum_squares).flatten()

    for k in range(1, int(np.ceil(np.log2(len(example_matrix) ** 2))) + 1):
        lst_combination.append(list(map(list, itertools.product([0, 1], repeat=k))))
    container = []
    for lst in lst_combination:
        container.append([''.join([''.join(str(j)) for j in i]) for i in lst])
    all_combinations = [item for c in container for item in c]

    general_bitstring = [''.join([''.join(str(j)) for j in i]) for i in list(
        map(list, itertools.product([0, 1], repeat=int(np.ceil(np.log2(len(example_matrix) ** 2))))))][
                        :len(input_probabilities)]

    # Nodes contains all the values of the tree (except for the root)
    nodes = []
    for st in all_combinations:
        # print(st)
        starts = [general_bitstring.index(l) for l in general_bitstring if l.startswith(st)]
        # print(starts)
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

def generate_qram_circuit(example_matrix,thetas, all_combinations):
    right_nodes_indexes = list(range(0, len(thetas), 2))
    rotations_list = list(zip(np.array(all_combinations)[right_nodes_indexes], np.array(thetas)[right_nodes_indexes]))

    # qc=QuantumCircuit(len(example_matrix))
    qc = QuantumCircuit(int(np.ceil(np.log2(len(example_matrix) ** 2))))

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

def generate_phase_estimation_circuit(resolution,input_matrix):
    #unitary_backend = Aer.get_backend("unitary_simulator")
    u_circuit = NumPyMatrix(input_matrix, evolution_time=2*np.pi/(2**resolution))
    pe = PhaseEstimation(resolution, u_circuit, name = "PE")
    #pe.decompose().draw("mpl")

    return pe

def computing_amplitudes(matrix_circuit,pe,n_shots=5000):
    tot_qubit = pe.qregs[0].size+matrix_circuit.qregs[0].size

    qr_total = QuantumRegister(tot_qubit, 'total')
    # classical = ClassicalRegister(4, 'measure')

    total_circuit_1 = QuantumCircuit(qr_total , name='matrix')

    total_circuit_1.append(matrix_circuit.to_gate(), qr_total[pe.qregs[0].size:])
    total_circuit_1.append(pe.to_gate(), qr_total[0:pe.num_qubits])
    # total_circuit.measure(qr_total[:2], classical[:])
    #total_circuit_1.swap(qr_total[0],qr_total[1])
    total_circuit_1.measure_all()

    total_circuit_1.decompose(reps=1).draw("mpl")

    backend_total = Aer.get_backend("qasm_simulator")
    job = backend_total.run(transpile(total_circuit_1, backend=backend_total), shots=n_shots)
    counts = job.result().get_counts()
    #plot_histogram(counts)

    for i in counts:
        counts[i]/=n_shots

    statevector=np.zeros(2**tot_qubit)
    for i in counts:
        statevector[int(i,2)]=counts[i]
    
    return statevector

def sign_estimation(statevector, matrix_circuit, pe, n_shots):
    tot_qubit = pe.qregs[0].size+matrix_circuit.qregs[0].size

    qr_total_xi = QuantumRegister(tot_qubit, 'xi')
    qr_total_pi = QuantumRegister(tot_qubit, 'pi')
    qr_control = QuantumRegister(1, 'control_qubit')
    n_classical_register=tot_qubit+1
    classical = ClassicalRegister(n_classical_register, 'measure')

    total_circuit_2 = QuantumCircuit(qr_total_xi,qr_total_pi,qr_control ,classical, name='matrix')

    total_circuit_2.append(matrix_circuit.to_gate(), qr_total_xi[pe.qregs[0].size:])
    total_circuit_2.append(pe.to_gate(), qr_total_xi[0:pe.num_qubits])

    #total_circuit_2.swap(qr_total_xi[0],qr_total_xi[1])
    total_circuit_2.initialize(np.sqrt(statevector),qr_total_pi)
    total_circuit_2.h(qr_control)
    for i in range(tot_qubit):
        total_circuit_2.cswap(control_qubit=qr_control, target_qubit1=qr_total_xi[i],target_qubit2=qr_total_pi[i])

    total_circuit_2.h(qr_control)
    total_circuit_2.measure(qr_total_xi,classical[0:n_classical_register-1])
    total_circuit_2.measure(qr_control,classical[n_classical_register-1])

    total_circuit_2.draw("mpl")

    backend_total = Aer.get_backend("qasm_simulator")
    job = backend_total.run(transpile(total_circuit_2, backend=backend_total), shots=n_shots)
    counts_for_sign = job.result().get_counts()
    #plot_histogram(counts_for_sign)

    #Take only counts with control qubits equal to 0
    tmp=np.zeros(2**tot_qubit)
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
        sign_dictionary.update({bin(e)[2:].zfill(tot_qubit):sign})

    statevector_dictionary={}
    for e,key in enumerate(sign_dictionary):
        statevector_dictionary[key]=sign_dictionary[key]*np.sqrt(statevector[e])

    return statevector_dictionary

def state_vector_tomography(matrix_circuit,pe,n_shots):

    #1st part of tomography algorithm: amplitudes estimation
    statevector=computing_amplitudes(matrix_circuit=matrix_circuit,pe=pe,n_shots=n_shots)

    
    statevector_dictionary=sign_estimation(statevector=statevector,matrix_circuit=matrix_circuit,pe=pe,n_shots=n_shots)

    
    
    return statevector_dictionary

def eigenvectors_reconstruction(statevector_dictionary,resolution,input_matrix):
    binary_lambda=[]
    for d in statevector_dictionary:
        if statevector_dictionary[d]!=0:
            binary_lambda.append(d[-resolution:])
    l_list=[]
    eigenvalues=[]
    for b_l in np.unique(binary_lambda):
        eigenvalues.append(int(b_l[::-1],2))
        tmp_list=[]
        for key in list(statevector_dictionary.keys()):
            if key[-resolution:]==b_l:
                tmp_list.append(statevector_dictionary[key])
        l_list.append(np.asarray(tmp_list))
    #print(l_list)
    for l in l_list:
        normalization_factor=np.sqrt((1/(sum(l**2))))
        l*=normalization_factor
    #print(l_list)
    #TODO: Capire se fare la media tra i vari fattori di rescaling
    eigenvectors=[]
    for ll, eig in zip(l_list,eigenvalues):
        #print(ll,eig)
        eigenvector=np.zeros(len(input_matrix)) #put length of eigenvector
        save_sign=np.sign(ll)
        statevector=abs(ll)
        max_list=[]
        scaled_statevectors=[]
        for e,i in enumerate(range(0,len(statevector),len(input_matrix))):
            max_list.append(max(statevector[i:i+len(input_matrix)]))
            scaled_statevectors.append(statevector[i:i+len(input_matrix)]/max_list[e])
            #print(max_list,scaled_statevectors)
        idx_max=np.argmax(max_list)
        #print(idx_max)
        max_max=max_list[idx_max]
        #print(max_max)
        value=np.sqrt(max_max)
        eigenvector=scaled_statevectors[idx_max]*value*save_sign[:len(input_matrix)]
        eigenvectors.append((eig,eigenvector))

    return eigenvectors
        #print(eigenvector)
        #print('eigenvalue:', eig)

def eigenvectors_benchmarking(originals, reconstructed_eigenvectors,print_distances=True):
    idx=0
    for eig,eigenvector in sorted(reconstructed_eigenvectors,reverse=True):
        #print(eig,eigenvector)
        
        plt.figure()
        
        plt.plot(list(range(1,len(originals)+1)),abs(eigenvector),marker='o',label='reconstructed')
        plt.plot(list(range(1,len(originals)+1)),abs(np.linalg.eig(originals)[1][:,idx]),marker='o',label='original')
        #print(np.linalg.norm(eigenvector-np.linalg.eig(originals)[1][:,idx]))
        plt.ylabel("eigenvector's values")
        if print_distances:
            plt.plot([], [], ' ', label="L2-norm distance: "+str(np.round(np.linalg.norm(abs(eigenvector)-abs(np.linalg.eig(originals)[1][:,idx])),4)))
        plt.legend()
        
        plt.title('Eigenvectors corresponding to eigenvalues '+str(eig))
        plt.show()
        idx+=1
        
    