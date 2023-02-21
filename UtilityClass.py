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

warnings.filterwarnings("ignore")

class QPCA():
    
    def __init__(self,resolution, seed=None):
        self.resolution=resolution
        self.seed=seed
            
    


    def generate_matrix(self,matrix_dimension,eigenvalues_list=None,replicate_paper=True):
        
        if replicate_paper == False:
            # Set dimension for the matrix
            if self.seed!=None:
                np.random.seed(self.seed)
            random_matrix=np.random.rand(matrix_dimension, matrix_dimension) 
            hermitian_matrix=np.dot(random_matrix, random_matrix.T)

            # choose eigenvalues for the matrix
            if eigenvalues_list:
                eig, e_v = np.linalg.eig(hermitian_matrix)
                eigenvalues_list=sorted(eigenvalues_list,reverse=True)
                b = np.array(eigenvalues_list)

                self.input_matrix = e_v @ np.diag(b) @ e_v.T
            else:
                self.input_matrix=hermitian_matrix

        else:
            if matrix_dimension==2:
                self.input_matrix = np.array([[1.5, 0.5],[0.5, 1.5]])
            elif matrix_dimension==4:
                self.input_matrix = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]])
            else:
                raise Exception('Only matrix 2x2 or 4x4 are allowed to replicate the reference paper')


        print(f'Matrix:\n {self.input_matrix.round(2)}\n')
        for eigenval, eigenvec in zip(np.linalg.eig(self.input_matrix)[0][::-1], np.rot90(np.linalg.eig(self.input_matrix)[1])):
            print(f'eigenvalue: {eigenval:.0f} - eigenvector: {eigenvec.round(3)}')

        return self.input_matrix




# Tommaso's great code for computing the thetas
    def thetas_computation(self,debug=False):
        lst_combination = []

        sum_squares = (self.input_matrix ** 2).sum()
        input_probabilities = (self.input_matrix ** 2 / sum_squares).flatten()

        for k in range(1, int(np.ceil(np.log2(len(self.input_matrix) ** 2))) + 1):
            lst_combination.append(list(map(list, itertools.product([0, 1], repeat=k))))
        container = []
        for lst in lst_combination:
            container.append([''.join([''.join(str(j)) for j in i]) for i in lst])
        all_combinations = [item for c in container for item in c]

        general_bitstring = [''.join([''.join(str(j)) for j in i]) for i in list(
            map(list, itertools.product([0, 1], repeat=int(np.ceil(np.log2(len(self.input_matrix) ** 2))))))][
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
        self.thetas=thetas
        self.all_combinations=all_combinations
        return thetas, all_combinations

    def generate_qram_circuit(self):
        right_nodes_indexes = list(range(0, len(self.thetas), 2))
        rotations_list = list(zip(np.array(self.all_combinations)[right_nodes_indexes], np.array(self.thetas)[right_nodes_indexes]))

        # qc=QuantumCircuit(len(example_matrix))
        qc = QuantumCircuit(int(np.ceil(np.log2(len(self.input_matrix) ** 2))))

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
                
        self.qram_circuit=qc
        
        return qc

    def generate_phase_estimation_circuit(self):
        #unitary_backend = Aer.get_backend("unitary_simulator")
        u_circuit = NumPyMatrix(self.input_matrix, evolution_time=2*np.pi/(2**self.resolution))
        pe = PhaseEstimation(self.resolution, u_circuit, name = "PE")
        #pe.decompose().draw("mpl")
        
        self.pe_circuit=pe
        return pe

    def computing_amplitudes(self):
        tot_qubit = self.pe_circuit.qregs[0].size+self.qram_circuit.qregs[0].size

        qr_total = QuantumRegister(tot_qubit, 'total')
        # classical = ClassicalRegister(4, 'measure')

        total_circuit_1 = QuantumCircuit(qr_total , name='matrix')

        total_circuit_1.append(self.qram_circuit.to_gate(), qr_total[self.pe_circuit.qregs[0].size:])
        total_circuit_1.append(self.pe_circuit.to_gate(), qr_total[0:self.pe_circuit.num_qubits])
        # total_circuit.measure(qr_total[:2], classical[:])
        #total_circuit_1.swap(qr_total[0],qr_total[1])
        total_circuit_1.measure_all()

        total_circuit_1.decompose(reps=1).draw("mpl")

        backend_total = Aer.get_backend("qasm_simulator")
        job = backend_total.run(transpile(total_circuit_1, backend=backend_total), shots=self.n_shots)
        counts = job.result().get_counts()
        #plot_histogram(counts)

        for i in counts:
            counts[i]/=self.n_shots

        statevector=np.zeros(2**tot_qubit)
        for i in counts:
            statevector[int(i,2)]=counts[i]
        self.statevector=statevector
        return statevector

    def sign_estimation(self):
        tot_qubit = self.pe_circuit.qregs[0].size+self.qram_circuit.qregs[0].size

        qr_total_xi = QuantumRegister(tot_qubit, 'xi')
        qr_total_pi = QuantumRegister(tot_qubit, 'pi')
        qr_control = QuantumRegister(1, 'control_qubit')
        n_classical_register=tot_qubit+1
        classical = ClassicalRegister(n_classical_register, 'measure')

        total_circuit_2 = QuantumCircuit(qr_total_xi,qr_total_pi,qr_control ,classical, name='matrix')

        total_circuit_2.append(self.qram_circuit.to_gate(), qr_total_xi[self.pe_circuit.qregs[0].size:])
        total_circuit_2.append(self.pe_circuit.to_gate(), qr_total_xi[0:self.pe_circuit.num_qubits])

        #total_circuit_2.swap(qr_total_xi[0],qr_total_xi[1])
        total_circuit_2.initialize(np.sqrt(self.statevector),qr_total_pi)
        total_circuit_2.h(qr_control)
        for i in range(tot_qubit):
            total_circuit_2.cswap(control_qubit=qr_control, target_qubit1=qr_total_xi[i],target_qubit2=qr_total_pi[i])

        total_circuit_2.h(qr_control)
        total_circuit_2.measure(qr_total_xi,classical[0:n_classical_register-1])
        total_circuit_2.measure(qr_control,classical[n_classical_register-1])

        total_circuit_2.draw("mpl")

        backend_total = Aer.get_backend("qasm_simulator")
        job = backend_total.run(transpile(total_circuit_2, backend=backend_total), shots=self.n_shots)
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
        for e, (count, prob) in enumerate(zip(tmp, self.statevector)):
            if count>0.4*prob*self.n_shots:
                sign=1
            else:
                sign=-1
            if prob==0:
                sign=0
            sign_dictionary.update({bin(e)[2:].zfill(tot_qubit):sign})

        statevector_dictionary={}
        for e,key in enumerate(sign_dictionary):
            statevector_dictionary[key]=sign_dictionary[key]*np.sqrt(self.statevector[e])
        self.statevector_dictionary=statevector_dictionary
        return statevector_dictionary

    def state_vector_tomography(self, n_shots=50000):
        
        self.n_shots=n_shots
        
        #1st part of tomography algorithm: amplitudes estimation
        statevector=self.computing_amplitudes()


        statevector_dictionary=self.sign_estimation()

        self.statevector_dictionary=statevector_dictionary

        return statevector_dictionary

    def eigenvectors_reconstruction(self):
        binary_lambda=[]
        for d in self.statevector_dictionary:
            if self.statevector_dictionary[d]!=0:
                binary_lambda.append(d[-self.resolution:])
        l_list=[]
        eigenvalues=[]
        for b_l in np.unique(binary_lambda):
            eigenvalues.append(int(b_l[::-1],2))
            tmp_list=[]
            for key in list(self.statevector_dictionary.keys()):
                if key[-self.resolution:]==b_l:
                    tmp_list.append(self.statevector_dictionary[key])
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
            eigenvector=np.zeros(len(self.input_matrix)) #put length of eigenvector
            save_sign=np.sign(ll)
            statevector=abs(ll)
            max_list=[]
            scaled_statevectors=[]
            for e,i in enumerate(range(0,len(statevector),len(self.input_matrix))):
                max_list.append(max(statevector[i:i+len(self.input_matrix)]))
                scaled_statevectors.append(statevector[i:i+len(self.input_matrix)]/max_list[e])
                #print(max_list,scaled_statevectors)
            idx_max=np.argmax(max_list)
            #print(idx_max)
            max_max=max_list[idx_max]
            #print(max_max)
            value=np.sqrt(max_max)
            eigenvector=scaled_statevectors[idx_max]*value*save_sign[:len(self.input_matrix)]
            eigenvectors.append((eig,eigenvector))
        
        self.reconstructed_eigenvectors=eigenvectors
        return eigenvectors
          

    def eigenvectors_benchmarking(self,print_distances=True,only_first_eigenvectors=True):
        idx=0
        for eig,eigenvector in sorted(self.reconstructed_eigenvectors,reverse=True):
            #print(eig,eigenvector)

            plt.figure()

            plt.plot(list(range(1,len(self.input_matrix)+1)),abs(eigenvector),marker='o',label='reconstructed')
            plt.plot(list(range(1,len(self.input_matrix)+1)),abs(np.linalg.eig(self.input_matrix)[1][:,idx]),marker='o',label='original')
            #print(np.linalg.norm(eigenvector-np.linalg.eig(originals)[1][:,idx]))
            plt.ylabel("eigenvector's values")
            if print_distances:
                plt.plot([], [], ' ', label="L2-norm distance: "+str(np.round(np.linalg.norm(abs(eigenvector)-abs(np.linalg.eig(self.input_matrix)[1][:,idx])),4)))
            plt.legend()

            plt.title('Eigenvectors corresponding to eigenvalues '+str(eig))
            plt.show()
            if only_first_eigenvectors:
                break
            idx+=1
            
    
    def l2_norm_benchmark(self):
        idx=0
        l2_norm_list=[]
        for eig,eigenvector in sorted(self.reconstructed_eigenvectors,reverse=True):
            
            l2_norm_list.append((eig,np.round(np.linalg.norm(abs(eigenvector)-abs(np.linalg.eig(self.input_matrix)[1][:,idx])),4)))
            idx+=1
        return l2_norm_list
        

def check_measure(arr, faster_measure_increment):
    incr = 10 + faster_measure_increment

    for i in range(len(arr) - 1):
        if arr[i + 1] == arr[i]:
            arr[i + 1] += incr
        if arr[i + 1] <= arr[i]:
            arr[i + 1] = arr[i] + incr
    return arr