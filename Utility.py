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
        
        
        
        
# BENCHMARKING    
    
    def spectral_benchmarking(self, eigenvector_benchmarking=True, eigenvalues_benchmarching=False,print_distances=True,only_first_eigenvectors=True,plot_delta=False,distance_type='l2'):
        
        """ Method to benchmark the reconstructed eigenvectors/eigenvalues.

        Parameters
        ----------
        eigenvector_benchmarking: bool value, default=True.
                If True, an eigenvectors benchmarking is performed to show how the quantum algorithm approximate the eigenvectors.
        
        eigenvalues_benchmarching: bool value, default=False.
                If True, an eigenvalues benchmarking is performed to show how the quantum algorithm approximate the eigenvalues.
                
        print_distances: bool value, default=True.
                If True, the distance (defined by the parameter distance_type) between the reconstructed and original eigenvectors is printed in the legend.
                
        only_first_eigenvectors: bool value, default=True.
                If True, the benchmarking is performed only for the first eigenvector. Otherwise, all the eigenvectors are considered.
                
        plot_delta: bool value, default=False.
                If True, a plot showing the trend of the tomography error is showed.
                
        distance_type: string value, default='l2'
            It defines the distance measure used to benchmark the eigenvectors:

                    -'l2': the l2 distance between original and reconstructed eigenvectors is computed.

                    -'cosine': the cosine distance between original and reconstructed eigenvectors is computed.

        Returns
        -------
        If eigenvector_benchmarking is True:
        
            - save_list: array-like. 
                List of distances between all the original and reconsructed eigenvectors.
            - delta: float value.
                The tomography error value.
                    
        Notes
        -----
        The execution of this method shows the distance between original and reconstructed eigenvector's values and allows to visualize the tomography error. In this way, you can check that the reconstruction of the eigenvectors always takes place with an error conforming to the one expressed in the tomography algorithm in the "A Quantum Interior Point Method for LPs and SDPs" paper.
        """
        
        
        if eigenvector_benchmarking:
            error_list, delta=_eigenvectors_benchmarking(reconstructed_eigenvalue_eigenvector_tuple=self.reconstructed_eigenvalue_eigenvector_tuple,
                                                         original_eigenVectors=self.original_eigenVectors,input_matrix=self.input_matrix,n_shots=self.n_shots,
                                                         print_distances=print_distances,only_first_eigenvectors=only_first_eigenvectors,plot_delta=plot_delta,distance_type=distance_type)
        if eigenvalues_benchmarching:
            _eigenvalues_benchmarking(reconstructed_eigenvalue_eigenvector_tuple=self.reconstructed_eigenvalue_eigenvector_tuple,original_eigenValues=self.original_eigenValues)
        
        if eigenvector_benchmarking:
            return error_list, delta
            
            
    def error_benchmarking(self,shots_list,errors_list=None,delta_list=None,n_tomography_repetitions=1,plot_delta=False,distance_type='l2'):
        
        if errors_list==None:
            
            delta_list=[]
            ll=[]
            for s in shots_list:
                if plot_delta:
                    delta_error = (np.sqrt((36*len(self.original_eigenVectors[:,0])*np.log(len(self.original_eigenVectors[:,0])))/(s)))
                    delta_list.append(delta_error)

                reconstructed_eigenvectors=self.eigenvectors_reconstruction(n_shots=s,n_repetitions=n_tomography_repetitions)
                eig_count=0  
                for eigenvalue, eigenvector in reconstructed_eigenvectors:
                    distance=distance_function_wrapper(distance_type,abs(eigenvector),abs(self.original_eigenVectors[:,eig_count]))
                    ll.append((eigenvalue,np.round(distance,4)))
                    eig_count+=1
    
            dict__ = {k: [v for k1, v in ll if k1 == k] for k, v in ll}
            
            _error_benchmark_from_scratch(original_eigenVectors=self.original_eigenVectors,shots_list=shots_list,error_dict=dict__,plot_delta=plot_delta,label_error=distance_type,delta_list=delta_list)
        else:
            
            _error_benchmark(original_eigenVectors=self.original_eigenVectors,shots_list=shots_list,errors_list=errors_list,delta_list=delta_list,plot_delta=plot_delta,
                         label_error=distance_type,n_tomography_repetitions=n_tomography_repetitions)
    
    
    '''def eigenvectors_benchmarking(self,print_distances=True,only_first_eigenvectors=True,plot_delta=False,distance_type='l2'):
        
        """ Method to benchmark the quality of the reconstructed eigenvectors.

        Parameters
        ----------
        print_distances: bool value, default=True.
                If True, the distance (defined by distance_type value) between the original and reconstructed eigenvector is printed in the legend.
                
        only_first_eigenvectors: bool value, default=True.
                If True, the function returns only the plot relative to the first eigenvalues. Otherwise, all the plot are showed.
                
        plot_delta: bool value, default=False.
                If True, the function also returns the plot that shows how the tomography error decreases as the number of shots increases. 
        
        distance_type: string value, default='l2'
                It defines the distance measure used to benchmark the eigenvectors:
                
                        -'l2': the l2 distance between original and reconstructed eigenvectors is computed.
                        
                        -'cosine': the cosine distance between original and reconstructed eigenvectors is computed.

        Returns
        -------
        save_list: array-like. 
                List of distances between all the original and reconsructed eigenvectors.
        delta: float value.
                The tomography error value.
                    
        Notes
        -----
        The execution of this method shows the distance between original and reconstructed eigenvector's values and allows to visualize the tomography error. In this way, you can check that the reconstruction of the eigenvectors always takes place with an error conforming to the one expressed in the tomography algorithm in the "A Quantum Interior Point Method for LPs and SDPs" paper.
        """
        
        #global eigenvalues_reconstructed
        
        save_list=[]
        fig, ax = plt.subplots(1,len(self.reconstructed_eigenvalue_eigenvector_tuple),figsize=(20, 15))
        for e,chart in enumerate(ax.reshape(-1,order='F')):
            delta=np.sqrt((36*len(self.original_eigenVectors[:,e%len(self.input_matrix)])*np.log(len(self.original_eigenVectors[:,e%len(self.input_matrix)])))/(self.n_shots))
            
            if plot_delta:
                
                for i in range(len(self.original_eigenVectors[:,(e%len(self.input_matrix))])):
                    circle=plt.Circle((i+1,abs(self.original_eigenVectors[:,e%len(self.input_matrix)])[i]),np.sqrt(7)*delta,color='g',alpha=0.1)
                    chart.add_patch(circle)
                    chart.axis("equal")
                    chart.hlines(abs(self.original_eigenVectors[:,e%len(self.input_matrix)])[i],xmin=i+1,xmax=i+1+(np.sqrt(7)*delta))
                    chart.text(i+1+((i+1+(np.sqrt(7)*delta))-(i+1))/2,abs(self.original_eigenVectors[:,e%len(self.input_matrix)])[i]+0.01,r'$\sqrt{7}\delta$')
                chart.plot([], [], ' ', label=r'$\delta$='+str(round(delta,4)))
                chart.plot(list(range(1,len(self.input_matrix)+1)),abs(self.reconstructed_eigenvalue_eigenvector_tuple[e%len(self.input_matrix)][1]),marker='*',label='reconstructed',linestyle='None',markersize=12,alpha=0.5,color='r')
                chart.plot(list(range(1,len(self.input_matrix)+1)),abs(self.original_eigenVectors[:,e%len(self.input_matrix)]),marker='o',label='original',linestyle='None',markersize=12,alpha=0.4)

            else:
                chart.plot(list(range(1,len(self.input_matrix)+1)),abs(self.reconstructed_eigenvalue_eigenvector_tuple[e%len(self.input_matrix)][1]),marker='o',label='reconstructed')
                chart.plot(list(range(1,len(self.input_matrix)+1)),abs(self.original_eigenVectors[:,e%len(self.input_matrix)]),marker='o',label='original')
            
            if print_distances:
                distance=distance_function_wrapper(distance_type,abs(self.reconstructed_eigenvalue_eigenvector_tuple[e%len(self.input_matrix)][1]),abs(self.original_eigenVectors[:,e%len(self.input_matrix)]))
                chart.plot([], [], ' ', label=distance_type+"_error "+str(np.round(distance,4)))
                
            save_list.append((self.reconstructed_eigenvalue_eigenvector_tuple[e%len(self.input_matrix)][0],np.round(distance,4)))
            chart.plot([], [], ' ', label="n_shots "+str(self.n_shots))
            chart.legend()
            chart.set_ylabel("eigenvector's values")
            chart.set_title('Eigenvectors corresponding to eigenvalues '+str(self.reconstructed_eigenvalue_eigenvector_tuple[e%len(self.input_matrix)][0]))
            if only_first_eigenvectors:
                break
           
        fig.tight_layout()
        plt.show()
        
        return save_list,delta
    
    def eigenvalues_benchmarking(self):
        """ Method to benchmark the quality of the reconstructed eigenvalues. 

        Parameters
        ----------
        
        Returns
        -------
                    
        Notes
        -----
        """
    
        fig, ax = plt.subplots(figsize=(20, 15))

        eigenvalues=[i[0] for i in self.reconstructed_eigenvalue_eigenvector_tuple]
        ax.plot(list(range(1,len(eigenvalues)+1)),eigenvalues,marker='o',label='reconstructed',linestyle='None',markersize=25,alpha=0.3,color='r')
        ax.plot(list(range(1,len(self.original_eigenValues)+1)),self.original_eigenValues,marker='x',label='original',linestyle='None',markersize=20,color='black')
        ax.legend(labelspacing = 3)
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Matching between original and reconstructed eigenvalues')
        
        plt.show()
        
            

    
    
    def error_benchmark(self,shots_list,errors_list=None,delta_list=None,plot_delta=False,label_error='l2',n_tomography_repetitions=1):
        
        """ Method to benchmark the eigenvector's reconstruction error. The execution of this function shows the trend of the error as the number of shots increases.

        Parameters
        ----------
        shots_list: array-like.
                List that contains the shots that you want to perform in the tomography to evaluate the trend of the reconstruction error.
                
        errors_list: array-like, default=None.
                List of tuples that contains the reconstruction error of each eigenvector and the corresponding eigenvalue. It can be retrieved using the function eigenvectors_benchmarking(). If not provided, the reconstruction errors are computed by scratch using the internal_error_benchmarking. 
                
        delta_list: array-like, default=None.
                List that contains all the tomography error computed for each different number of shots. If None, the plot of the tomography error is not showed. 
        
        plot_delta: bool value, default=False.
                If True, the function also returns the plot that shows how the tomography error decreases as the number of shots increases. 
        
        label_error: string value, default='l2'
                It defines the distance measure used to benchmark the eigenvectors:
                
                        -'l2': the l2 distance between original and reconstructed eigenvectors is computed.
                        
                        -'cosine': the cosine distance between original and reconstructed eigenvectors is computed.
        
        n_tomography_repetitions: int value, default=1.
                Number of times that state vector tomography will be executed. If a value greater than 1 is passed, the final result will be the average result
                of all the execution of the tomography.
        
        Returns
        -------
                    
        Notes
        -----
        """
        
        def internal_error_benchmark():
            delta_list=[]
            ll=[]
            
            for s in shots_list:
                if plot_delta:
                    delta_error = (np.sqrt((36*len(self.original_eigenVectors[:,0])*np.log(len(self.original_eigenVectors[:,0])))/(s)))
                    delta_list.append(delta_error)

                
                reconstructed_eigenvectors=self.eigenvectors_reconstruction(n_shots=s,n_repetitions=n_tomography_repetitions)
                eig_count=0  
                for eigenvalue, eigenvector in reconstructed_eigenvectors:
                    distance=distance_function_wrapper(label_error,abs(eigenvector),abs(self.original_eigenVectors[:,eig_count]))
                    ll.append((eigenvalue,np.round(distance,4)))
                    eig_count+=1

            dict__ = {k: [v for k1, v in ll if k1 == k] for k, v in ll}

            fig, ax = plt.subplots(1,len(dict__),figsize=(25, 10))
            for e,chart in enumerate(ax.reshape(-1)):
                chart.plot(shots_list,dict__[list(dict__.keys())[e]],'-o')
                chart.set_xticks(shots_list)
                chart.set_xscale('log')
                chart.set_xlabel('n_shots')
                chart.set_ylabel(label_error+'_error')
                chart.set_title(label_error+'_error for eigenvector wrt the eigenvalues {}'.format(list(dict__.keys())[e]))

            fig.tight_layout()
            fig, ax = plt.subplots(figsize=(25, 10))
            if plot_delta:
                if delta_list==None:
                     raise Exception('You need to provide a delta error list!')
                else:
                    plt.plot(shots_list,delta_list,'-o')
                    plt.xticks(shots_list)
                    plt.xscale('log')
                    plt.xlabel('n_shots')
                    plt.ylabel(r'$\delta$')
                    plt.title(r'Tomography error')
            #fig.tight_layout()
            plt.show()
        
        if errors_list==None:
            internal_error_benchmark()
        else:
            if plot_delta==False and delta_list:
                warnings.warn("Attention! delta_list that you passed has actually no effect since the flag plot_delta is set to False. Please set it to True if you want to get the delta decreasing plot.")
     
            e_list=[sub for e in errors_list for sub in e]
            dict_ = {k: [v for k1, v in e_list if k1 == k] for k, v in e_list}
            
            fig, ax = plt.subplots(1,len(dict_),figsize=(25, 10))
            fig.tight_layout()
            for e,chart in enumerate(ax.reshape(-1)):
                chart.plot(shots_list,dict_[list(dict_.keys())[e]],'-o')
                chart.set_xticks(shots_list)
                chart.set_xscale('log')
                chart.set_xlabel('n_shots')
                chart.set_ylabel(label_error+'_error')
                chart.set_title(label_error+'_error for eigenvector wrt the eigenvalues {}'.format(list(dict_.keys())[e]))
            
            fig, ax = plt.subplots(figsize=(25, 10))
            if plot_delta:
                if delta_list==None:
                     raise Exception('You need to provide a delta error list!')
                else:
                    plt.plot(shots_list,delta_list,'-o')
                    plt.xticks(shots_list)
                    plt.xscale('log')
                    plt.xlabel('n_shots')
                    plt.ylabel(r'$\delta$')
                    plt.title(r'Tomography error')
            
            plt.show()
'''
    
    
        
    #TODO: rifinire metodo     
    def runtime_comparison(self, max_n_samples, n_features,classical_principal_components,eps,rand_PCA=True):
        
        #TODO: aggiustare il condition number creando una matrice random per ogni dimensionalità diversa e calcolando il condition number di conseguenza.
        
        n=np.linspace(1, max_n_samples, dtype=np.int64, num=100)
        zoomed=False
        if rand_PCA:
            classical_complexity=n*n_features*np.log(classical_principal_components)
            label_PCA='randomized PCA'
        else:
            classical_complexity=(n*(n_features**2)+n_features**3)
            label_PCA='full PCA'
            

        delta=np.sqrt((36*n_features*np.log(n_features))/self.n_shots)
        martrix_encoding_complexity=np.log(n*n_features)
        #pe_complexity=(((np.linalg.cond(self.input_matrix)/eps))**2)*(1/eps)*np.log(n*n_features)
        
        A=[np.random.rand(i,n_features) for i in n]
        pe_complexity=[]
        cond_number_list=[]
        for i in range(100):
            cond_number_list.append(np.linalg.cond(A[i]))
        cond_number=np.mean(cond_number_list)
            
        #pe_complexity.append((((np.linalg.cond(A[i])/eps))**2)*(1/eps)*np.log(n[i]*n_features))
        pe_complexity=(((cond_number/eps))**2)*(1/eps)*np.log(n*n_features)
        #pe_complexity=np.array(pe_complexity)
        tomography_complexity=n_features/(delta**2)
        
        #print('delta',delta)
        #print(np.linalg.cond(self.input_matrix))
        
        total_complexity=martrix_encoding_complexity+pe_complexity+tomography_complexity
        
        #print(total_complexity,classical_complexity)

        # Base plot
        fig, ax = plt.subplots(figsize=[10,9])

        plt.plot(n,classical_complexity,color='red',label='classical_runtime')
        plt.plot(n,total_complexity,color='green',label='quantum_runtime')
        plt.plot([],[],' ',label=r'$\epsilon=$'+str(eps))
        plt.plot([],[],' ',label='classical PC retained='+str(classical_principal_components))
        plt.plot([],[],' ',label='n_features='+str(n_features))
        xticks=plt.xticks()[0]
        #print(plt.xticks()[0])
        plt.xlabel('n_samples')
        plt.ylabel('runtime')
        plt.title('Runtime comparison between quantum and '+label_PCA)
        idx_ = np.argwhere(np.diff(np.sign(classical_complexity-total_complexity))).flatten()
        #print(n[idx_])
        '''if len(n[idx_])>0:
            
            order_of_magnitude_diff=math.floor(math.log(n[-1], 10))- math.floor(math.log(n[idx_], 10))
            print(order_of_magnitude_diff)
            if order_of_magnitude_diff>1 and math.floor(math.log(n[idx_], 10))!=0:
                zoomed=True
                #Zoomed plot
                n_zoomed=np.linspace(1, n[idx_][0]*2, dtype=np.int64, num=100)
                if rand_PCA:
                    classical_complexity2=n_zoomed*n_features*np.log(classical_principal_components)
                else:
                    #TODO: define full pca complexity
                    classical_complexity2=(n_zoomed*(n_features**2)+n_features**3)

                #classical_complexity2=n_zoomed*n_features*np.log(classical_principal_components)
                quantum_complexity_2=np.log(n_zoomed*n_features)+(((np.linalg.cond(self.input_matrix)/eps))**2)*(1/eps)*np.log(n_zoomed*n_features)+(n_features/(delta**2))

                #find intersection of two zoomed line
                idx = np.argwhere(np.diff(np.sign(classical_complexity2-quantum_complexity_2))).flatten()

                ax2 = plt.axes([.65, .3, .2, .2])
                plt.plot(n_zoomed,quantum_complexity_2,color='green')
                plt.plot(n_zoomed,classical_complexity2,color='red')
                ax2.vlines(n_zoomed[idx],ymin=0,ymax=quantum_complexity_2[idx],linestyle='--',color='black',alpha=0.3)
                plt.setp(ax2,xticks=[],yticks=[])
                ax2.set_xticks(n_zoomed[idx])
                ax2.set_yticks(classical_complexity2[idx])
                #ax2.yaxis.get_major_locator().set_params(nbins=3)
                #ax2.xaxis.get_major_locator().set_params(nbins=3)
                #x1, x2, y1, y2 = .65, .6, .2, 2
                #ax2.set_xlim(x1, x2)
                #ax2.set_ylim(y1, y2)

                #plt.xticks(visible=True)
                #plt.yticks(visible=True)

                mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec="0.7",linestyle='--')'''
        #print(np.array(xticks)[1:-1],np.array(xticks)[1:-1]+n[idx_],)
        #if zoomed==False:
        ax.vlines(n[idx_],ymin=0,ymax=total_complexity[idx_],linestyle='--',color='black',alpha=0.3)
        ax.set_xticks(np.append(np.array(xticks)[1:-1],n[idx_]))
            #ax.set_yticks(classical_complexity[idx_])
            
        ax.legend()
        plt.show()
    

def check_measure(arr, faster_measure_increment):
    incr = 10 + faster_measure_increment

    for i in range(len(arr) - 1):
        if arr[i + 1] == arr[i]:
            arr[i + 1] += incr
        if arr[i + 1] <= arr[i]:
            arr[i + 1] = arr[i] + incr
    return arr

def distance_function_wrapper(distance_type, *args):
    #print(*args,*args[0])
    reconstructed_eig= args[0]
    original_eig = args[1]
    if distance_type=='l2':
        return np.linalg.norm(reconstructed_eig-original_eig)
    if distance_type=='cosine':
        return distance.cosine(reconstructed_eig,original_eig)
        
    