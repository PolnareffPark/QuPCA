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
from .._quantumUtilities.quantum_utilities import thetas_computation,from_binary_tree_to_qcircuit,state_vector_tomography,q_ram_pHe_quantum_circuit_generation
from .._postprocessingUtilities.postprocessing_eig_reconstruction import postprocessing
from .._benchmark.benchmark import _eigenvectors_benchmarking,_eigenvalues_benchmarking,_error_benchmark,_error_benchmark_from_scratch
from scipy.spatial import distance
#warnings.filterwarnings("ignore")

class QPCA():
    """Quantum Principal component analysis (QPCA).
    Implementation of the QPCA algorithm proposed in "A Low Complexity Quantum Principal Component Analysis Algorithm" paper.
    Parameters
    ----------
    input_matrix : array-like
        Input hermitian matrix on which you want to apply QPCA.
        
    resolution : int value
        Number of qubit used into the phase estimation process to encode the eigenvalues.
    
    Attributes
    ----------
    qram_circuit : QuantumCircuit 
                The quantum circuit that encodes the input matrix.
                    
    pe_circuit : QuantumCircuit. 
                The quantum circuit that performs Phase Estimation.
                
    n_shots : int value.
                Number of shots for the tomography process.
    
    reconstructed_eigenvectors : array-like.
                Array of tuples of the form [(e_1,v_1),(e_2,v_2),..] where e_s are the eigenvalues and v_s are the reconstructed eigenvectors.
    
    original_eigenValues: array-like,.
                Original eigenvalues of the input matrix.
                
    original_eigenVectors: array-like.
                Orignal eigenvectors of the input matrix.
    
    """
    
    
    def __init__(self,input_matrix,resolution):
        self.input_matrix=input_matrix
        self.resolution=resolution
        eigenValues,eigenVectors=np.linalg.eig(self.input_matrix)
        idx = eigenValues.argsort()[::-1]   
        self.original_eigenValues = eigenValues[idx]
        self.original_eigenVectors = eigenVectors[:,idx]
            

    def generate_qram_circuit(self,input_matrix):
        
        """
        Generate qram circuit.

        Parameters
        ----------

        input_matrix: array-like.
                    Input hermitian matrix that has to be encoded in quantum circuit to perform QPCA.

        Returns
        -------
        qc: QuantumCircuit. 
                    The quantum circuit that encodes the input matrix.
                    
        Notes
        -----
        This method implements the quantum circuit generation to encode a generic input matrix. It is important to note the spatial complexity of the circuit that is in the order of
        log2(p*q), with p number of rows and q number of columns of the input matrix.
        """
        
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
        -------
        pe: QuantumCircuit. 
                The quantum circuit that performs Phase Estimation.
                    
        Notes
        -----

        """
        #unitary_backend = Aer.get_backend("unitary_simulator")
        u_circuit = NumPyMatrix(self.input_matrix, evolution_time=2*np.pi/(2**self.resolution))
        pe = PhaseEstimation(self.resolution, u_circuit, name = "PE")
        #pe.decompose().draw("mpl")
        self.pe_circuit=pe
        return pe


    def eigenvectors_reconstruction(self,n_shots=50000,n_repetitions=1):
        
        """ Method that reconstructs the eigenvalues/eigenvectors once performed Phase Estimation. 

        Parameters
        ----------
        n_shots: int value, default=50000.
                Number of shots to perform in the state vector tomography function.
                
        n_repetitions: int value, default=1.
                Number of times that state vector tomography will be executed. If a value greater than 1 is passed, the final result will be the average result
                of all the execution of the tomography.

        Returns
        -------
        eigenvectors: array-like. 
                List of tuples containing as first value the reconstructed eigenvalue and as second value the reconstructed eigenvector.
                    
        Notes
        -----
        To classically reconstruct the eigenvectors, state vector tomography function is performed (implemented from algorithm 4.1 of "A Quantum Interior Point Method for LPs and SDPs" paper). In this
        way, the statevector of the quantum state is reconstructed and a postprocessing method is executed to get the eigenvectors from the reconstructed statevector.
        """
        
        def wrapper_state_vector_tomography(quantum_circuit,n_shots):
            
            assert n_repetitions>0, "n_repetitions must be greater than 0."
            self.n_shots=n_shots

            if n_repetitions==1:
                tomo_dict=state_vector_tomography(quantum_circuit,n_shots)
                #self.statevector_dictionary=tomo_dict
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
                    #self.statevector_dictionary=new_tomo_dict
                    statevector_dictionary=new_tomo_dict

            return statevector_dictionary
        
        qc=q_ram_pHe_quantum_circuit_generation(self.pe_circuit,self.qram_circuit)
        
        statevector_dictionary=wrapper_state_vector_tomography(quantum_circuit=qc,n_shots=n_shots)

        eigenvalue_eigenvector_tuple=postprocessing(input_matrix=self.input_matrix,statevector_dictionary=statevector_dictionary,resolution=self.resolution)
        self.reconstructed_eigenvalue_eigenvector_tuple=eigenvalue_eigenvector_tuple

        return eigenvalue_eigenvector_tuple
    
    def quantum_input_matrix_reconstruction(self):
        
        """ Method to reconstruct the input matrix.

        Parameters
        ----------

        Returns
        -------
        reconstructed_input_matrix: array-like. 
                Reconstructed input matrix.
                    
        Notes
        -----
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
        
        reconstructed_input_matrix = reconstructed_eigenvectors @ np.diag(reconstructed_eigenvalues) @ reconstructed_eigenvectors.T
        return reconstructed_input_matrix
    
        
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