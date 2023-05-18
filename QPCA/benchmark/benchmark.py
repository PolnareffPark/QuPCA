import numpy as np
import warnings
import matplotlib.pyplot as plt
from texttable import Texttable
from scipy.spatial import distance 
from .benchmark_utility import *

class Benchmark_Manager():
    
    def __init__(self,eigenvector_benchmarking_=False, eigenvalues_benchmarching_=False, sign_benchmarking_=False, print_distances_=False, 
                 only_first_eigenvectors_=False, plot_delta_=False, distance_type_=None, error_with_sign_=False, hide_plot_=False, print_error_=False):
        
        self.eigenvector_benchmarking=eigenvector_benchmarking_
        self.eigenvalues_benchmarching=eigenvalues_benchmarching_
        self.sign_benchmarking=sign_benchmarking_
        self.print_distances=print_distances_
        self.only_first_eigenvectors=only_first_eigenvectors_
        self.plot_delta=plot_delta_
        self.distance_type=distance_type_
        self.error_with_sign=error_with_sign_
        self.hide_plot=hide_plot_
        self.print_error=print_error_
        
                    
    def benchmark(self, input_matrix_=None,reconstructed_eigenvalues_=None, reconstructed_eigenvectors_=None, mean_threshold_=None, n_shots_=1000):
        
        eigenValues,eigenVectors=np.linalg.eig(input_matrix_)
        idx = eigenValues.argsort()[::-1]   
        original_eigenvalues_ = eigenValues[idx]
        original_eigenvectors_ = eigenVectors[:,idx]
        
        returning_results_wrapper=[]
        
        if self.eigenvector_benchmarking:
            
            error_list,delta=self.__eigenvectors_benchmarking(input_matrix=input_matrix_, original_eigenvalues=original_eigenvalues_, original_eigenvectors=original_eigenvectors_,
                                                            reconstructed_eigenvalues=reconstructed_eigenvalues_, reconstructed_eigenvectors=reconstructed_eigenvectors_,
                                                            mean_threshold=mean_threshold_, n_shots=n_shots_,print_distances=self.print_distances, only_first_eigenvectors=self.only_first_eigenvectors,
                                                            plot_delta=self.plot_delta,distance_type=self.distance_type,error_with_sign=self.error_with_sign,hide_plot=self.hide_plot)
            returning_results_wrapper.append([error_list,delta])
            
        if self.eigenvalues_benchmarching:
            _=self.__eigenvalues_benchmarking(original_eigenvalues=original_eigenvalues_, reconstructed_eigenvalues=reconstructed_eigenvalues_,
                                            mean_threshold=mean_threshold_, print_error=self.print_error)

        
        if self.sign_benchmarking:
            _=self.__sign_reconstruction_benchmarking(input_matrix=input_matrix_, original_eigenvalues=original_eigenvalues_, original_eigenvectors=original_eigenvectors_,
                                                      reconstructed_eigenvalues=reconstructed_eigenvalues_, reconstructed_eigenvectors=reconstructed_eigenvectors_, mean_threshold=mean_threshold_,n_shots=n_shots_)
            
    
        return returning_results_wrapper     
            
    
    @classmethod
    def error_benchmark(self, input_matrix, shots_dict, error_dict, label_error='l2'):

        """ Method to benchmark the eigenvector's reconstruction error. The execution of this function shows the trend of the error as the number of shots increases.

        Parameters
        ----------

        shots_dict: dict-like
                Dictionary that contains as keys the reconstructed eigenvalues and as values the list of shots for which you are able to reconstruct the corresponding eigenvalue.

        error_dict: dict-like
                Dictionary that contains as keys the reconstructed eigenvalues and as values the list of reconstruction errors for the corresponding eigenvectors as the number of shots increases.

        dict_original_eigenvalues: dict-like
                Dictionary that contains as key the original eigenvalue and as value its index (ordered position).

        label_error: string value, default='l2'
                It defines the distance measure used to benchmark the eigenvectors:

                        -'l2': the l2 distance between original and reconstructed eigenvectors is computed.

                        -'cosine': the cosine distance between original and reconstructed eigenvectors is computed.

        Returns
        -------

        Notes
        -----
        """
        
        eigenValues,eigenVectors=np.linalg.eig(input_matrix/np.trace(input_matrix))
        idx = eigenValues.argsort()[::-1]   
        original_eigenvalues = eigenValues[idx]
        original_eigenvectors = eigenVectors[:,idx]
        dict_original_eigenvalues={}
        
        # dictionary useful to make comparison betweeen the correct eigenvectors

        for i in range(len(original_eigenvalues)):
            dict_original_eigenvalues.update({original_eigenvalues[i]:i})

        fig, ax = plt.subplots(1,len(dict_original_eigenvalues),figsize=(30, 10))
        fig.tight_layout()
        color = {'blue','red','green'}
        for res,c in zip(error_dict,color):

            e_list=[sub for e_ in error_dict[res] for sub in e_]

            dict__ = {k: [v for k1, v in e_list if k1 == k] for k, v in e_list}

            tmp_dict={}
            tmp_shots_list={}
            for key,value in dict__.items():

                x,min_=find_nearest(original_eigenvalues,key)
                tmp_dict.setdefault(x, [])
                tmp_shots_list.setdefault(x, [])
                tmp_shots_list[x]+=shots_dict[res][key]
                tmp_dict[x]+=value
            for k,j in tmp_dict.items():
                idx=dict_original_eigenvalues[k]
                ax[idx].plot(tmp_shots_list[k],j,'-o',c=c,label='resolution '+str(res))
                ax[idx].set_xticks(tmp_shots_list[k])
                ax[idx].set_xscale('log')
                ax[idx].set_xlabel('n_shots')
                ax[idx].set_ylabel(label_error+'_error')
                ax[idx].set_title(label_error+'_error for eigenvector wrt the eigenvalues {}'.format(k))
                ax[idx].legend()

        plt.show()
    
    def __eigenvectors_benchmarking(self,input_matrix, original_eigenvectors, original_eigenvalues, reconstructed_eigenvalues, reconstructed_eigenvectors, mean_threshold, 
                                  n_shots, print_distances, only_first_eigenvectors, plot_delta, distance_type, error_with_sign, hide_plot):

        """ Method to benchmark the quality of the reconstructed eigenvectors.

        Parameters
        ----------

        input_matrix: array-like of shape (n_samples, n_features)
                    Input hermitian matrix on which you want to apply QPCA divided by its trace, where `n_samples` is the number of samples
                    and `n_features` is the number of features.

        original_eigenvectors: array-like
                    Array representing the original eigenvectors of the input matrix.

        original_eigenvalues: array-like
                    Array representing the original eigenvalues of the input matrix.

        reconstructed_eigenvalues: array-like
                    Array of reconstructed eigenvalues.

        reconstructed_eigenvectors: array-like
                    Array of reconstructed eigenvectors.

        mean_threshold: array-like
                    This array contains the mean between the left and right peaks vertical distance to its neighbouring samples. It is useful for the benchmark process to cut out the bad eigenvalues.

        n_shots: int value
                    Number of measures performed in the tomography process.

        print_distances: bool value, default=True
                If True, the distance (defined by distance_type value) between the original and reconstructed eigenvector is printed in the legend.

        only_first_eigenvectors: bool value, default=True.
                If True, the function returns only the plot relative to the first eigenvalues. Otherwise, all the plot are showed.

        plot_delta: bool value, default=False
                If True, the function also returns the plot that shows how the tomography error decreases as the number of shots increases. 

        distance_type: string value, default='l2'
                It defines the distance measure used to benchmark the eigenvectors:

                        -'l2': the l2 distance between original and reconstructed eigenvectors is computed.

                        -'cosine': the cosine distance between original and reconstructed eigenvectors is computed.

        error_with_sign: bool value, default=False
                If True, the benchmarking is performed considering the reconstructed sign of the eigenvectors. Otherwise, the benchmarking is performed between absolute values of eigenvectors.

        hide_plot: bool value, default=False
                If True, the plot showing the eigenvector's benchmarking is not showed.


        Returns
        -------
        save_list: array-like
                List of tuples where the first element is the eigenvalue and the second is the distance between the corresponding reconstructed eigenvector and the original one.

        delta: float value
                The tomography error value.

        Notes
        -----
        The execution of this method shows the distance between original and reconstructed eigenvector's values and allows to visualize the tomography error. In this way, you can check that the reconstruction of the eigenvectors always takes place with an error conforming to the one expressed in the tomography algorithm in the "A Quantum Interior Point Method for LPs and SDPs" paper.
        """

        save_list=[]

        correct_reconstructed_eigenvalues=remove_usless_peaks(reconstructed_eigenvalues,mean_threshold,original_eigenvalues)

        correct_reconstructed_eigenvectors=[reconstructed_eigenvectors[:,j] for c_r_e in correct_reconstructed_eigenvalues for j in range(len(reconstructed_eigenvalues)) if c_r_e==reconstructed_eigenvalues[j]]

        original_eigenValues,original_eigenVectors=reorder_original_eigenvalues_eigenvectors(input_matrix,original_eigenvectors,original_eigenvalues,correct_reconstructed_eigenvalues)

        fig, ax = plt.subplots(1,len(correct_reconstructed_eigenvalues),figsize=(30, 10))
        if len(correct_reconstructed_eigenvalues)>1:

            for e,chart in enumerate(ax.reshape(-1,order='F')):
                delta=np.sqrt((36*len(original_eigenVectors[:,e%len(input_matrix)])*np.log(len(original_eigenVectors[:,e%len(input_matrix)])))/(n_shots))

                if error_with_sign==True:

                    sign_original=np.sign(original_eigenVectors[:,e%len(input_matrix)])
                    sign_original[sign_original==0]=1
                    sign_reconstructed=np.sign(correct_reconstructed_eigenvectors[e%len(input_matrix)])
                    sign_reconstructed[sign_reconstructed==0]=1
                    inverse_sign_original=sign_original*-1       
                    sign_difference=(sign_original==sign_reconstructed).sum()
                    inverse_sign_difference=(inverse_sign_original==sign_reconstructed).sum()

                    if sign_difference>=inverse_sign_difference:
                        original_eigenvector=original_eigenVectors[:,e%len(input_matrix)]
                    else:
                        original_eigenvector=original_eigenVectors[:,e%len(input_matrix)]*-1
                    reconstructed_eigenvector=correct_reconstructed_eigenvectors[e%len(input_matrix)]
                else:
                    original_eigenvector=abs(original_eigenVectors[:,e%len(input_matrix)])
                    reconstructed_eigenvector=abs(correct_reconstructed_eigenvectors[e%len(input_matrix)])
                if plot_delta:

                    for i in range(len(original_eigenVectors[:,(e%len(input_matrix))])):
                        circle=plt.Circle((i+1,original_eigenvector[i]),np.sqrt(7)*delta,color='g',alpha=0.1)
                        chart.add_patch(circle)
                        chart.axis("equal")
                        chart.hlines(original_eigenvector[i],xmin=i+1,xmax=i+1+(np.sqrt(7)*delta))
                        chart.text(i+1+((i+1+(np.sqrt(7)*delta))-(i+1))/2,original_eigenvector[i]+0.01,r'$\sqrt{7}\delta$')
                    chart.plot([], [], ' ', label=r'$\delta$='+str(round(delta,4)))   
                    chart.plot(list(range(1,len(input_matrix)+1)),reconstructed_eigenvector,marker='*',label='reconstructed',linestyle='None',markersize=12,alpha=0.5,color='r')
                    chart.plot(list(range(1,len(input_matrix)+1)),original_eigenvector,marker='o',label='original',linestyle='None',markersize=12,alpha=0.4)
                else:
                    chart.plot(list(range(1,len(input_matrix)+1)),reconstructed_eigenvector,marker='*',label='reconstructed',linestyle='None',markersize=12,alpha=0.5,color='r')
                    chart.plot(list(range(1,len(input_matrix)+1)),original_eigenvector,marker='o',label='original',linestyle='None',markersize=12,alpha=0.4)


                if print_distances:

                    distance=distance_function_wrapper(distance_type,reconstructed_eigenvector,original_eigenvector)
                    chart.plot([], [], ' ', label=distance_type+"_error "+str(np.round(distance,4)))
                else:
                    distance=np.nan

                save_list.append((correct_reconstructed_eigenvalues[e%len(input_matrix)],np.round(distance,4)))
                chart.plot([], [], ' ', label="n_shots "+str(n_shots))
                chart.legend()
                chart.set_ylabel("eigenvector's values")
                chart.set_title('Eigenvectors corresponding to eigenvalues '+str(correct_reconstructed_eigenvalues[e%len(input_matrix)]))
                if only_first_eigenvectors:
                    break
        else:

            delta=np.sqrt((36*len(original_eigenVectors[:,0])*np.log(len(original_eigenVectors[:,0])))/(n_shots))

            if error_with_sign==True:

                sign_original=np.sign(original_eigenVectors[:,0])
                sign_original[sign_original==0]=1
                sign_reconstructed=np.sign(correct_reconstructed_eigenvectors[0])
                sign_reconstructed[sign_reconstructed==0]=1
                inverse_sign_original=sign_original*-1       
                sign_difference=(sign_original==sign_reconstructed).sum()
                inverse_sign_difference=(inverse_sign_original==sign_reconstructed).sum()

                if sign_difference>=inverse_sign_difference:
                    original_eigenvector=original_eigenVectors[:,0]
                else:
                    original_eigenvector=original_eigenVectors[:,0]*-1
                reconstructed_eigenvector=correct_reconstructed_eigenvectors[0]
            else:
                original_eigenvector=abs(original_eigenVectors[:,0])
                reconstructed_eigenvector=abs(correct_reconstructed_eigenvectors[0])
            if plot_delta:

                for i in range(len(original_eigenVectors[:,0])):
                    circle=plt.Circle((i+1,original_eigenvector[i]),np.sqrt(7)*delta,color='g',alpha=0.1)
                    ax.add_patch(circle)
                    ax.axis("equal")
                    ax.hlines(original_eigenvector[i],xmin=i+1,xmax=i+1+(np.sqrt(7)*delta))
                    ax.text(i+1+((i+1+(np.sqrt(7)*delta))-(i+1))/2,original_eigenvector[i]+0.01,r'$\sqrt{7}\delta$')
                ax.plot([], [], ' ', label=r'$\delta$='+str(round(delta,4)))   
                ax.plot(list(range(1,len(input_matrix)+1)),reconstructed_eigenvector,marker='*',label='reconstructed',linestyle='None',markersize=12,alpha=0.5,color='r')
                ax.plot(list(range(1,len(input_matrix)+1)),original_eigenvector,marker='o',label='original',linestyle='None',markersize=12,alpha=0.4)
            else:
                ax.plot(list(range(1,len(input_matrix)+1)),reconstructed_eigenvector,marker='*',label='reconstructed',linestyle='None',markersize=12,alpha=0.5,color='r')
                ax.plot(list(range(1,len(input_matrix)+1)),original_eigenvector,marker='o',label='original',linestyle='None',markersize=12,alpha=0.4)


            if print_distances:

                distance=distance_function_wrapper(distance_type,reconstructed_eigenvector,original_eigenvector)
                ax.plot([], [], ' ', label=distance_type+"_error "+str(np.round(distance,4)))
            else:
                distance=np.nan

            save_list.append((correct_reconstructed_eigenvalues[0],np.round(distance,4)))
            ax.plot([], [], ' ', label="n_shots "+str(n_shots))
            ax.legend()
            ax.set_ylabel("eigenvector's values")
            ax.set_title('Eigenvectors corresponding to eigenvalues '+str(correct_reconstructed_eigenvalues[0]))
        if hide_plot:
            plt.close()
        else:
            fig.tight_layout()
            plt.show()

        return save_list,delta

    def __eigenvalues_benchmarking(self,original_eigenvalues, reconstructed_eigenvalues, mean_threshold, print_error):
        """ Method to benchmark the quality of the reconstructed eigenvalues. 

        Parameters
        ----------

        original_eigenvalues: array-like
                    Array representing the original eigenvalues of the input matrix.

        reconstructed_eigenvalues: array-like
                    Array of reconstructed eigenvalues.

        mean_threshold: array-like
                    This array contains the mean between the left and right peaks vertical distance to its neighbouring samples. It is useful for the benchmark process to cut out the bad eigenvalues.

        print_error: bool value
                    If True, a table showing the absolute reconstruction error for each eigenvalue is reported

        Returns
        -------

        Notes
        -----
        """    
        #reconstructed_eigenvalues=[eig[0] for eig in reconstructed_eigenvalue_eigenvector_tuple]

        fig, ax = plt.subplots(figsize=(10, 10))

        correct_reconstructed_eigenvalues=remove_usless_peaks(reconstructed_eigenvalues,mean_threshold,original_eigenvalues)
        dict_original_eigenvalues = {original_eigenvalues[e]:e+1 for e in range(len(original_eigenvalues))}
        idx_list=[]
        if print_error:
            t = Texttable()
        for eig in correct_reconstructed_eigenvalues:
            x,min_=find_nearest(original_eigenvalues,eig)
            idx_list.append(dict_original_eigenvalues[x])
            if print_error:
                error=abs(eig-x)
                lista=[['True eigenvalue','Reconstructed eigenvalue' ,'error'], [x,eig, error]]
                t.add_rows(lista)
        ax.plot(idx_list,correct_reconstructed_eigenvalues,marker='o',label='reconstructed',linestyle='None',markersize=25,alpha=0.3,color='r')
        ax.plot(list(range(1,len(original_eigenvalues)+1)),original_eigenvalues,marker='x',label='original',linestyle='None',markersize=20,color='black')
        ax.legend(labelspacing = 3)
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Matching between original and reconstructed eigenvalues')
        plt.show()
        if print_error:
            print(t.draw())
        return 
    

            
    def __sign_reconstruction_benchmarking(self,input_matrix, original_eigenvalues, original_eigenvectors, reconstructed_eigenvalues, reconstructed_eigenvectors, mean_threshold, n_shots):

        """ Method to benchmark the quality of the sign reconstruction for the reconstructed eigenvectors. 

        Parameters
        ----------

        input_matrix: array-like of shape (n_samples, n_features)
                    Input hermitian matrix on which you want to apply QPCA divided by its trace, where `n_samples` is the number of samples
                    and `n_features` is the number of features.

        original_eigenvalues: array-like
                    Array representing the original eigenvalues of the input matrix.

        original_eigenvectors: array-like
                    Array representing the original eigenvectors of the input matrix.

        reconstructed_eigenvalues: array-like
                    Array of reconstructed eigenvalues.

         reconstructed_eigenvectors: array-like
                    Array of reconstructed eigenvectors.

        mean_threshold: array-like
                    This array contains the mean between the left and right peaks vertical distance to its neighbouring samples. It is useful for the benchmark process to cut out the bad eigenvalues.

        n_shots: int value
                    The number of measures performed for the reconstruction of the eigenvectors.


        Returns
        -------

        Notes
        -----
        """ 

        correct_reconstructed_eigenvalues=remove_usless_peaks(reconstructed_eigenvalues,mean_threshold,original_eigenvalues)

        correct_reconstructed_eigenvectors=[reconstructed_eigenvectors[:,j] for c_r_e in correct_reconstructed_eigenvalues for j in range(len(reconstructed_eigenvalues)) if c_r_e==reconstructed_eigenvalues[j]]

        original_eigenValues,original_eigenVectors=reorder_original_eigenvalues_eigenvectors(input_matrix,original_eigenvectors,original_eigenvalues,correct_reconstructed_eigenvalues)

        t = Texttable()
        for e in range(len(correct_reconstructed_eigenvalues)):
            eigenvalue=original_eigenValues[e]
            o_eigenvector=original_eigenVectors[:,e]
            r_eigenvector=correct_reconstructed_eigenvectors[e]
            sign_original=np.sign(o_eigenvector)
            sign_original[sign_original==0]=1
            sign_reconstructed=np.sign(r_eigenvector)
            sign_reconstructed[sign_reconstructed==0]=1
            inverse_sign_original=np.sign(o_eigenvector)*-1
            inverse_sign_original[inverse_sign_original==0]=1        
            sign_difference=(sign_original==sign_reconstructed).sum()
            inverse_sign_difference=(inverse_sign_original==sign_reconstructed).sum()
            correct_sign=max(sign_difference,inverse_sign_difference)
            wrong_sign=len(o_eigenvector)-correct_sign
            lista=[['Eigenvalue','n_shots' ,'correct_sign','wrong_sign'], [eigenvalue,n_shots,correct_sign,wrong_sign ]]
            t.add_rows(lista)
        print(t.draw())
        return 
       
    
