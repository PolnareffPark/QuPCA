import numpy as np
import itertools
import pandas as pd
from scipy.signal import find_peaks
    
def general_postprocessing(input_matrix,statevector_dictionary,resolution,n_shots, plot_peaks):
        
        """ Eigenvectors reconstruction process from the reconstructed statevector.
        
        Parameters
        ----------
        
        input_matrix: array-like
                        Hermitian input matrix.
                        
        statevector_dictionary: dict-like.
                        Dictionary where the keys represent the eigenvalues/eigenvectors encoded in the qubits and the values represent the reconstructed statevector's values.
                        
        resolution: int value
                        Number of qubits used in the phase estimation process to represent the eigenvalues.
                        
        n_shots: int value
                        Number of measures performed in the tomography process.
        
        plot_peaks: bool value
                        If True, it returns a plot of the peaks which correspond to the eigenvalues finded by the phase estimation procedure.
        
        Returns
        -------
        eigenvalue_eigenvector_tuples: array-like 
                        List of tuples containing as first value the reconstructed eigenvalue and as second value the reconstructed eigenvector (sorted from the highest to the lowest eigenvalues).
        
        mean_threshold: array-like
                        This array contains the mean between the left and right peaks vertical distance to its neighbouring samples. It is useful for the benchmark process to cut out the bad eigenvalues.
        
        Notes
        -----
        """
        
        len_input_matrix=len(input_matrix)
        probabilities=np.array([abs(statevector_dictionary[s_d])**2 for s_d in statevector_dictionary])
        bitstrings=[''.join([''.join(str(j)) for j in i]) for i in list(map(list, itertools.product([0, 1], repeat=resolution+len_input_matrix)))]
        probabilities_bitstringed=list(zip(bitstrings, probabilities))
        df=pd.DataFrame(probabilities_bitstringed)
        df.columns=['state','module']
        df['lambda']=df['state'].apply(lambda x: x[-resolution:])
        df1=df.groupby('lambda').agg({'module':'sum'})
        df1=df1.sort_values('module',ascending=False)
        
        tail=df1.reset_index()
        tail['eigenvalue']=tail['lambda'].apply(lambda x :int(x[::-1],base=2)/(2**resolution))
        if plot_peaks:
            tail[['eigenvalue','module']].sort_values('eigenvalue').set_index('eigenvalue').plot(style='-*',figsize=(15,10))
        lambdas,lambdas_num,mean_threshold=__peaks_extraction(tail,len_input_matrix,n_shots)
        df.columns=['state','module','lambda']
        signs=np.sign(np.array(list(statevector_dictionary.values())))
        signs[np.where(signs==0)]=1
        df['sign']=signs
        df['module']=df['module'].multiply(signs, axis=0)
        df=df.fillna(0)
        
        l_list=[]
        save_sign=[]
        eigenvalues=[]
        for l in lambdas:
            eigenvalues.append(int(l[::-1],base=2)/(2**resolution))
            a_=np.array(df.query("state.str.endswith(@l)")['module'].values)
            save_sign.append(np.sign(a_))
            l_list.append(np.sqrt(abs(a_)))
        for i in range(len(l_list)):
            normalization_factor=np.sqrt((1/(sum(l_list[i]**2))))
            l_list[i]*=normalization_factor
            l_list[i]*=save_sign[i]    

        eigenvalue_eigenvector_tuples=[]
        for ll, eig in zip(l_list,eigenvalues):
            eigenvector=np.zeros(len_input_matrix)
            save_sign=np.sign(ll)
            statevector=abs(ll)
            max_list=[]
            scaled_statevectors=[]
            for e,i in enumerate(range(0,len(statevector),len_input_matrix)):
                max_list.append(max(statevector[i:i+len_input_matrix]))
                scaled_statevectors.append(statevector[i:i+len_input_matrix]/max_list[e])
            
            idx_max=np.argmax(max_list)
            max_max=max_list[idx_max]
            value=np.sqrt(max_max)
            eigenvector=scaled_statevectors[idx_max]*value*save_sign[len_input_matrix*idx_max:len_input_matrix*idx_max+len_input_matrix]
            eigenvalue_eigenvector_tuples.append((eig,eigenvector))
        
        return eigenvalue_eigenvector_tuples,mean_threshold
    

def __peaks_extraction(df,len_input_matrix,n_shots):
    """ 
    Process to find the correct eigenvalues peaks after the PE procedure.
        
        Parameters
        ----------
        
        df: DataFrame
                        It contains the extracted eigenvalues ordered by module squared.
        
        len_input_matrix: int value
                        Length of the input matrix.
        
        n_shots: int value
                        Number of measures performed in the tomography process.
        
        Returns
        -------
        sorted_peaks: list-like
                        List of peaks in binary format that represent eigenvalues extracted by PE.
                        
        sorted_num_peaks: list-like
                        List of peaks in decimal format that represent eigenvalues extracted by PE.
                        
        mean_threshold: ndarray-like
                        This array contains the mean between the left and right peaks vertical distance to its neighbouring samples.
        
        
        Notes
        -----
        This algorithm uses the find_peaks SciPy function to extract the right peaks (which correspond to the eigenvalues) of the PE outputs. This is an iterative function that search for the right number of peaks. If an extracted peak is not the right one, it means that the PE resolution needs to be increased to correctly identify the right peaks.
    """
    
    peaks=[]
    nums_peaks=[]
    offset=1/n_shots
    stop=False
    while stop==False:
        p_=find_peaks(df.sort_values(['eigenvalue'])['module'],threshold=offset)
        p=p_[0]
        right_thresholds=p_[1]['right_thresholds']
        left_thresholds=p_[1]['left_thresholds']
        
        
        if len(p)>len_input_matrix or len(p)==0:
            offset+=1/n_shots
        else:
            stop=True
            for i in p:
                el = df.sort_values(['eigenvalue']).iloc[i]
                nums_peaks.append(el['eigenvalue'])
                peaks.append(el['lambda'])
    
    mean_threshold=(left_thresholds+right_thresholds)/2
    sorted_peaks=np.array(peaks)[mean_threshold.argsort()[::-1]]
    sorted_num_peaks=np.array(nums_peaks)[mean_threshold.argsort()[::-1]]
    return sorted_peaks,sorted_num_peaks,sorted(mean_threshold,reverse=True)
    
