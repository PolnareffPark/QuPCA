import numpy as np
import itertools
import pandas as pd


def postprocessing(input_matrix,statevector_dictionary,resolution):
        
        """ Eigenvectors reconstruction process from the reconstructed statevector.
        
        Parameters
        ----------
        
        input_matrix: array-like.
                        Hermitian input matrix.
                        
        statevector_dictionary: dict-like.
                        Dictionary where the keys represent the eigenvalues/eigenvectors encoded in the qubits and the values represent the reconstructed statevector's values.
                        
        resolution: int value.
                        Number of qubits used in the phase estimation process to represent the eigenvalues.
        
        
        Returns
        -------
        eigenvectors: array-like. 
                        List of tuples containing as first value the reconstructed eigenvalue and as second value the reconstructed eigenvector.
        
        Notes
        -----
        """
        
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
        for l in l_list:
            normalization_factor=np.sqrt((1/(sum(l**2))))
            l*=normalization_factor

        eigenvalue_eigenvector_tuples=[]
        for ll, eig in zip(l_list,eigenvalues):
            eigenvector=np.zeros(len(input_matrix))
            save_sign=np.sign(ll)
            statevector=abs(ll)
            max_list=[]
            scaled_statevectors=[]
            for e,i in enumerate(range(0,len(statevector),len(input_matrix))):
                max_list.append(max(statevector[i:i+len(input_matrix)]))
                
                scaled_statevectors.append(statevector[i:i+len(input_matrix)]/max_list[e])
            
            idx_max=np.argmax(max_list)
            max_max=max_list[idx_max]
           
            value=np.sqrt(max_max)

            eigenvector=scaled_statevectors[idx_max]*value*save_sign[len(input_matrix)*idx_max:len(input_matrix)*idx_max+len(input_matrix)]
            #eigenvector=scaled_statevectors[idx_max]*value*save_sign[:len(input_matrix)]
            eigenvalue_eigenvector_tuples.append((eig,eigenvector))
            
        eigenvalue_eigenvector_tuples=sorted(eigenvalue_eigenvector_tuples,reverse=True)
        #self.reconstructed_eigenvectors=sorted(eigenvectors,reverse=True)
        return eigenvalue_eigenvector_tuples
    
    
def general_postprocessing(input_matrix,statevector_dictionary,resolution):
        
        """ Eigenvectors reconstruction process from the reconstructed statevector.
        
        Parameters
        ----------
        
        input_matrix: array-like.
                        Hermitian input matrix.
                        
        statevector_dictionary: dict-like.
                        Dictionary where the keys represent the eigenvalues/eigenvectors encoded in the qubits and the values represent the reconstructed statevector's values.
                        
        resolution: int value.
                        Number of qubits used in the phase estimation process to represent the eigenvalues.
        
        
        Returns
        -------
        eigenvalue_eigenvector_tuples: array-like. 
                        List of tuples containing as first value the reconstructed eigenvalue and as second value the reconstructed eigenvector (sorted from the highest to the lowest eigenvalues).
        
        Notes
        -----
        """
        
        probabilities=np.array([abs(statevector_dictionary[s_d])**2 for s_d in statevector_dictionary])
        bitstrings=[''.join([''.join(str(j)) for j in i]) for i in list(map(list, itertools.product([0, 1], repeat=resolution+len(input_matrix))))]
        
        ss=list(zip(bitstrings, probabilities))

        df=pd.DataFrame(ss)
        
        df.columns=['state','module']

        df['lambda']=df['state'].apply(lambda x: x[-resolution:])
        df1=df.groupby('lambda').agg({'module':'sum'})
        df1=df1.sort_values('module',ascending=False)
        
        
        tail=df1.reset_index()
        tail['num']=tail['lambda'].apply(lambda x :int(x[::-1],base=2)/(2**resolution))
        
        lambdas=find_peaks(tail,input_matrix,resolution)

        #lambdas=df1.tail(len(input_matrix)).index.values
        
      
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
            eigenvector=np.zeros(len(input_matrix))
            save_sign=np.sign(ll)
            statevector=abs(ll)
            max_list=[]
            scaled_statevectors=[]
            for e,i in enumerate(range(0,len(statevector),len(input_matrix))):
                max_list.append(max(statevector[i:i+len(input_matrix)]))
                
                scaled_statevectors.append(statevector[i:i+len(input_matrix)]/max_list[e])
            
            idx_max=np.argmax(max_list)
            max_max=max_list[idx_max]
           
            value=np.sqrt(max_max)

            eigenvector=scaled_statevectors[idx_max]*value*save_sign[len(input_matrix)*idx_max:len(input_matrix)*idx_max+len(input_matrix)]
            #eigenvector=scaled_statevectors[idx_max]*value*save_sign[:len(input_matrix)]
            eigenvalue_eigenvector_tuples.append((eig,eigenvector))
            
        eigenvalue_eigenvector_tuples=sorted(eigenvalue_eigenvector_tuples,reverse=True)
        #self.reconstructed_eigenvectors=sorted(eigenvectors,reverse=True)
        return eigenvalue_eigenvector_tuples
    

def find_peaks(df,input_matrix,resolution):
    """ 
    Process to find the correct eigenvalues peaks after the PE procedure.
        
        Parameters
        ----------
        
        df: DataFrame.
                        It contains the extracted eigenvalues ordered by module squared.
        
        input_matrix: array-like.
                        Hermitian input matrix.
                        
        resolution: int value.
                        Number of qubits used in the phase estimation process to represent the eigenvalues.
        
        
        Returns
        -------
        peaks: list-like. 
                        List of peaks that represent eigenvalues extracted by PE.
        
        Notes
        -----
    """
    
    peaks=[]
    nums_peaks=[]
    peaks.append(df.iloc[0]['lambda'])
    nums_peaks.append(df.iloc[0]['num'])
    for i in range(1,len(df)):

        #for n_ in nums_peaks:

        if any(abs(df.iloc[i]['num']-n_)<= 4/(2**resolution) for n_ in nums_peaks):
            #if any(abs(tail1.iloc[i]['num']-n_)/n_<= 0.2 for n_ in nums_peaks):
            pass
        else:

            nums_peaks.append(df.iloc[i]['num'])
            peaks.append(df.iloc[i]['lambda'])
            pass
        if len(peaks)==len(input_matrix):
            break
        #print(tail.iloc[i])
        
    return peaks