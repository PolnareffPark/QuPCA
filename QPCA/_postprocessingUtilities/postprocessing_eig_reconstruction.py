import numpy as np
import itertools


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