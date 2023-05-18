import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.spatial import distance 

    
def reorder_original_eigenvalues_eigenvectors(input_matrix, original_eigenVectors, original_eigenValues, lambdas_num):
    
    check_eigenvalues={}
    new_original_eigenvalues=[]
    new_original_eigenvectors=[]
    
    for i in original_eigenValues:
        check_eigenvalues.update({i:0})
    
    for l_n in lambdas_num:
        x,min_=find_nearest(original_eigenValues,l_n)
        if check_eigenvalues[x]==0:
            check_eigenvalues.update({x:1})
            new_original_eigenvalues.append(x)
        else:
            continue
    
    for d in list(check_eigenvalues.keys()):
        if check_eigenvalues[d]==1:
            check_eigenvalues.pop(d)
           
    if len(check_eigenvalues)>0:
        new_original_eigenvalues+=list(check_eigenvalues.keys())
    
    
    for i in new_original_eigenvalues:
        for j in range(len(original_eigenValues)):
            if i==original_eigenValues[j]:
                new_original_eigenvectors.append(original_eigenVectors[:,j])
    new_original_eigenvectors=np.array(new_original_eigenvectors)
    new_original_eigenvectors=new_original_eigenvectors.reshape(len(input_matrix),len(input_matrix)).T
    return np.array(new_original_eigenvalues),new_original_eigenvectors

def remove_usless_peaks(lambdas_num, mean_threshold, original_eig):
    
    check_eigenvalues={}
    peaks_to_keep=[]
    peaks_not_to_keep=[]
    
    #hashmap to store original eigenvalue 

    for i in original_eig:

        check_eigenvalues.update({i:0}) 
    
    #check if we have two eigenvalues with the same mean_threshold values: it means that one of them will be a wrong estimated one

    if len(np.array(lambdas_num)[np.argwhere(mean_threshold == np.amin(mean_threshold))])>1:
        
        #the eigenvalues with the mean_threshold values different from each other will be taken in a different list 
        
        not_equal_threshold=np.array(lambdas_num)[np.argwhere(mean_threshold != np.amin(mean_threshold))]
        equal_thresholds=np.array(lambdas_num)[np.argwhere(mean_threshold == np.amin(mean_threshold))]
        dict_for_equal_threshold={}
        
        #check if the not_equal_thresholds eigenvalues are correctly estimated: if the nearest original eigenvalue , finded using the check_eigenvalues hashmap, has value 0 it means that it has not yet been considered
        #Therefore the corresponding estimated eigenvalue can be considered as correctly estimated. 
        
        for n_e_t in not_equal_threshold:
            x,min_=find_nearest(original_eig,n_e_t)
            if check_eigenvalues[x]==0:
                check_eigenvalues.update({x:1})
                peaks_to_keep.append(n_e_t)
            else:
                peaks_not_to_keep.append(n_e_t)
        
        #same check of before but for the equal_thresholds eigenvalues, that we know having some issues due to the same mean_threshold values.
        #In case of two very similar eigenvalues with the same mean_threshold, we keep the right one by looking at the minimum distance with respect to the original eigenvalue not already considered.

        for e_t in equal_thresholds:
            x,min_=find_nearest(original_eig,e_t)
            tuple_=(e_t,min_)
            dict_for_equal_threshold.setdefault(x, []).append(tuple_)
       
        for d_d in dict_for_equal_threshold:
            minimum_to_keep=min(dict_for_equal_threshold[d_d], key = lambda t: t[1])[0]
            not_minimum_to_discard=[e[0] for e in dict_for_equal_threshold[d_d] if e[0]!= minimum_to_keep]

            if check_eigenvalues[d_d]==0:   
                check_eigenvalues.update({d_d:1})
                peaks_to_keep.append(minimum_to_keep)
            else:
            
                peaks_not_to_keep.append(minimum_to_keep)
            peaks_not_to_keep+=not_minimum_to_discard
    else:
        for n_p in lambdas_num:
            x,min_=find_nearest(original_eig,n_p)
            if check_eigenvalues[x]==0:
                check_eigenvalues.update({x:1})
                peaks_to_keep.append(n_p)
            else:
                peaks_not_to_keep.append(n_p)
    
    #remove wrongly estimated eigenvalue
    
    if len(peaks_not_to_keep)>0:
        idxs=[list(lambdas_num).index(x) for x in peaks_not_to_keep if x in lambdas_num]
        lambdas_num=np.delete(lambdas_num,idxs)
    return sorted(lambdas_num,reverse=True)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    min_=(np.abs(array - value)).min()
    return array[idx],min_
    
def distance_function_wrapper(distance_type, *args):
    reconstructed_eig= args[0]
    original_eig = args[1]
    if distance_type=='l2':
        return np.linalg.norm(reconstructed_eig-original_eig)
    if distance_type=='cosine':
        return distance.cosine(reconstructed_eig,original_eig)