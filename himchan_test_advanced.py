from QPCA.decomposition.Qpca import QPCA
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle 
from QPCA.preprocessingUtilities.preprocessing import generate_matrix
from QPCA.benchmark.benchmark import *
from qiskit_ibm_runtime import QiskitRuntimeService
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

token = '18084cb8c91eee736662c2f23065983acf97af6734e14355c78b8e424c4d0f480efc8b254aff7586b13810c48eb71baf384d26bc382aad7017cec64e857fbae3'

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='yonsei-dedicated/internal/ymsphdcandid',
    token=token
)

matrix_sizes = [4, 8, 16, 32, 64]
resolution = 8  # QPCA에 사용할 고정된 해상도
n_shots = 50000
n_repetitions = 1
seed = 7000

qPCA_times = []
PCA_times = []

for dimension in tqdm(matrix_sizes, desc="Measuring computation times for various matrix sizes"):
    # 해당 차원에 맞는 eigenvalues 리스트 생성 및 정규화
    eigenvalues_list_for_matrix = np.random.random(dimension)
    eigenvalues_list_for_matrix = eigenvalues_list_for_matrix / np.sum(eigenvalues_list_for_matrix)
    
    # 해당 차원의 행렬 생성
    input_matrix = generate_matrix(matrix_dimension=dimension,
                                   replicate_paper=False,
                                   seed=seed,
                                   eigenvalues_list=None)
    
    # QPCA 시간 측정 (fit부터 eigenvectors_reconstruction까지)
    start_time = time.time()
    qpca = QPCA().fit(input_matrix, resolution=resolution)
    _ , _ = qpca.eigenvectors_reconstruction(n_shots=n_shots,
                                            n_repetitions=n_repetitions,
                                            plot_peaks=False)
    qPCA_times.append(time.time() - start_time)
    
    # sklearn PCA 시간 측정
    start_time = time.time()
    pca = PCA()
    pca.fit(input_matrix)
    PCA_times.append(time.time() - start_time)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, qPCA_times, marker='o', label='QPCA')
plt.plot(matrix_sizes, PCA_times, marker='s', label='Sklearn PCA')
plt.title('Computation Time Comparison by Matrix Size')
plt.xlabel('Matrix Dimension')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()
