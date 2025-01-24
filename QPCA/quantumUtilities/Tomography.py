import numpy as np
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import PhaseEstimation
from qiskit import transpile
from qiskit_aer import Aer
import matplotlib.pyplot as plt
from qiskit.circuit.library.data_preparation.state_preparation import StatePreparation
from ..warnings_utils.warning_utility import *
from qiskit_ibm_runtime import Sampler, Options

class StateVectorTomography():

    def __computing_amplitudes(quantum_circuit,q_size,c_size,n_shots,drawing_amplitude_circuit,backend,qubits_to_be_measured):
        
        """ This is the first step of the state vector tomography algorithm described in Algorithm 4.1 in "A Quantum Interior Point Method for LPs and SDPs" paper. It is
            useful to reconstruct each statevector components.

            Parameters
            ----------

            q_size: int value
                        The size of the quantum register under consideration.

            c_size: int value
                        The size of the classical register under consideration.

            qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None
                        Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).

            Returns
            -------
            probabilities: array-like
                        The reconstructed probabilities statevector (without sign reconstruction).

            Notes
            -----
            This method is used to reconstruct the amplitudes of the values of the statevector that is under consideration. In addition to this function, state vector tomography also includes a sign estimation function.
            """
        
        #initialize a zero vector where we store the estimated probabilities
        
        # 1) 기존과 동일하게 회로를 구성
        probabilities = np.zeros(2**c_size)
        quantum_regs = QuantumRegister(q_size)
        classical_regs = ClassicalRegister(c_size, 'classical')
        amplitude_estimation_circuit = QuantumCircuit(quantum_regs, classical_regs)
        amplitude_estimation_circuit.append(quantum_circuit, quantum_regs)
        amplitude_estimation_circuit.measure(quantum_regs[qubits_to_be_measured], classical_regs)

        if drawing_amplitude_circuit:
            display(amplitude_estimation_circuit.draw('mpl'))

        # Transpile the circuit for the target backend
        transpiled_circuit = transpile(amplitude_estimation_circuit, backend=backend)

        # 2) backend.run() 대신 무조건 Sampler 사용
        #    - IBM Runtime에서는 반드시 Sampler(또는 Estimator)와 같은 프리미티브를 써야 함.
        sampler = Sampler(backend)  
        
        # 3) Sampler 실행
        job = sampler.run([transpiled_circuit], shots=n_shots)

        # 4) shots와 quasi_dists를 통해 counts 계산 (기존 코드와 동일)
        shots = n_shots
        counts = {
            k: round(v * shots)
            for k, v in job.result().quasi_dists[0].binary_probabilities().items()
        }

        # 5) 확률로 변환 후 저장
        for bitstring in counts:
            counts[bitstring] /= n_shots
            probabilities[int(bitstring, 2)] = counts[bitstring]

        return probabilities
    
    def __sign_estimation(quantum_circuit,probabilities,q_size,c_size,n_shots,drawing_sign_circuit,backend,qubits_to_be_measured):
        
        """ This is the second and last step of the state vector tomography algorithm described in Algorithm 4.1 in "A Quantum Interior Point Method for LPs and SDPs" paper. It is
            useful to reconstruct the sign of each statevector's components.

            Parameters
            ----------
            
            probabilities: array-like 
                        The reconstructed probabilities statevector (without sign reconstruction) obtained from computing_amplitudes function.
            
            q_size: int value 
                        The size of the quantum register under consideration.

            c_size: int value 
                        The size of the classical register under consideration.

            qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None
                        Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).

            Returns
            -------

            statevector_dictionary: dict-like
                        Dictionary where the keys represent the eigenvalues/eigenvectors encoded in the qubits and the values represent the reconstructed statevector's values (with sign).

            Notes
            -----
            This method is used to reconstruct the correct sign of the statevector's values under consideration.
            """
        
        
        qr_total_xi = QuantumRegister(q_size, 'xi')
        n_classical_register=c_size+1
        classical_registers=ClassicalRegister(n_classical_register,'classical')
        qr_control = QuantumRegister(1, 'control_qubit')
        
        #creation of the first controlled operator U from the circuit that contains the matrix encoding and the phase estimation operator
        
        op_U=quantum_circuit.to_gate(label='op_U').control()
        
        #initialize the second controlled operator with the estimated amplitudes computed in the first step of the tomography
        
        op_V = StatePreparation(np.sqrt(probabilities),label='op_V').control()
        
        #implement the sign estimation circuit as an Hadamard test

        sign_estimation_circuit = QuantumCircuit(qr_total_xi,qr_control, classical_registers,name='matrix')
        sign_estimation_circuit.h(qr_control)
        sign_estimation_circuit.x(qr_control)
        sign_estimation_circuit.append(op_U, qr_control[:]+qr_total_xi[:])
        sign_estimation_circuit.x(qr_control)
        sign_estimation_circuit.append(op_V, qr_control[:]+qr_total_xi[qubits_to_be_measured])
        sign_estimation_circuit.h(qr_control)
        sign_estimation_circuit.measure(qr_total_xi[qubits_to_be_measured],classical_registers[0:n_classical_register-1])
        sign_estimation_circuit.measure(qr_control,classical_registers[n_classical_register-1])
    
        if drawing_sign_circuit:
            display(sign_estimation_circuit.draw('mpl'))

        # Transpile the circuit for the target backend
        transpiled_circuit = transpile(sign_estimation_circuit, backend=backend)

        # Sampler만 사용하도록 수정
        sampler = Sampler(backend)  
        job = sampler.run([transpiled_circuit], shots=n_shots)

        shots = n_shots
        counts_for_sign = {
            k: round(v * shots)
            for k, v in job.result().quasi_dists[0].binary_probabilities().items()
        }

        # 컨트롤 큐빗(맨 앞 비트)이 0인 경우만 추려서 tmp에 저장
        tmp = np.zeros(2**c_size)
        for bitstring in counts_for_sign:
            if bitstring[0] == '0':  # 컨트롤 큐빗이 0
                tmp[int(bitstring[1:], 2)] = counts_for_sign[bitstring]

        # (기존 로직) tmp와 probabilities를 비교해 부호 판정
        sign_dictionary = {}
        for e, (count, prob) in enumerate(zip(tmp, probabilities)):
            if count > 0.4 * prob * n_shots:
                sign = 1
            else:
                sign = -1
            if prob == 0:
                sign = 1
            sign_dictionary[bin(e)[2:].zfill(c_size)] = sign

        # (기존 로직) 각 상태에 대해 sign * sqrt(prob)
        statevector_dictionary = {}
        for e, key in enumerate(sign_dictionary):
            statevector_dictionary[key] = sign_dictionary[key] * np.sqrt(probabilities[e])

        return statevector_dictionary
    
    @classmethod
    def state_vector_tomography(cls,quantum_circuit,n_shots,n_repetitions,qubits_to_be_measured=None,backend=None,drawing_amplitude_circuit=False,drawing_sign_circuit=False):
        """
        State vector tomography to estimate real vectors.

        Parameters
        ----------

        quantum_circuit: QuantumCircuit 
                    The quantum circuit to be reconstructed. 

        n_shots: int value
                    Number of measures performed in the tomography process.
        
        n_repetitions: int value
                Number of times that state vector tomography will be executed. If > 1, the final result will be the average result
                of all the execution of the tomography.
                    
        qubits_to_be_measured: Union[Qubit, QuantumRegister, int, slice, Sequence[Union[Qubit, int]]]), default=None.
                    Qubits to be measured. If None, all the qubits will be measured (like measure_all() instruction).
        
        backend: Qiskit backend, default value=None.
                    The Qiskit backend used to execute the circuit. If None, the qasm simulator is used by default.
        
        drawing_amplitude_circuit: bool value, default=False.
                    If True, a drawing of the amplitude estimation circuit of the tomography algorithm is displayed. Otherwise, only the reconstructed statevector is returned.
        
        drawing_sign_circuit: bool value, default=False.
                    If True, a drawing of the sign estimation circuit of the tomography algorithm is displayed. Otherwise, only the reconstructed statevector is returned.

        Returns
        -------
        tomography_dict: dict-like. 
                    The reconstructed statevector of the input quantum circuit.

        Notes
        -----
        This method reconstruct the real statevector of the input quantum circuit. It is an implementation of the Algorithm 4.1 in "A Quantum Interior Point Method for LPs and SDPs" paper, and it is composed of 
        two parts: amplitudes estimation and sign estimation.
        """
        
        if backend==None:
            backend = Aer.get_backend("qasm_simulator")
            
        #Set the number of quantum and classical register for tomography procedure
            
        q_size=quantum_circuit.qregs[0].size
        if qubits_to_be_measured==None:
            c_size=q_size
            qubits_to_be_measured=list(range(q_size))
        elif isinstance(qubits_to_be_measured,int):
            c_size=1
        else:
            tmp_array=np.array(list(range(q_size)))
            c_size=len(tmp_array[qubits_to_be_measured])

        tomography_list_dict=[]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            for j in range(n_repetitions):

                probabilities=cls.__computing_amplitudes(quantum_circuit,q_size,c_size,n_shots,drawing_amplitude_circuit,backend,qubits_to_be_measured)
                tomography_list_dict.append(cls.__sign_estimation(quantum_circuit,probabilities,q_size,c_size,n_shots,drawing_sign_circuit,backend,qubits_to_be_measured))

        states=list(tomography_list_dict[0].keys())
        tomography_dict={}
        for s in states:
            
            tmp=[d[s] for d in tomography_list_dict]
            mean=np.mean(tmp)
            tomography_dict.update({s:mean})

        return tomography_dict