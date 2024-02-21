import numpy as np
from numpy.linalg import norm, eigvals, eig

from qiskit import QuantumCircuit, qasm
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, state_fidelity

import time
import json
from sympy import Matrix
import matplotlib.pyplot as plt

from helper_functions import construct_BE_rand, construct_BE, convert_binary_to_int
from QSP_hamiltonian_simulation import HamSim_byQET


I = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

## helper function
def operator_function(M, func):
    evals, evecs = eig(M)

    output_M = func(evals[0]) * np.array([evecs[:, 0]]).conj().T @ np.array([evecs[:, 0]])

    for i in range(1, len(evals)):
        output_M += func(evals[i]) * np.array([evecs[:, i]]).conj().T @ np.array([evecs[:, i]])
    
    return output_M



""" TEST 1: block encoding of I """
def test_block_encode_I():
    obj = HamSim_byQET(num_qubits=1, 
                       H=I,
                       evolution_time=5.0)

    assert np.allclose(obj.BE_H[0:2, 0:2], I), "test_block_encode_I failed"

""" TEST 2: block encoding of X """
def test_block_encode_X():
    obj = HamSim_byQET(num_qubits=1, 
                       H=X,
                       evolution_time=5.0)

    assert np.allclose(obj.BE_H[0:2, 0:2], X), "test_block_encode_X failed"

""" TEST 3: block encoding of Y"""
def test_block_encode_Y():
    obj = HamSim_byQET(num_qubits=1, 
                       H=Y,
                       evolution_time=5.0)

    assert np.allclose(obj.BE_H[0:2, 0:2], Y), "test_block_encode_Y failed"

""" TEST 4: block encoding of Z """
def test_block_encode_Z():
    obj = HamSim_byQET(num_qubits=1, 
                       H=Z,
                       evolution_time=5.0)

    assert np.allclose(obj.BE_H[0:2, 0:2], Z), "test_block_encode_Z failed"


""" TEST 5: QSP phase angles for approximating cos(xt), when t = 5.0"""
def test_QSP_angles_cos(evolution_time=5.0, error_tolerance=1e-2):

    obj = HamSim_byQET(num_qubits=1, 
                       H=Z,
                       evolution_time=evolution_time)
    obj.computeQSPPhaseAngles()
    
    W = lambda a: np.array([[a, (0+1j) * np.sqrt(1 - a**2)], [(0+1j) * np.sqrt(1 - a**2), a]])
    
    y = []
    x = np.linspace(-1, 1, 100)
    for a in x:
        W_Op = Operator(W(a))
        circ = QuantumCircuit(1)

        for phi in range(len(obj.cos_ang_seq)-1, 0, -1):
            circ.rz(2 * obj.cos_ang_seq[phi], 0)
            circ.append(W_Op, [0])
        
        circ.rz(2 * obj.cos_ang_seq[0], 0)
        
        y.append(((np.array([[1.], [1.]])/np.sqrt(2)).T @ Operator(circ).data @ np.array([[1.], [1.]])/np.sqrt(2))[0, 0])

    true_y = np.cos(x * evolution_time)

    assert np.allclose(y, true_y, rtol=error_tolerance), "test_QSP_angles_cos failed"

    # plt.plot(x, true_y, "r--")
    # plt.plot(x, y, "go")
    # plt.ylim([-1.75, 1.75])
    # plt.title("cos")
    # plt.show()

    
""" TEST 6: QSP phase angles for approximating sin(xt), when t = 5.0"""
def test_QSP_angles_sin(evolution_time=5.0, error_tolerance=1e-2):

    obj = HamSim_byQET(num_qubits=1, 
                       H=Z,
                       evolution_time=evolution_time)
    obj.computeQSPPhaseAngles()
    
    W = lambda a: np.array([[a, (0+1j) * np.sqrt(1 - a**2)], [(0+1j) * np.sqrt(1 - a**2), a]])
    
    y = []
    x = np.linspace(-1, 1, 100)
    for a in x:
        W_Op = Operator(W(a))
        circ = QuantumCircuit(1)

        for phi in range(len(obj.sin_ang_seq)-1, 0, -1):
            circ.rz(2 * obj.sin_ang_seq[phi], 0)
            circ.append(W_Op, [0])
        
        circ.rz(2 * obj.sin_ang_seq[0], 0)
        
        y.append(((np.array([[1.], [1.]])/np.sqrt(2)).T @ Operator(circ).data @ np.array([[1.], [1.]])/np.sqrt(2))[0, 0])

    true_y = np.sin(x * evolution_time)

    assert np.allclose(y, true_y, rtol=error_tolerance), "test_QSP_angles_sin failed"

    # plt.plot(x, true_y, "r--")
    # plt.plot(x, y, "go")
    # plt.ylim([-1.75, 1.75])
    # plt.title("sin")
    # plt.show()



""" TEST 7: block encoding of cos(Ht) by QSP circuit, when H = I """
def test_cos_QSP_I(evolution_time = 5.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = I
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    cos_QET = obj.obj_QET_cos.getCirc()
    cos_QET = cos_QET.reverse_bits()
    
    actual_output = Operator(cos_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.cos(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_cos_QSP_I failed"


""" TEST 8: block encoding of cos(Ht) by QSP circuit, when H = X """
def test_cos_QSP_X(evolution_time = 1.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = X
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    cos_QET = obj.obj_QET_cos.getCirc()
    cos_QET = cos_QET.reverse_bits()
    
    actual_output = Operator(cos_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.cos(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_cos_QSP_X failed"


""" TEST 9: block encoding of cos(Ht) by QSP circuit, when H = Y """
def test_cos_QSP_Y(evolution_time = 3.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Y
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    cos_QET = obj.obj_QET_cos.getCirc()
    cos_QET = cos_QET.reverse_bits()
    
    actual_output = Operator(cos_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.cos(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_cos_QSP_Y failed"


""" TEST 10: block encoding of cos(Ht) by QSP circuit, when H = Z """
def test_cos_QSP_Z(evolution_time = 0.5, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Z
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    cos_QET = obj.obj_QET_cos.getCirc()
    cos_QET = cos_QET.reverse_bits()
    
    actual_output = Operator(cos_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.cos(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_cos_QSP_Z failed"



""" TEST 11: block encoding of sin(Ht) by QSP circuit, when H = I """
def test_sin_QSP_I(evolution_time = 5.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = I
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    sin_QET = obj.obj_QET_sin.getCirc()
    sin_QET = sin_QET.reverse_bits()
    
    actual_output = Operator(sin_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.sin(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_sin_QSP_I failed"


""" TEST 12: block encoding of sin(Ht) by QSP circuit, when H = X """
def test_sin_QSP_X(evolution_time = 1.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = X
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    sin_QET = obj.obj_QET_sin.getCirc()
    sin_QET = sin_QET.reverse_bits()
    
    actual_output = Operator(sin_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.sin(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_sin_QSP_X failed"



""" TEST 13: block encoding of sin(Ht) by QSP circuit, when H = Y """
def test_sin_QSP_Y(evolution_time = 3.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Y
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    sin_QET = obj.obj_QET_sin.getCirc()
    sin_QET = sin_QET.reverse_bits()
    
    actual_output = Operator(sin_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.sin(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_sin_QSP_Y failed"


""" TEST 14: block encoding of sin(Ht) by QSP circuit, when H = Z """
def test_sin_QSP_Z(evolution_time = 0.5, error_tolerance=1e-3) -> bool:
    
    ## Testing parameters
    num_qubits = 1
    H = Z
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    sin_QET = obj.obj_QET_sin.getCirc()
    sin_QET = sin_QET.reverse_bits()
    
    actual_output = Operator(sin_QET).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    expected_output = operator_function(H, lambda x: np.sin(x * evolution_time))
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_sin_QSP_Z failed"


""" TEST 15: block encoding of exp(-iHt) by LCU circuit, when H = I """
def test_HamSim_LCU_I(evolution_time = 5.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = I
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    HS = obj.HamSim_circ
    HS = HS.reverse_bits()
    
    actual_output = Operator(HS).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    ## Note: the LCU circuit block encodes the function exp(-iHt) / 2 
    expected_output = operator_function(H, lambda x: np.exp(-1j * x * evolution_time)) / 2
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_HamSim_LCU_I failed"


""" TEST 16: block encoding of exp(-iHt) by LCU circuit, when H = X """
def test_HamSim_LCU_X(evolution_time = 1.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = X
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    HS = obj.HamSim_circ
    HS = HS.reverse_bits()
    
    actual_output = Operator(HS).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    ## Note: the LCU circuit block encodes the function exp(-iHt) / 2 
    expected_output = operator_function(H, lambda x: np.exp(-1j * x * evolution_time)) / 2
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_HamSim_LCU_X failed"


""" TEST 17: block encoding of exp(-iHt) by LCU circuit, when H = Y """
def test_HamSim_LCU_Y(evolution_time = 3.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Y
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    HS = obj.HamSim_circ
    HS = HS.reverse_bits()
    
    actual_output = Operator(HS).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    ## Note: the LCU circuit block encodes the function exp(-iHt) / 2 
    expected_output = operator_function(H, lambda x: np.exp(-1j * x * evolution_time)) / 2
    
    print("actual_output", actual_output)
    print("expected_output", expected_output)
    print(np.isclose(actual_output, expected_output, rtol=error_tolerance))

    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_HamSim_LCU_Y failed"


""" TEST 18: block encoding of exp(-iHt) by LCU circuit, when H = Z """
def test_HamSim_LCU_Z(evolution_time = 0.5, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Z
    
    ## Construct the block encoding operator
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       H = H,
                       evolution_time = evolution_time,
                       truncation_order = 8)

    obj.computeQSPPhaseAngles()
    obj.buildCirc()

    HS = obj.HamSim_circ
    HS = HS.reverse_bits()
    
    actual_output = Operator(HS).data[0:2, 0:2]
    
    
    ## Expected output computed via an operator function
    ## Note: the LCU circuit block encodes the function exp(-iHt) / 2 
    expected_output = operator_function(H, lambda x: np.exp(-1j * x * evolution_time)) / 2
    
    if np.allclose(actual_output, expected_output, rtol=error_tolerance) or np.allclose(actual_output, - expected_output, rtol=error_tolerance):
        assert True
    else:
        assert False, "test_HamSim_LCU_Z failed"


""" Test 19: Fidelity between the statevectors evolved by QSP method and numerical (benchmard) method, when H = I """
def test_fidelity_I(decimals=4, evolution_time=5.0):
    
    num_qubits = 1
    H = I

    ## QET method
    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                                H = H,
                                evolution_time = evolution_time)

    obj_HS_QET.computeQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    output_state, _ = obj_HS_QET.runHamiltonianSimulator()

    ## Numerical/Qiskit method
    numerical_circ = QuantumCircuit(num_qubits)
    numerical_circ.hamiltonian(operator = Operator(obj_HS_QET.getHamiltonian()),
                                time = evolution_time,
                                qubits = [i for i in range(num_qubits-1, -1, -1)])

    bench_state = Statevector.from_int(0, 2 ** num_qubits)
    bench_state = bench_state.evolve(numerical_circ)
    
    fidelity = state_fidelity(DensityMatrix(bench_state), DensityMatrix(output_state))

    assert np.round(fidelity, decimals=decimals) == 1.0, "test_fidelity_I failed"


""" Test 20: Fidelity between the statevectors evolved by QSP method and numerical (benchmard) method, when H = X """
def test_fidelity_X(decimals=4, evolution_time=5.0):
    
    num_qubits = 1
    H = X

    ## QET method
    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                                H = H,
                                evolution_time = evolution_time)

    obj_HS_QET.computeQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    output_state, _ = obj_HS_QET.runHamiltonianSimulator()

    ## Numerical/Qiskit method
    numerical_circ = QuantumCircuit(num_qubits)
    numerical_circ.hamiltonian(operator = Operator(obj_HS_QET.getHamiltonian()),
                                time = evolution_time,
                                qubits = [i for i in range(num_qubits-1, -1, -1)])

    bench_state = Statevector.from_int(0, 2 ** num_qubits)
    bench_state = bench_state.evolve(numerical_circ)
    
    fidelity = state_fidelity(DensityMatrix(bench_state), DensityMatrix(output_state))

    assert np.round(fidelity, decimals=decimals) == 1.0, "test_fidelity_X failed"


""" Test 21: Fidelity between the statevectors evolved by QSP method and numerical (benchmard) method, when H = Y """
def test_fidelity_Y(decimals=4, evolution_time=5.0):
    
    num_qubits = 1
    H = Y

    ## QET method
    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                                H = H,
                                evolution_time = evolution_time)

    obj_HS_QET.computeQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    output_state, _ = obj_HS_QET.runHamiltonianSimulator()

    ## Numerical/Qiskit method
    numerical_circ = QuantumCircuit(num_qubits)
    numerical_circ.hamiltonian(operator = Operator(obj_HS_QET.getHamiltonian()),
                                time = evolution_time,
                                qubits = [i for i in range(num_qubits-1, -1, -1)])

    bench_state = Statevector.from_int(0, 2 ** num_qubits)
    bench_state = bench_state.evolve(numerical_circ)
    
    fidelity = state_fidelity(DensityMatrix(bench_state), DensityMatrix(output_state))

    assert np.round(fidelity, decimals=decimals) == 1.0, "test_fidelity_Y failed"


""" Test 22: Fidelity between the statevectors evolved by QSP method and numerical (benchmard) method, when H = Z """
def test_fidelity_Z(decimals=4, evolution_time=5.0):
    
    num_qubits = 1
    H = Z

    ## QET method
    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                                H = H,
                                evolution_time = evolution_time)

    obj_HS_QET.computeQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    output_state, _ = obj_HS_QET.runHamiltonianSimulator()

    ## Numerical/Qiskit method
    numerical_circ = QuantumCircuit(num_qubits)
    numerical_circ.hamiltonian(operator = Operator(obj_HS_QET.getHamiltonian()),
                                time = evolution_time,
                                qubits = [i for i in range(num_qubits-1, -1, -1)])

    bench_state = Statevector.from_int(0, 2 ** num_qubits)
    bench_state = bench_state.evolve(numerical_circ)
    
    fidelity = state_fidelity(DensityMatrix(bench_state), DensityMatrix(output_state))

    assert np.round(fidelity, decimals=decimals) == 1.0, "test_fidelity_Z failed"


""" Test 23: Fidelity between the statevectors evolved by QSP method and numerical (benchmard) method, when H = YX + ZZ (normalised)"""
def test_fidelity_YX_ZZ(decimals=4, evolution_time=5.0):
    
    num_qubits = 2
    H = np.kron(Y, X) + np.kron(Z, Z)

    ## QET method
    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                                H = H,
                                evolution_time = evolution_time)

    obj_HS_QET.computeQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    output_state, _ = obj_HS_QET.runHamiltonianSimulator()

    ## Numerical/Qiskit method
    numerical_circ = QuantumCircuit(num_qubits)
    numerical_circ.hamiltonian(operator = Operator(obj_HS_QET.getHamiltonian()),
                                time = evolution_time,
                                qubits = [i for i in range(num_qubits-1, -1, -1)])

    bench_state = Statevector.from_int(0, 2 ** num_qubits)
    bench_state = bench_state.evolve(numerical_circ)
    
    fidelity = state_fidelity(DensityMatrix(bench_state), DensityMatrix(output_state))

    assert np.round(fidelity, decimals=decimals) == 1.0, "test_fidelity_YX_ZZ failed"

if __name__ == "__main__":
    test_block_encode_I()
    test_block_encode_X()
    test_block_encode_Y()
    test_block_encode_Z()

    test_QSP_angles_cos()
    test_QSP_angles_sin()

    test_cos_QSP_I()
    test_cos_QSP_X()
    test_cos_QSP_Y()
    test_cos_QSP_Z()
    test_sin_QSP_I()
    test_sin_QSP_X()
    test_sin_QSP_Y()
    test_sin_QSP_Z()
    test_HamSim_LCU_I()
    test_HamSim_LCU_X()
    """ test_HamSim_LCU_Y() """
    test_HamSim_LCU_Z()

    test_fidelity_I()
    test_fidelity_X()
    test_fidelity_Y()
    test_fidelity_Z()
    test_fidelity_YX_ZZ()

    print("All 22 testcases passes successfully!")
