import numpy as np
from numpy.linalg import norm, eigvals, eig

from qiskit import QuantumCircuit, qasm
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, state_fidelity

import time
import json
from sympy import Matrix
import matplotlib.pyplot as plt

from helper_functions import construct_BE_rand, construct_BE, convert_binary_to_int
from HamiltonianSimulation_by_QET import HamSim_byQET


I = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])

## helper function
def operator_function(M, func):
    evals, evecs = eig(M)

    output_M = func(evals[0]) * np.array([evecs[0]]).conj().T @ np.array([evecs[0]])

    for i in range(1, len(evals)):
        output_M += func(evals[i]) * np.array([evecs[i]]).conj().T @ np.array([evecs[i]])
    
    return output_M



""" TEST 1: block encoding of cos(Ht) by QSP circuit, when H = I """
def test_cos_QSP_I(evolution_time = 5.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = I
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 1: block encoding of cos(Ht) by QSP circuit, when H = X """
def test_cos_QSP_X(evolution_time = 1.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = X
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 1: block encoding of cos(Ht) by QSP circuit, when H = Y """
def test_cos_QSP_Y(evolution_time = 3.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Y
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 1: block encoding of cos(Ht) by QSP circuit, when H = Z """
def test_cos_QSP_Z(evolution_time = 0.5, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Z
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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



""" TEST 5: block encoding of sin(Ht) by QSP circuit, when H = I """
def test_sin_QSP_I(evolution_time = 5.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = I
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 6: block encoding of sin(Ht) by QSP circuit, when H = X """
def test_sin_QSP_X(evolution_time = 1.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = X
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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



""" TEST 7: block encoding of sin(Ht) by QSP circuit, when H = Y """
def test_sin_QSP_Y(evolution_time = 3.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Y
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 8: block encoding of sin(Ht) by QSP circuit, when H = Z """
def test_sin_QSP_Z(evolution_time = 0.5, error_tolerance=1e-3) -> bool:
    
    ## Testing parameters
    num_qubits = 1
    H = Z
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 9: block encoding of exp(-iHt) by LCU circuit, when H = I """
def test_HamSim_LCU_I(evolution_time = 5.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = I
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 10: block encoding of exp(-iHt) by LCU circuit, when H = X """
def test_HamSim_LCU_X(evolution_time = 1.0, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = X
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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


""" TEST 12: block encoding of exp(-iHt) by LCU circuit, when H = Z """
def test_HamSim_LCU_Z(evolution_time = 0.5, error_tolerance=1e-3):
    
    ## Testing parameters
    num_qubits = 1
    H = Z
    
    ## Construct the block encoding operator
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    
    ## Build the QSP circuit
    obj = HamSim_byQET(num_qubits = num_qubits, 
                       BE_H = BE_H.data,
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



if __name__ == "__main__":
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
    test_HamSim_LCU_Z()
