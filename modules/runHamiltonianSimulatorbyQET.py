import numpy as np
from numpy.linalg import norm, eigvals, eig

from qiskit import QuantumCircuit, qasm
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, state_fidelity

import time
import json
from sympy import Matrix
import matplotlib.pyplot as plt

from .helper_functions import construct_BE_rand, construct_BE, convert_binary_to_int
from .HamiltonianSimulation_by_QET import HamSim_byQET


def run_HamSim_H(num_qubits: int,
                H: np.ndarray,
                evolution_time: float = 5.0,
                truncation_order: int = 8,
                starting_state: np.ndarray = None,
                ) -> (Statevector, Statevector):

    
    H, BE_H = construct_BE(num_qubits=num_qubits, H=H)

    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                              BE_H = BE_H.data,
                              evolution_time = evolution_time,
                              truncation_order = truncation_order,
                              error_tolerance=1e-6,
                              starting_state = None)

    obj_HS_QET.computeQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    # QSP_circ = obj_HS_QET.getCirc()

    output_state, output_state_reversed = obj_HS_QET.runHamiltonianSimulator()
    # QSP_circ.qasm(filename="QSP_circuit")


    return output_state, output_state_reversed