import numpy as np
from numpy.linalg import norm
import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, partial_trace, state_fidelity, random_hermitian, random_unitary

from helper_functions import construct_BE
from __QSP_hamiltonian_simulation import HamSim_byQET


I = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])


def testing_HamSim_H(num_qubits: int,
                     H: np.ndarray,
                     evolution_time: float = 5.0,
                     truncation_order: int = 5,
                   ) -> (float, Statevector, Statevector):

    
    # if (num_qubits != int(np.log2(H.shape[0]))) or (num_qubits != int(np.log2(H.shape[1]))):
    #     print("The dimension of H and num_qubits do not match. Returning None.")
    #     return None, None, None
    
    # H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
    # print("BE successfully constructed!!")
    
    ## Benchmark method


    H = H / norm(H, 2)
    H = H / norm(H, 2)

    benchmark_circ = QuantumCircuit(num_qubits)
    benchmark_circ.hamiltonian(operator = Operator(H),
                               time = evolution_time,
                               qubits = [i for i in range(num_qubits-1, -1, -1)])
    # print(benchmark_circ.draw())

    # benchmark_circ = benchmark_circ.reverse_bits()

    bench_state = Statevector.from_int(0, 2**num_qubits)
    bench_state = bench_state.evolve(benchmark_circ)
    bench_state.draw("latex")
    bench_state.draw("city")

    
    ## QET method
    obj_HS_QET = HamSim_byQET(num_qubits = num_qubits, 
                              H = H,
                              evolution_time = evolution_time,
                              truncation_order = truncation_order,
                              error_tolerance=1e-6)

    obj_HS_QET.computeQSPPhaseAngles()
    # obj_HS_QET.plotQSPPhaseAngles()
    obj_HS_QET.buildCirc()
    # obj_HS_QET.drawCirc()

    output_state = obj_HS_QET.runHamiltonianSimulator()[0]
    # print(output_state)
    output_state.draw("latex")
    output_state.draw("city")
    
    # fidelity = state_fidelity(bench_state, output_state)
    fidelity = state_fidelity(DensityMatrix(bench_state), DensityMatrix(output_state))
    print("Fidelity between bench_state and qet_state = {}".format(fidelity))

    return fidelity, bench_state, output_state, obj_HS_QET


if __name__ == "__main__":

    list_fidelity = []

    num_qubits = 2
    evolution_time = 5.0
    truncation_order = 8

    # H = np.kron(np.kron(Y, X), Z)
    # H = np.kron(np.kron(np.kron(np.kron(Y, X), Z), Z), Y) ## takes 2 mins
    H = np.kron(Y, X) + np.kron(Z, Z) ## + np.kron(X, Y)
    # H = X + Y ## Good
    ## H = Z + X or Z + Y won't even compile
    # H = X + Z
    ## H = np.kron(Z, Z) +  np.kron(X, I) + np.kron(I, X)
    # H = Z

    time_start = time.perf_counter()

    for _ in range(1):
        fidelity, bench_state, output_state, obj_HS_QET = testing_HamSim_H(num_qubits, H = H,
                                                            evolution_time=evolution_time, truncation_order = truncation_order)
        list_fidelity.append(fidelity)


        print("Bench Energy", np.real(bench_state.data.conj().T @ (H / norm(H, 2)) @ bench_state.data))
        print("Output Energy", np.real(output_state.data.conj().T @ (H / norm(H, 2)) @ output_state.data))



    time_end = time.perf_counter()

    print("Time took:", time_end - time_start)
    print(list_fidelity)

    # bench_state.draw("latex")
    # output_state.draw("latex")

    print(obj_HS_QET.getEnergy())
    obj_HS_QET.saveFidelity("KarlLIN", time_steps = 5)
    obj_HS_QET.getOpenQASM("KARLLIN")