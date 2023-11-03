import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import BasicAer
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, partial_trace, state_fidelity, random_hermitian, random_unitary
from qiskit.extensions import UnitaryGate
from qiskit.visualization import plot_state_city

from .pyqsp_master.pyqsp.angle_sequence import QuantumSignalProcessingPhases
from .pyqsp_master.pyqsp.response import PlotQSPResponse, PlotQSPPhases

from .helper_functions import cos_xt_taylor, sin_xt_taylor, cos_xt_JA, sin_xt_JA, construct_BE_rand, construct_BE
from .helper_functions import PostMsmtError, QSPGenerationError, BlockEncodingError

import csv


class SignalProcessingOperators_OneBEQubit():
    """
    A class for building the QSP signal processing operators, also called the projector-controlled phase shift gates

    Params:
        ang_list (list): list of QSP phase angles (given by imported package PyQSP)    
    """
    def __init__(self,
                ang_list: list):

        if not isinstance(ang_list, list):
            raise ValueError("Error Msg")

        self.phase_ops = []
        self.phase_ops_op = []
        self.ang_list = ang_list

    def buildCircs(self):
        """
        Builds a list of projector-controlled phase shift gates, one for each QSP phase angles 
        """

        if not len(self.phase_ops) == 0:
            print("WARNING MSG: self.phase_ops is not empty - overwriting the old circuits now")
            
        master_list_ops = []
        master_list_ops_op = []
        
        for phi in self.ang_list:        
            circ = QuantumCircuit(2)
            circ.cx(1, 0, ctrl_state=0)
            circ.rz(2*phi, 0)
            circ.cx(1, 0, ctrl_state=0)
            
            master_list_ops.append(circ)
            master_list_ops_op.append(Operator(circ))

        self.phase_ops = master_list_ops
        self.phase_ops_op = master_list_ops_op

    def getCircs(self) -> list:
        return self.phase_ops
    
    def getOps(self) -> list:
        return self.phase_ops_op
    

class QET_LinLin():
    """
    A class for building a quantum signal processing (QSP)/ quantum eigenvalue transformation (QET) circuit, 
    which is mainly composed of alternating projector-controlled phase shift gates and block encoding of some input
    Hamiltonian (Hermitian matrix)

    Params:
        num_qubits (int): number of qubits the input Hamiltonian acts on
        BE_H (np.ndarray): block encoding matrix of some input Hamiltonian
        phase_angles (list): a list of QSP phase angles (given by imported package PyQSP)
    
    Outputs:
        Block encoding of the input Hamiltonian after its eigenvalues have been transformed by a polynomial function
        as specificed by the QSP phase angles 
        
    """
    def _check_if_properBE(self,
                          num_qubits: int,
                          num_BE_qubits: int,
                          BE_H: np.ndarray):

        ## Check if BE_H is unitary
        if not Operator(BE_H).is_unitary():
            print("ERROR: Input np.ndarray BE_H is not unitary")
            return False
    
        ## Check if n+m = BE_H.dimen
        BE_num_qubits = int(np.log2(BE_H.shape[0]))
        if not (num_qubits + num_BE_qubits) == BE_num_qubits:
            print("ERROR: Num of qubits of the Hamiltonian + num of Block-encoding qubits does not match with the number of qubits of BE_H")
            return False
    
        return True

    
    def __init__(self,
                 num_qubits: int,
                 BE_H: np.ndarray,
                 phase_angles: list,
                 num_BE_qubits: int = 1,
                 phase_angles_order: str = "desc",
                 real_poly: bool = True,
                 special_case: bool = False,
                ): 
        """
        Input:
            1. num_qubits
            type: int
            Number of qubits of the Hamiltonian

            2. BE_H
            type: np.ndarray
            Block encoding unitary of the Hamiltonian H

            3. phase_angles
            type: python list
            Ordered in Lin Lin's fashion

            4. num_BE_qubits
            type: int, default value = 1
            The number of qubits that block encode the Hamiltonian H
        """

        if not isinstance(num_qubits, int):
            raise ValueError("Parameter num_qubits should be of type int.")
        if not isinstance(BE_H, np.ndarray):
            raise ValueError("Parameter BE_H should be of type np.ndarray.")
        if not isinstance(phase_angles, list):
            raise ValueError("Parameter phase_angles should be of type list.")

        
        if not self._check_if_properBE(num_qubits, num_BE_qubits, BE_H):
            raise ValueError("ERROR MSG: BE_H is not a proper block encoding.")
        else:
            pass            
        
        self.num_qubits = num_qubits
        self.num_BE_qubits = num_BE_qubits
        self.BE_H = BE_H
        
        if phase_angles_order == "asce":
            phase_angles.reverse()
        self.phase_angles = phase_angles
        self.phase_angles_order = phase_angles_order

        self.real_poly = real_poly
        self.special_case = special_case

        self.d = len(phase_angles) - 1
        self.BE_Operator = None
        self.obj_phase = None
        self.main_circ = None
        self.main_circ_op = None


    def buildCirc(self):
        """
        Follows the QSP/QET Theorems to build the circuit
        """
        if self.main_circ is not None:
            print("WARNING MSG: self.main_circ is not None - overwriting the old QET circuit now")
            self.main_circ = None
            self.main_circ_op = None
        
        ## Quantum registers
        qreg_ctrl = QuantumRegister(1, "signal_ctrl")
        qreg_be = QuantumRegister(self.num_BE_qubits, "block_encode")
        qreg_input = QuantumRegister(self.num_qubits, "input")
                        
        qet_circ = QuantumCircuit(qreg_ctrl, qreg_be, qreg_input)

        ## Build an qiskit.quantum_info.Operator instance from BE_H 
        self.BE_Operator = Operator(self.BE_H)
        
        ## Build the projector-controlled phase shift gates
        self.obj_phase = SignalProcessingOperators_OneBEQubit(self.phase_angles)
        self.obj_phase.buildCircs()


        ## Implements the QSP/QET algorithms for when d is even
        if self.d % 2 == 0 and self.special_case is False:

            # print("d = {} is even. The theorem of QET for even d implemented.".format(self.d))

            ## Lin Lin Figure 7.9, for block encoding a real-valued polynomial
            if self.real_poly:
                qet_circ.h(qreg_ctrl)
            
            ## Since the order of the phase angles are descending, we need to transverse the list of angles reversely
            for i in range(self.d//2, 0, -1):
                qet_circ.append(self.obj_phase.getOps()[2*i],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                qet_circ.append(self.BE_Operator,
                                [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])                
                qet_circ.append(self.obj_phase.getOps()[2*i - 1],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                qet_circ.append(self.BE_Operator.adjoint(),
                                [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])
    
            qet_circ.append(self.obj_phase.getOps()[0],
                            [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])

            ## Lin Lin Figure 7.9, for block encoding a real-valued polynomial
            if self.real_poly:
                qet_circ.h(qreg_ctrl)

            self.main_circ = qet_circ
            self.main_circ_op = Operator(qet_circ)
            
            return
            
        ## Implements the QSP/QET algorithms for when d is odd
        elif self.d % 2 == 1 and self.special_case is False:

            # print("d = {} is odd. The theorem of QET for odd d implemented.".format(self.d))

            ## Lin Lin Figure 7.9, for block encoding a real-valued polynomial
            if self.real_poly:
               qet_circ.h(qreg_ctrl)      
            
            ## Since the order of the phase angles are descending, we need to transverse the list of angles reversely  
            for i in range((self.d - 1)//2, 0, -1):
                qet_circ.append(self.obj_phase.getOps()[2*i + 1],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                qet_circ.append(self.BE_Operator, 
                                [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])               
                qet_circ.append(self.obj_phase.getOps()[2*i],
                                [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])
                qet_circ.append(self.BE_Operator.adjoint(), 
                               [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])

            qet_circ.append(self.obj_phase.getOps()[1],
                            [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])            
            qet_circ.append(self.BE_Operator, 
                            [qreg_input[i] for i in range(self.num_qubits-1, -1, -1)] + [qreg_be[i] for i in range(self.num_BE_qubits-1, -1, -1)])   
            qet_circ.append(self.obj_phase.getOps()[0],
                            [qreg_ctrl] + [qreg_be[i] for i in range(self.num_BE_qubits)])

            ## Lin Lin Figure 7.9, for block encoding a real-valued polynomial
            if self.real_poly:
               qet_circ.h(qreg_ctrl)
            
            self.main_circ = qet_circ
            self.main_circ_op = Operator(qet_circ)
            return 
        
        ## Handles the edge case when evolution_time = 0.
        elif self.special_case is True:
            ## So far this only works for 1 BE qubit
            if self.num_BE_qubits > 1:
                raise ValueError('We cant do more than one RN!')
            else:
                big_X_block_size = 2 ** (self.num_qubits)
                big_X_matrix = np.block([
                    [np.zeros((big_X_block_size, big_X_block_size)), np.eye(big_X_block_size)],
                    [np.eye(big_X_block_size), np.zeros((big_X_block_size, big_X_block_size))]])
                
            big_X = UnitaryGate(Operator(big_X_matrix), label = 'Big X')
 
            qet_circ.append(big_X, [i for i in range(1,1+self.num_BE_qubits + self.num_qubits)])
 
            self.main_circ = qet_circ
            self.main_circ_op = Operator(qet_circ)
            return
            
            
    def getCirc(self):
        if self.main_circ is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
        return self.main_circ

    def getOp(self):
        if self.main_circ_op is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
        return self.main_circ_op
        
    def _getDecomposedCirc(self):
        if self.main_circ is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
            return None
        return self.main_circ.decompose() 
    
    def drawCirc(self,
                output: str = "text",
                decomp: bool = False):
        if self.main_circ is None:
            print("QET circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
            return None

        if decomp:
            print(self.main_circ.decompose().draw(output=output))
        else:
            print(self.main_circ.draw(output=output))


class HamSim_byQET():
    """
    A class for building a block encoding of Schrodinger's time evolution operator, exp(-iHt), via a LCU circuit.
    This is accomplished by defining one QSP/QET circuit that block encodes matrix cos(Ht) and another that block encodes matrix sin(Ht),
    then construct a block encoding of matrix cos(Ht) -i*sin(Ht) = exp(-iHt) via a simple LCU circuit.

    Params:
        num_qubits (int): number of qubits the input Hamiltonian acts on
        H (np.ndarray): input Hamiltonian H
        evolution_time (float): time parameter t in the time evolution operator, exp(-iHt)
        starting_state (np.ndarray): the initial state vector at t = 0.0
    
    Outputs:
        Block encoding of Schrodinger's time evolution operator, exp(-iHt)
    """

    def _check_if_properBE(self,
                          num_qubits: int,
                          num_BE_qubits: int,
                          BE_H: np.ndarray):

        ## Check if BE_H is unitary
        if not Operator(BE_H).is_unitary():
            print("ERROR: Input np.ndarray BE_H is not unitary")
            return False
    
        ## Check if n+m = BE_H.dimen
        BE_num_qubits = int(np.log2(BE_H.shape[0]))
        if not (num_qubits + num_BE_qubits) == BE_num_qubits:
            print("ERROR: Num of qubits of the Hamiltonian + num of Block-encoding qubits does not match with the number of qubits of BE_H")
            return False
    
        return True
    
    def __init__(self,
                num_qubits: int,
                H : np.ndarray,
                evolution_time: float,
                num_BE_qubits: int = 1,
                truncation_order: int = 8,
                approx_method: str = "Jacobi-Anger",
                error_tolerance: float = 1e-6,
                starting_state: np.ndarray = None,
                simulator_BasicAer: bool = False,
                ):
        
        self.num_qubits = num_qubits
        self.num_BE_qubits = num_BE_qubits

        H, BE_H = construct_BE(num_qubits=num_qubits, H=H)
        self.H = H.data
        self.BE_H = BE_H.data

        self.evolution_time = evolution_time
        self.truncation_order = truncation_order
        self.approx_method = approx_method
        
        self.error_tolerance = error_tolerance
        
        ## checking if starting state has the correct number of qubits
        if starting_state is not None:
            if len(starting_state) != 2 ** self.num_qubits:
                raise ValueError("Initial state has to be of size of {} qubits".format(2 ** self.num_qubits))
            
        self.starting_state = starting_state
        self.simulator_BasicAer = simulator_BasicAer

        self.cos_coeffs = None
        self.cos_ang_seq = None
        
        self.sin_coeffs = None
        self.sin_ang_seq = None

        self.obj_QET_cos = None
        self.obj_QET_sin = None
        
        self.HamSim_circ = None

        self.StateVector = None
        self.StateVector_reverse = None


    def computeQSPPhaseAngles(self):

        """
        This function calls the imported package PyQSP to compute the QSP phase angles that approximate cos(xt) and sin(xt).
        Since the PyQSP function being called, QuantumSignalProcessingPhases(), is a rather fragile and unstable method, 
        we decided to put QuantumSignalProcessingPhases() in a try statement and run it at most 5000 times until it succeeds.
        If the angles still cannot be computed after 5000 attempts, we raise an QSPGenerationError().
        """

        if (self.cos_ang_seq is not None) and (self.sin_ang_seq is not None):
            print("QSP phase angles have been generated. Returning None now.")
            return

        if self.approx_method == "Taylor":
            self.cos_coeffs, _ = cos_xt_taylor(t = self.evolution_time,
                                               d = self.truncation_order)
        
            self.sin_coeffs, _ = sin_xt_taylor(t = self.evolution_time,
                                               d = self.truncation_order)        
        elif self.approx_method == "Jacobi-Anger":
            self.cos_coeffs, _ = cos_xt_JA(t = self.evolution_time,
                                           d = self.truncation_order)
        
            self.sin_coeffs, _ = sin_xt_JA(t = self.evolution_time,
                                           d = self.truncation_order)
        
        """ Try to compute the QSP phase angles for at most 5000 times. After 5000 rounds of failures, raise QSPGenerationError()"""
        for _ in range(5000):
            try:
                if self.evolution_time > 0:
                    self.cos_ang_seq = QuantumSignalProcessingPhases(self.cos_coeffs, signal_operator="Wx", tolerance=self.error_tolerance)
                    self.sin_ang_seq = QuantumSignalProcessingPhases(self.sin_coeffs, signal_operator="Wx", tolerance=self.error_tolerance)
                
                ## Handles the special case of evolution_time = 0.0
                else:
                    self.cos_ang_seq, self.sin_ang_seq = [0], [None]
            
            except:
                self.cos_ang_seq = None
                self.sin_ang_seq = None
                continue
            else:
                return

        raise QSPGenerationError("""FAILURE: QSP phase angles failed to be generated after 5000 tries.
Methods for finding QSP angles become numerically unstable when the evolution time or truncation order is too high. 
We recommand reducing the evolution time to be less than 5.0 and the truncation order to less than or equal to 5.""")



    
    def plotQSPPhaseAngles(self):
        if (self.cos_ang_seq is None) or (self.sin_ang_seq is None):
            print("QSP phase angles haven't been computed yet. To compute the angles, please call self.computeQSPPhaseAngles(). Returing None now.")
            return None

        print("QSP phase angles for cos(xt) when t = {} are\n{}\n".format(self.evolution_time, self.cos_ang_seq))
        print("The plot for cos(xt) phase angles:")
        PlotQSPPhases(self.cos_ang_seq)
        print("The plot for QSP response function for cos(xt):")
        PlotQSPResponse(self.cos_ang_seq, target= lambda x: np.cos(x * self.evolution_time), signal_operator="Wx")

        print("QSP phase angles for sin(xt) when t = {} are\n{}\n".format(self.evolution_time, self.sin_ang_seq))
        print("The plot for sin(xt) phase angles:")
        PlotQSPPhases(self.sin_ang_seq)
        print("The plot for QSP response function for sin(xt):")
        PlotQSPResponse(self.sin_ang_seq, target= lambda x: np.sin(x * self.evolution_time), signal_operator="Wx")

    
    def buildCirc(self):
        if (self.cos_ang_seq is None) or (self.sin_ang_seq is None):
            print("QSP phase angles haven't been computed yet. To compute the angles, please call self.computeQSPPhaseAngles(). Returing None now.")
            return None
        if self.HamSim_circ is not None:
            print("WARNING MSG: self.main_circ is not None - overwriting the old QET circuit now")
            self.HamSim_circ = None
        
        ## Quantum registers
        qreg_HS_ctrl = QuantumRegister(1, "ctrl_LCU")
        qreg_QET_input = QuantumRegister(1 + self.num_BE_qubits + self.num_qubits)

        main_circ = QuantumCircuit(qreg_HS_ctrl, qreg_QET_input)

        ## calling the QET_LinLin() class twice, one to block encode cos(Ht), the other to block encode sin(Ht)
        obj_QET_cos = QET_LinLin(num_qubits = self.num_qubits,
                                 BE_H = self.BE_H,
                                 phase_angles = self.cos_ang_seq,
                                 num_BE_qubits = self.num_BE_qubits)
        ## Special case for t = 0.0
        if self.sin_ang_seq == [None]:
            obj_QET_sin = QET_LinLin(num_qubits = self.num_qubits,
                                    BE_H = self.BE_H,
                                    phase_angles = self.sin_ang_seq,
                                    num_BE_qubits = self.num_BE_qubits,
                                    special_case = True)
        else:
            obj_QET_sin = QET_LinLin(num_qubits = self.num_qubits,
                                    BE_H = self.BE_H,
                                    phase_angles = self.sin_ang_seq,
                                    num_BE_qubits = self.num_BE_qubits)

        
        ## Saving the two objects                        
        self.obj_QET_cos = obj_QET_cos
        self.obj_QET_sin = obj_QET_sin

        obj_QET_cos.buildCirc()
        obj_QET_sin.buildCirc()
        
        ctrl_op_cos = obj_QET_cos.getCirc().to_gate().control(num_ctrl_qubits=1, ctrl_state="0")
        op_sin = obj_QET_sin.getCirc().copy()        
        op_sin.append(Operator((0-1j) * np.identity(2 ** (1 + self.num_BE_qubits + self.num_qubits))),
                      [i for i in range(1 + self.num_BE_qubits + self.num_qubits-1, -1, -1)]) 
        ctrl_op_sin = op_sin.to_gate().control(num_ctrl_qubits=1, ctrl_state="1")        


        ## Build the LCU circuit        
        main_circ.h(qreg_HS_ctrl)
        main_circ.append(ctrl_op_cos,
                        [qreg_HS_ctrl] + [qreg_QET_input[i] for i in range(1 + self.num_BE_qubits + self.num_qubits)])
        main_circ.append(ctrl_op_sin,
                        [qreg_HS_ctrl] + [qreg_QET_input[i] for i in range(1 + self.num_BE_qubits + self.num_qubits)])
        main_circ.h(qreg_HS_ctrl)
                
        self.HamSim_circ = main_circ

        return 
    

    def getCirc(self):
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")           
        return self.HamSim_circ

    
    def drawCirc(self,
                output: str = "text",
                decomp: bool = False):
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")
            return None

        if decomp:
            print("The (decomposed) circuit for Hamiltonian Simulation by QET.")
            print(self.HamSim_circ.decompose().draw(output=output))
            print("The (decomposed) QET circuit for cos(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order, self.truncation_order, self.truncation_order, 2*self.truncation_order+1))
            print(self.obj_QET_cos.drawCirc(output=output, decomp=decomp))
            print("The (decomposed) QET circuit for sin(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order+1, self.truncation_order+1, self.truncation_order, 2*self.truncation_order+2))
            print(self.obj_QET_sin.drawCirc(output=output, decomp=decomp))
            
        else:
            print("The circuit for Hamiltonian Simulation by QET.")
            print(self.HamSim_circ.draw(output=output))
            print("The QET circuit for cos(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order, self.truncation_order, self.truncation_order, 2*self.truncation_order+1))
            print(self.obj_QET_cos.drawCirc(output=output, decomp=decomp))
            print("The QET circuit for sin(Ht) with t = {} and truncation order = {}, with {} BE operators, {} BE.adjoint operators, and {} QSP phase operators"\
                  .format(self.evolution_time, 2*self.truncation_order+1, self.truncation_order+1, self.truncation_order, 2*self.truncation_order+2))
            print(self.obj_QET_sin.drawCirc(output=output, decomp=decomp))
    

    
    def runHamiltonianSimulator(self) -> (Statevector, Statevector):
        """ 
        Returns the evoluted state vector after evolution_time 
        """
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")           
            return self.HamSim_circ

        if self.simulator_BasicAer is not True:
            
            if self.starting_state is None:
                state = Statevector.from_int(0, 2 ** (1 + 1 + self.num_BE_qubits + self.num_qubits))
            else:                               
                starting_state = Statevector(self.starting_state)
                if not starting_state.is_valid():
                    raise ValueError("The starting state {} is not a valid quantum state.".format(self.starting_state))
                
                starting_state = starting_state.reverse_qargs().data
                state = Statevector(np.kron(starting_state, np.array([1., 0., 0., 0., 0., 0., 0., 0.])))
           
            ## Evolve state vector under the LCU circuit
            state = state.evolve(self.HamSim_circ)

        else:
            simulator = BasicAer.get_backend("statevector_simulator")
            state = Statevector(simulator.run(transpile(self.HamSim_circ, simulator)).result().get_statevector())
            
        """
        Post-selection on the input register begins. If the desired measurement outcome (all 0's state) on the HamSim_ctrl, signal_ctrl, and BE registers is not obtained after 1000 tries, an PostMsmtError is raised.
        For post-selection, run the HamSim circuit for at most 1000 times; if the desired msmt outcome (all 0's state) cannot be obtained, raise a PostMsmtError(Exception)
        """
        for _ in range(1000):       
            msmt_result, post_msmt_state = state.measure([i for i in range(1 + 1 + self.num_BE_qubits)])
          
            if msmt_result != "0" * (1 + 1 + self.num_BE_qubits):
                continue
            else:
                ## Trace out the measured qubits
                density_matrix = partial_trace(post_msmt_state, [i for i in range(1 + 1 + self.num_BE_qubits)])

                self.StateVector = density_matrix.to_statevector()
                self.StateVector_reverse = self.StateVector.reverse_qargs()
                
                return self.StateVector, self.StateVector_reverse

        raise PostMsmtError("Post-selection after 1000 rounds failed.")
        return None
    

    def getHamiltonian(self) -> np.ndarray:
        return self.H
    

    def getEnergy(self) -> float:
        """
        Computes the total energy of the system described by the input Hamiltonian at time evolution_time
        """        
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.")           
            return self.HamSim_circ
        
        if self.StateVector is None or self.StateVector_reverse is None:
            print("Statevector hasn't been computed yet. To compute StateVector, please call self.runHamiltonianSimulator(). Returing None now.")           
            return self.HamSim_circ

        return np.real(self.StateVector_reverse.data.conj().T @ self.H @ self.StateVector_reverse.data)


    def saveFidelity(self,
                    filename: str,
                    max_time: float = None,
                    time_steps: int = 50):
        """
        Calculates the fidelity between the statevectors computed by QSP and numerical method, and save them in a .csv file.
        """

        max_time = self.evolution_time if max_time == None else max_time

        user_check = ""
        while user_check.upper() != "YES" and user_check.upper() != "NO": 
            user_check = input("WARNING: QSP could take a lot of time (e.g. 2-3 hrs for a 5-qubit Hamiltonian) executing this function. Would you like to continue? [yes/no]: ")
        
        if user_check.upper() == "NO":
            print("Returning None.")
            return
        else:
            
            fidelity_list = [1.0]
            
            for t in np.linspace(0, max_time, time_steps)[1:]:

                ## Numerical/Qiskit method: using the QuantumCircuit.hamiltonian function
                numerical_circ = QuantumCircuit(self.num_qubits)
                numerical_circ.hamiltonian(operator = Operator(self.H),
                                          time = t,
                                          qubits = [i for i in range(self.num_qubits-1, -1, -1)])


                bench_state = Statevector.from_int(0, 2 ** self.num_qubits)
                bench_state = bench_state.evolve(numerical_circ)
                
                ## QET method
                obj_HS_QET = HamSim_byQET(num_qubits = self.num_qubits, 
                                          H = self.H,
                                          evolution_time = t)

                obj_HS_QET.computeQSPPhaseAngles()
                obj_HS_QET.buildCirc()
                output_state, _ = obj_HS_QET.runHamiltonianSimulator()

                fidelity = state_fidelity(DensityMatrix(bench_state), DensityMatrix(output_state))
                fidelity_list.append(fidelity)

            data = [(a, x, y) for a, (x, y) in enumerate(zip(np.linspace(0, max_time, time_steps), fidelity_list))]
            
            with open("{}_QSP_fidelity.csv".format(filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "time", "fidelity"])
                writer.writerows(data)

            print("Data saved to {}_QSP_fidelity.csv".format(filename))
            return

    def saveEnergy(self, filename, max_time=None, time_steps=50):
        max_time = self.evolution_time if max_time == None else max_time
        return

    def getOpenQASM(self, filename: str):
        if self.HamSim_circ is None:
            print("HamSim circuit hasn't been built yet. To build the circuit, please call self.buildCirc(). Returing None now.") 
            return
        
        self.getCirc().qasm(filename="LCU_qasm_{}.txt".format(filename))
        self.obj_QET_cos.getCirc().qasm(filename="QSP_cos_qasm_{}.txt".format(filename))
        self.obj_QET_sin.getCirc().qasm(filename="QSP_sin_qasm_{}.txt".format(filename))

        print("Files saved as LCU_qasm_{}.txt, QSP_cos_qasm_{}.txt, and QSP_sin_qasm_{}.txt!".format(filename, filename, filename))
        return