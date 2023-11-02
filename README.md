# Hamiltonian Simulation Coding Project
This coding project is aimed at conducting Hamiltonian simulaiton via three mehtods:
1. Numerical (classical) calculation
2. Trottersiation Method
3. Quantum Signal Processing Algorithm

## Installation
First, create and active a Python virtual environment:
```
conda create -n <name> python=3.9
conda active <name>
```
Next, clone the repository and checkout the `final_submission` branch:
```
git clone https://github.com/GabrielWaite/QSP-Coding.git
cd QSP-Coding
git checkout final_submission
```
Then, pip install the required dependencies:
```
pip install -r requirement.txt
```
To run the file,ensure you are in the correct directory (`QSP-Coding`) then call the file `main.py`:
```
python main.py
```

## Taking User Inputs
If you have read and understood the documentation for the modules, the user inputs should be fairly trivial. The example we will work with is a trivial one.
Our input Hamiltonian is given by the hash_map = {'xx':1}. (Assume we do not know the Hamiltonian matrix explicitly)
Our execution time is 3.14
We do not have a block encoded matrix. We do not have the Hamiltonian matrix.
Our desired outputs are option '0' - statevector at time t

The first of a series of questions prompts:
1. *Enter the initial state in [...] format:*
```
Enter the initial state in [...] format: [1,1,1,1]
```

2. *Enter the time between 0 and 5 to execute*
```
Enter the time between 0 and 5 to execute : 3.14
```

3. *Enter the number of qubits*
```
Enter the number of qubits 2
```

4. *Do you have the block-encoded matrix? (y/n)*
```
Do you have the block-encoded matrix? (y/n) n
```

5. *Do you have the hamiltonian matrix? (y/n)*
```
Do you have the hamiltonian matrix? (y/n) n
```

6. *Provide a list of keys to perform desired tasks based on the following legend:*
    *0: statevector at time t*
    *1: energy at time t*
    *2: Energy evolution data up to max time*
    *3: fidelity evolution data up to max time*
                    *:*
```
Provide a list of keys to perform desired tasks based on the following legend:
0: statevector at time t
1: energy at time t
2: Energy evolution data up to max time
3: fidelity evolution data up to max time
                :[0]
```

7. *Provide a file name for your state vector data:*
```
Provide a file name for your state vector data: demo
```
8. *Provide a file name for your quantum circuit data:*
```
Provide a file name for your quantum circuit data: demo_qc
```
### Example Output
For the example just shown, **three** files will be returned:
1. `statevector_demo_metadata.txt`
2. `trotter_qasm_demo_qc.txt`
3. `qsp_qasm_demo_qc.txt`

## General Output
By default, the Trotter and QSP ```QISKIT``` OpenQASM files will be saved.

Based on tasks requested at question 6, a number of files will be saved also.

- 0: metadata.txt file with statevector information
- 1: metadata.txt file with energy informaiton
- 2: metadata.txt file for the problem setting and a data.csv file with the energy evolution data for the spread of times
- 3: metadata.txt file for the problem setting and a data.csv file with the fidelity evolution data for the spread of times

If '[]' is passed at question 6 then only the quantum circuit for the problem setting is saved.
## Testing
To test the numerical and Trotter Hamiltonian simulation module, run:
```
python tests/numerical_trotter_testing.py
```
To test the quantum signal processing Hamiltonian simulation module, run:
```
python tests/QSP_testing.py
```
## Contributions
This code has three main contributors. The main contributions of each author is as follows (not exhaustive contribution list):
- Gabriel Waite: numerical and Trotter Hamiltonian simulation code
- Karl Lin: Quantum signal processing Hamiltonian simulation codes
- Soumya Sarkar: Inputs from the user and required checkings
## Marking
Please note that we've decided to use the public package PyQSP (https://github.com/ichuang/pyqsp) to generate the QSP phase angles. Therefore, all files under the folder name "pyqsp_master" are not of our own work and can be skipped over when assessing our codes. The rest of files are assessable.
## References
- Grand Unification of Quantum Algorithms. John M. Martyn, Zane M. Rossi, Andrew K. Tan, and Isaac L. Chuang. PRX Quantum 2, 040203 – Published 3 December 2021
- PyQSP (https://github.com/ichuang/pyqsp)
- Finding Angles for Quantum Signal Processing with Machine Precision. Rui Chao, Dawei Ding, András Gilyén, Cupjin Huang, and Mario Szegedy. arXiv preprint arXiv:2003.02831 (2020).
- angle-sequence (https://github.com/alibaba-edu/angle-sequence)
