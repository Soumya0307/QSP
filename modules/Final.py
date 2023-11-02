#!/usr/bin/env python
# coding: utf-8

# # INPUTS AND CHECKS

# In this Notebook, we take all the required inputs and perform all the necessary checks to use those for the rest of our code. The main obstacle of this code is to perform the block encoding efficiently. Though, for our case, as we are restricting our user to input only Pauli string as Hamiltonian ($H = \sum_j \alpha_j(\sigma_1^{\mu}\otimes \cdots \otimes\sigma_n^{\nu})_j$), it would be reasonable to block encode the provided Hamiltonian following the convention
# 
# 
# $$
# \begin{equation}
# \begin{pmatrix}
# H & i\sqrt{1-H^2}\\
# i\sqrt{1-H^2} & -H
# \end{pmatrix}
# \end{equation}
# $$
# <br>
# 
# 
# where $H$ is the Hamiltonian the user wants to simulate.
# 
# To give flexibility to the user, we go for three options.
# 
# 1. The user knows the block encoded matrix, the Hamiltonian matrix and the Hamiltonian hash map.
# 2. The user knows about the Hamiltonian matrix and the Hamiltonian hash map.
# 3. The user only knows the hash map.
# 
# For the Trotterization and Numerical calculations we do not need the block encoding of the Hamiltonian, we need it only for the QSP part. We add a block diagram to showcase the input model for our code.
# ***
# 
# ![yuval2.png](attachment:yuval2.png)
# 
# ***
# 
# We define some global variables which can be used further when calling a class or method in the rest of the code. At the end, when the user calls `user()` function the user will be asked the questions one by one as every other function is called in that.

# ## Global Variables

# Let's define some global variables. These are used to save the inputs from the user and use these for further computations

# In[3]:


initial_state = None            #(1D matrix)      #The intial state is assigned to this global variable
execution_time = None           #(integer)        #The time of execution (0,5) is assigned to this global variable
qubit_number = None             #(integer)        #The number of qubits is assigned to this global variable
trotter_step = None             #(integer)        #The number of Trotter steps is assigned to this global variable

error_global = None             #(float)          #The user inputs an error tolerance upto which an approximated unitary is made work as a perfect unitary, that value is assigned to this global variable
unitary_matrix = None           #(2D matrix)      #If the user knows the block-encoded matrix, that is assigned to this global variable. This can be an approximated unitary matrix

hamitlonian = None              #(2D matrix)      #If the user knows the hamiltonian matrix, that is assigned to this global variable
normalized_hamiltonian = None   #(2D matrix)      #The hamiltonian matrix is normalized using Operator norm and that is assigned to this global variable

hash_map = None                 #(Dictionary)     #The hash map is assigned to this global variable
U_block = None                  #(2D matrix)      #If the user does not know the block encoded matrix beforehand, we block encode that, and that is assigned to this global variable


# In[ ]:





# In[32]:


def other_inputs():
    '''
    -> other_inputs()
    -> functionailty: Takes an initial state, execution time, trotter steps, qubit numbers as inputs from the user
    -> Return: None
    -> Description: It takes the inputs from the user and assigns the values to the global variables
    '''
    global initial_state
    global execution_time
    global qubit_number
    global trotter_step 
    step_number = int(input("Write doen the trotter-step "))
    trotter_step = step_number
    state = np.array(eval(input("Enter the initial state in [...] format: ")))
    if state.ndim != 1:
        raise TypeError("Please enter a 1D matrix")
    initial_state = state
    time = float(input("Enter the time between 0 and 5 to execute "))
    if time>5.0 or time<0.0:
        raise ValueError("Please choose the execution time between 0 and 5 ")
    execution_time = time
    qubit = int(input("Enter the number of qubits "))
    if qubit > 5 or qubit < 0:
        raise TypeError("Please choose qubit number between 0 and 5")
    if qubit != np.log2(len(state)):
        raise ValueError('You cannot have', len(state) ,'with', qubit, 'qubit(s)')
    qubit_number = qubit


# In[5]:


#Input: Block Encoded Matrix, Check: It's unitary and square
import numpy as np
def get_matrix():

    '''
    -> get_matrix()
    -> functionailty: Takes matrix (string in [[],[],...] format) and error (float) as inputs from the user
    -> Return: The Matrix and error
    -> Description: It takes a string written in the mentioned format and error from user, converts the string into an array
    and returns the inputs. We did this so that if a user has a matrix as an output in python, he/she can just can copy and
    paste it in the prompt message or write it as string instead of putting each element one by one. The function also stores
    the matrix in global variable
    '''
    global unitary_matrix
    global error_global
    M = np.array(eval(input("Enter the block encoded matrix in [[],[],..] format: ")))
    if M.shape[0] != 4*qubit_number:
        raise ValueError("The dimension of the block encoded matrix cannot be performed with this numner of qubits")
    err = float(input("Input the tolerance "))
    error_global = err
    unitary_matrix = M

    return M, err

#matrix, err = get_matrix()  To store the input matrix and use it further in the code

def is_square(M):
    '''
    -> is_square()
    -> functionailty: Takes matrix (array) and checks if the number of rows is equal to the number of columns
    -> Return: True/False
    -> Description: It takes the 2D array M and checks if the number of rows ('M.shape[0]') is equal to the number of columns
    (M.shape[1]), if they are equal, it returns 'True', else it returns 'False'
    '''
    return M.shape[0] == M.shape[1] #Check if it's square

def is_unitary(M = None,error = None):
    '''
    -> is_unitary()
    -> functionailty: Takes matrix (array) and checks if it's unitary given the allowed error
    -> Return: True/False
    -> Description: It takes the 2D array M and calculates the transpose conjugate of it (MT) and performs the dot
       product between M and MT. If the dot product is close enough to the identity matrix provided the error
       tolerance then it returns 'True' indicating M is unitary, else it returns 'False'
    '''
    global unitary_matrix
    global product
    if M is None:
        M = unitary_matrix
    if error is None:
        error = error_global
    MT = M.conjugate().T
    prod = np.dot(M, MT)
    for i in range(prod.shape[0]):
        for j in range(prod.shape[1]):
            if i == j:  # Diagonal element
                if abs(prod[i, j] - 1) < error:
                    prod[i, j] = 1
            else:  # Off-diagonal element
                if abs(prod[i, j]) < error:
                    prod[i, j] = 0
    if np.array_equal(prod, np.identity(prod.shape[0])):
        print("The block_encoded matrix is unitary")
    else:
        raise ValueError("The block encoded matrix is not unitary under the threshold, check again!")
    product = prod
    unitary_matrix = M

    return M, prod

def is_square_unitary(M = None, error = None):
    '''
    -> is_square_unitary()
    -> functionailty: Takes matrix (array) and error, checks if it's square and unitary given the allowed error
    -> Return: True/False
    -> Description: This is created specially for testing purpose.

    1. this function takes the arguements matrix M (array) and error (float) if that's provided otherwise it just calls
       get_matrix() i.e. it takes input from the user.
    2. Then it checks if the matrix fulfills the square restriction by calling is_square(), and also checks is the matrix is
       unitary provided the allowed error.

       If both constrains are fulfilled by the matrix, it returns 'True', else 'False'
    '''

    if M is None and error is None:
        M, error = get_matrix()
    if not is_square(M):
        raise AssertionError("Test Failed: Not square")

    if not is_unitary(M,error):
        raise AssertionError("Test Failed: Not unitary")
    return True


def test1_unitarity():       #tests and shows that it returns True when the matrix is square and unitary under allowed error
    '''
    -> test1_unitarity()  [True case: Unitary, square]
    -> functionailty: Takes a particular matrix (array) and error, checks if it's square and unitary given the allowed error
    -> Return: True (Test Passed)/ test Failed
    -> Description: This shows if our function is_square_unitary() works correctly by calling the function and running it on
    the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        inp = np.array([[0.99,0.01],[0.02,0.99]])
        err = 0.1
        expected = is_square_unitary(inp, err)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as error:
        print(error)
        return False

def test2_not_unitarity():    #tests and shows that it returns False when the matrix is square but not unitary under allowed error
    '''
    -> test2_not_unitarity()  [False case: Unitary, True case: square]
    -> functionailty: Takes a particular matrix (array) and error, checks if it's square and unitary given the allowed error
    -> Return: True (Test Passed)/ test Failed
    -> Description: This shows if our function is_square_unitary() works correctly by calling the function and running it on
    the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        inp = np.array([[0.99,0.01],[0.02,0.99]])
        err = 0.01
        expected = is_square_unitary(inp, err)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as error:
        print(error)
        return False

def test3_not_square():      #tests and shows that it returns False when the matrix is not square and hence doesn't go on checking unitarity
    '''
    -> test3_not_square()  [False case: square]
    -> functionailty: Takes a particular matrix (array) and error, checks if it's square and unitary given the allowed error
    -> Return: True (Test Passed)/ test Failed
    -> Description: This shows if our function is_square_unitary() works correctly by calling the function and running it on
    the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        inp = np.array([[0.99,0.01]])
        err = 0.1
        expected = is_square_unitary(inp, err)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as error:
        print(error)
        return False


# In[6]:


#Taking the Hamiltonian matrix as input from the user and checking if it's Hermitian and square.

import numpy as np

def get_hamiltonian_matrix():

    '''
    -> get_hamiltonian_matrix()
    -> functionailty: Takes matrix (string in [[],[],...] format) as input from the user
    -> Return: The Matrix and error
    -> Description: It takes a string written in the mentioned format and error from user, converts the string into an array
    and returns the inputs. We did this so that if a user has a matrix as an output in python, he/she can just can copy and
    paste it in the prompt message or write it as string instead of putting each element one by one
    '''
    global hamiltonian
    H = np.array(eval(input("Enter the Hamiltonian matrix in [[],[],..] format: "))) #converts the string in np.array format and returns as array
    if H.shape[0] != 2*qubit_number:
        raise ValueError("The dimension of the Hamiltonian matrix cannot be performed with this numner of qubits")
    hamiltonian = H

    return H


def is_hermitian(H):
    '''
    -> is_hermitian()
    -> functionailty: Takes matrix (array) and checks if it's hermitian
    -> Return: True/False
    -> Description: It takes the 2D array H and calculates the transpose of it (HT) and checkes if H and it's transpose
       conjugate are same. If they are same then it returns 'True' indicating H is Hermitian, else it returns 'False'
    '''
    return np.array_equal(H, np.conj(H).T)

def normalized_Hamiltonian(H1):
    '''
    -> normalized_Hamiltonian()
    -> functionailty: Takes hamiltonian matrix (array) and returns the normalized matrix
    -> Return: Normalized matrix
    -> Description: It takes the 2D array H and calculates the Frobenius norm of it. It then returns the normalized Hamiltonian
       matrix
    '''
    global normalized_hamiltonian
    H = H1/np.linalg.norm(H1, ord=2)
    normalized_hamiltonian = H
    return H

def is_square_hermitian(H = None):
    '''
    -> is_square_hermitian()
    -> functionailty: Takes matrix (array) and error, checks if it's square, hermitian
    -> Return: True/False
    -> Description: This is created specially for testing purpose.

    1. this function takes the arguements matrix M (array) if that's provided otherwise it just calls get_matrix() i.e. it
       takes input from the user.
    2. Then it checks if the matrix fulfills the square restriction by calling is_square(), and also checks is the matrix is
       unitary provided the allowed error.

       If all constrains are fulfilled by the matrix, it returns 'True', else 'False'
    '''

    if H is None:
        H = get_hamiltonian_matrix()
    if not is_square(H):
        raise AssertionError("Test Failed: Not square")
    if not is_hermitian(H):
        raise AssertionError("Test Failed: Not hermitian")
    H2 = normalized_Hamiltonian(hamiltonian)
    return True

def test4_hermitian():     #tests and shows that it returns True when the matrix is hermitian
    '''
    -> test4_hermitian() [True case]
    -> functionailty: Takes a particular matrix (array), checks if it's hermitian
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function is_square_hermitian() works correctly by calling is_hermitian() function 
    and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        inp = np.array([[1,4],[4,1]])
        expected = is_hermitian(inp)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as err:
        print(err)
        return False

def test5_not_hermitian():     #tests and shows that it returns False when the matrix is not hermitian
    '''
    -> test5_not_hermitian()  [False case]
    -> functionailty: Takes a particular matrix (array), checks if it's hermitian
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function is_square_hermitian() works correctly by calling is_hermitian() function 
    and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        inp = np.array([[1,0],[-4,1]])
        expected = is_hermitian(inp)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as err:
        print(err)
        return False

def test6_normalised():       #tests and shows that it returns True when the matrix is normalized according to Operator norm
    '''
    -> test6_normalised()  [True case]
    -> functionailty: Takes a particular matrix (array), checks if it's normalized correctly according to Operator norm
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function is_square_hermitian() works correctly by calling the normalized_Hamiltonian()
    function and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise 
    AssertionError.
    '''
    try:
        inp = np.array([[1,-6],[-6,4]])
        expected = normalized_Hamiltonian(inp)
        calculated = np.array([[ 0.11514558, -0.69087346],[-0.69087346,  0.4605823 ]])
        assert np.allclose(calculated, expected)
        return True
    except AssertionError:
        return False

def test7_not_normalised():     #tests and shows that it returns False when the matrix is not normalized
    '''
    -> test7_normalised()  [False case]
    -> functionailty: Takes a particular matrix (array), checks if it's normalized correctly according to Operator norm
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function is_square_hermitian() works correctly by calling the normalized_Hamiltonian()
    function and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise 
    AssertionError.
    '''
    try:
        inp = np.array([[1,-6],[-6,4]])
        expected = normalized_Hamiltonian(inp)
        calculated = np.array([[ 0.10599979, -0.63599873],[-0.63599873,  0.42399915]])
        assert np.allclose(calculated, expected)
        return True
    except AssertionError:
        return False


# In[7]:


def is_upper_left(H = None, U = None):
    '''
    -> is_upper_left() 
    -> functionailty: Checks if the input Hamiltonian matrix is in the upper left of block encoded matrix U
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: The function takes two matrices as arguements. If they are not provided the function takes the assigned 
    values of 'normalized_hamiltonian' as its first argument and 'unitary_matrix' as its second and checks if shape of U is
    bigger than the shape of H and also if H is correctly in the upper-left position of U by slicing out the portion of same 
    shape of H from U and matching it with H. If all elements are close then it returns 'True', otherwise it returns 'False'
    '''

    if H is None:
        H = normalized_hamiltonian
    if U is None:
        U = unitary_matrix

    if H.shape[0] > U.shape[0] or H.shape[1] > U.shape[1]:
        raise AssertionError("The hamiltonian matrix should be smaller than block encoded matrix")
    upper_left_U = U[:H.shape[0],:H.shape[1]]
    if not np.allclose(H, upper_left_U):
        raise AssertionError("The Hamiltonian isn't block-encoded correctly in the upper left of U")

    return True, "The Hamiltonian matrix is in the upper left of U"

def test8_upper_left():     #tests and shows that it returns True when the hamiltonian matrix is the upper-left block of U
    '''
    -> test8_upper_left()  [True case]
    -> functionailty: tests if is_upper_left() works correctly
    -> Return: True (Test Passed)/ False
    -> Description: This shows if our function is_upper_left() works correctly by calling the function and running it on
    particular H and U. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        H = np.array([[1,0],[0,1]])
        U = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        expected, _ = is_upper_left(H,U)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as err:
        print(err)
        return False

def test9_not_upper_left():     #tests and shows that it returns False when the hamiltonian matrix is not the upper-left block of U
    '''
    -> test9_upper_left()  [False case]
    -> functionailty: tests if is_upper_left() works correctly
    -> Return: True (Test Passed)/ False
    -> Description: This shows if our function is_upper_left() works correctly by calling the function and running it on
    particular H and U. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        H = np.array([[1,0],[0,1]])
        U = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,1]])
        expected, _ = is_upper_left(H,U)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as err:
        print(err)
        return False


# In[39]:


# User can just copy and paste the Hash map in dictionary format


def get_hashmap():
    '''
    -> get_hashmap() 
    -> functionailty: Takes the hashmap from the user in {'gate': weight, ...} format as a string and stores that
    -> Return: A dictionary of hashmap with gate terms and the coefficients
    -> Description: The function takes a hashmap from user as a string in the mentioned format and converts it to a dictionary
    automatically. It stores the input value in hash_map variable. 
    '''
    hashmap_string = input("Enter the Hamiltonian in the format {'gate': weight, ...}: ")
    global hash_map
    # Evaluating the string to convert it to a Python dictionary
    hashmap = eval(hashmap_string)
    hash_map = hashmap
    return hashmap

def valid_hashmap(hashmap = None):
    
    '''
    -> valid_hashmap() 
    -> functionailty: Takes the hashmap and checks if the gate terms belong to valid_gates set
    -> Return: True/TypeError 
    -> Description: The function calls get_hashmap function and iterate over the elements to check if any gate term is invalid
    and raises error.
    '''
    valid_gates = {"I", "X", "Y", "Z"}

    if hashmap is None:
        hashmap = get_hashmap()
    for term in hashmap.keys():
        for gate in term:
            if gate not in valid_gates:
                raise TypeError("The provided Gate term doesn't belong to Pauli Gates")
    return True

def test10_validmap():     #tests and shows that it returns True when user provides valid hashmap
    '''
    -> test10_validmap() [True case]
    -> functionailty: checks if valid_hashmap works correctly
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function is_square_hermitian() works correctly by calling the function and running it on
    the string. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        hashmap = {'XY': 2}
        expected = valid_hashmap(hashmap)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as err:
        print(err)
        return False

def test11_not_validmap():     #tests and shows that it returns False when user doesn't provide valid hashmap
    '''
    -> test11_validmap() [False case]
    -> functionailty: checks if valid_hashmap works correctly
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function is_square_hermitian() works correctly by calling the function and running it on
    the string. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        hashmap = {'XR': 2}
        expected = valid_hashmap(hashmap)
        assert expected == True
        return True, "Test Passed"
    except AssertionError as err:
        print(err)
        return False


# In[9]:


#Check that the hash_map and hamiltonian matrix match

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

paulis = {"I": I, "X": X, "Y": Y, "Z": Z}
map_matrix = None
def hashmap_to_matrix(hashmap=None):
    '''
    -> hashmap_to_matrix() 
    -> functionailty: Takes the hashmap from the user and creates the corresponding matrix
    -> Return: A matrix corresponding to the hashmap
    -> Description: The function takes a hashmap from user 
    '''
    global map_matrix
    if hashmap is None:
        hashmap = hash_map
    matrix = np.zeros((2**len(list(hashmap.keys())[0]), 2**len(list(hashmap.keys())[0])), dtype=complex)
    for terms, weights in hashmap.items():
        term_matrix = np.array([[1.0]])
        for gate in terms:
            term_matrix = np.kron(term_matrix, paulis[gate])
        matrix += weights * term_matrix
    map_matrix = matrix
    return matrix

def hash_and_hamiltonian(hamiltonian_map=None, hamiltonian_matrix=None):
    '''
    -> hash_and_hamiltonian() 
    -> functionailty: checks if the hamiltonian matrix and hashmap match
    -> Return: True/ValueError
    -> Description: If the arguments are not provided then it considers map_matrix as first argument and hamiltonian matrix
    as the second argument and returns 'True' if the elements are almost close and raises error otherwise
    '''
    if hamiltonian_map is None:
        hamiltonian_map = hashmap_to_matrix()
    if hamiltonian_matrix is None:
        hamiltonian_matrix = hamiltonian
    if not np.allclose(hamiltonian_map, hamiltonian_matrix):
        raise ValueError("Hamiltonian matrix and hash_map don't match, please check!!")
    return True  # Return matrix1 as required

def test12_hashmap_to_matrix():     #tests and shows that it returns True when the hashmap correctly returns the matrix
    '''
    -> test12_hashmap_to_matrix()  [True case]
    -> functionailty: tests hashmap_to_matrix() function
    -> Return: True (Test Passed)/False (test Failed) 
    -> Description: This shows if our function hashmap_to_matrix() works correctly by calling the function and running it on
    the particular string. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        hashmap = {'XZ': 2}
        expected = hashmap_to_matrix(hashmap)
        calculated = [[0,0,2,0],[0,0,0,-2],[2,0,0,0],[0,-2,0,0]]
        assert np.array_equal(calculated, expected)
        return True, "Test passed"
    except AssertionError:
        return False, "Test failed"

def test13_not_hashmap_to_matrix():     #tests and shows that it returns False when the hashmap doesn't correctly return the matrix
    '''
    -> test13_hashmap_to_matrix()  [False case]
    -> functionailty: tests hashmap_to_matrix() function
    -> Return: True (Test Passed)/False (test Failed) 
    -> Description: This shows if our function hashmap_to_matrix() works correctly by calling the function and running it on
    the particular string. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        hashmap = {'XZ': 2}
        expected = hashmap_to_matrix(hashmap)
        calculated = [[0,0,4,0],[0,0,0,-2],[2,0,0,0],[0,-2,0,0]]
        assert np.array_equal(calculated, expected)
        return True, "Test passed"
    except AssertionError:
        return False, "Test failed"


# In[10]:


import scipy as sp
def block_encoded_U(Ham = None):
    '''
    -> block_encoded_U() 
    -> functionailty: Takes the hamiltonian matrix and block encode it
    -> Return: Block encoded matrix
    -> Description: It takes the normalized hamiltonian as argument if that's not provided separately by the user. It then 
    makes a all-zero matrix twice the size of H matrix and add H as upper-left block, \sqrt(1-H^2) as upper-right and 
    lower-left block and -H as lower right block and returns the block-encoded matrix
    '''
    global U_block
    if Ham is None:
        Ham = normalized_hamiltonian
    H2=np.matmul(Ham,Ham)
    n = H2.shape[0]
    
    U = np.zeros([2*n,2*n],dtype=complex)

    I=np.identity(n)
    P = sp.linalg.sqrtm(I-H2)

    U[:n,:n]=Ham
    U[:n,n:]=P
    U[n:,n:]=-Ham
    U[n:,:n]=P

    Uf=np.matmul(U,U.conjugate().transpose())
    U_block = U
    return U

def test14_block_encoded_U():     #tests and shows that it returns True when the matrix is normalized according to Operator norm
    '''
    -> test14_block_encoded_U()  [True case]
    -> functionailty: tests the block_encoded_U() function
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function block_encoded_U() works correctly by calling the function and running it on
    the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
    '''
    try:
        Ham = np.array([[1,0],[0,1]])
        expected = block_encoded_U(Ham)
        calculated = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])  
        assert np.allclose(calculated, expected)
        return True, "Test passed"
    except AssertionError:
        return False, "Test failed"
    
def test15_block_encoded_U():     #tests and shows that it returns True when the matrix is normalized according to Operator norm
    '''
    -> test15_block_encoded_U()  [False case]
    -> functionailty: Takes a particular matrix (array), checks if it's square, hermitian and the norm is close to 1
    -> Return: True (Test Passed)/ test Failed (with the message why it fails)
    -> Description: This shows if our function block_encoded_U() works correctly by calling the function and running it on
    the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError
    '''
    try:
        Ham = np.array([[1,0],[0,1]])
        expected = block_encoded_U(Ham)
        calculated = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])  
        assert np.allclose(calculated, expected)
        return True, "Test passed"
    except AssertionError:
        return False, "Test failed"


# In[40]:



def user():
    other_inputs()
    print("If you have the block-encoded matrix, be sure that the norm is an Operator norm")
    Input = input("Do you have the block-encoded matrix? (y/n) ")
    if Input == 'y':
        if not is_square_unitary():
            raise ValueError("Given block encoded matrix is not unitary")
        if not is_square_hermitian():
            raise ValueError("Given Hamiltonian matrix is not hermitian")
        if not is_upper_left():
            raise ValueError("The Hamiltonian is not correctly block encoded")
        if not valid_hashmap():
            raise ValueError("Given hash map is not valid")
        if not hash_and_hamiltonian():
            raise ValueError("Given hash map and the Hamiltonian matrix don't match")
    elif Input == 'n':
        Input_2 = input("Do you have the hamiltonian matrix? (y/n) ")
        if Input_2 == 'y':
            if not is_square_hermitian():
                raise ValueError("Given Hamiltonian matrix is not hermitian")
            if not valid_hashmap():
                raise ValueError("Given hash map is not valid")
            if not hash_and_hamiltonian():
                raise ValueError("Given hash map and the Hamiltonian matrix don't match")
            normalized_Hamiltonian(map_matrix)
            block_encoded_U()
        elif Input_2 == 'n':
            if not valid_hashmap():
                raise ValueError("Given hashmap is not valid")
            hashmap_to_matrix()
            normalized_Hamiltonian(map_matrix)
            block_encoded_U()
        else:
            raise TypeError("Invalid input")
    else:
        raise TypeError("Invalid input")
    return True




# In[41]:


user()

