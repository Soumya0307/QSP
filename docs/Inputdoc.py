#!/usr/bin/env python
# coding: utf-8

# We first need to import .....

# In[26]:


from Final import *
import Final
from Final import initial_state, execution_time, qubit_number, trotter_step, error_global, unitary_matrix, hamiltonian, normalized_hamiltonian, hash_map, U_block


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
# 
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

# ## 1. The user already has the block-encoded matrix:

# First let's look at the situation when the user knows the block encoded matrix beforehand. So, we take other inputs from the user - 1. Trotter step, 2. Initial statevector, 3. Execution time, 4. Number of qubits

# In[2]:


Final.other_inputs()


# Now, we print the inputs below to show that the inputs are correctly assigned to the global variables. Note that, we are printing the global variables to show this.

# In[3]:


print("Trotter step:", Final.trotter_step)
print("Initial statevector:", Final.initial_state)
print("Evolution time:", Final.execution_time)
print("Number of qubits:", Final.qubit_number)


# We call `is_square_unitary()` function to take the input of the block-encoded matrix by calling `get_matrix()` function inside it. It then checks if it's square by calling `is_sqaure()` and unitary by calling `is_unitary()` functions. All the other functions are called inside this function.
# 
# Now, if the user inputs an approxiated unitary matrix, then the function checks if all the diagonal elements are close to 1 and off-diagonal terms are close to 0 according to the tolerance $\epsilon$.
# 
# Let's say $M$ is the input matrix, then $T = M^{\dagger} M$ is close to identity if $|\text{diagonal terms} - 1| < \epsilon$ and $|\text{off-diagonal terms} - 0| < \epsilon$, then the input matrix $M$ is unitary under the tolerance.

# In[4]:


Final.is_square_unitary()


# The input matrix is unitary under the tolerance 0.15, whereas below if you put tolerance = 0 for the same matrix, that is clearly not a perfect unitary and gives an error

# In[29]:


Final.is_square_unitary()


# We print the global variable to show that it's assigned to `unitary_matrix` successfully, we can use it later in the code

# In[5]:


print(Final.unitary_matrix)
print("It is unitary under the error threshold ",Final.error_global)


# Now, we take the Hamiltonian matrix as an input from the user. `is_square_hermitian()` calls `get_hamiltonian_matrix()` to take input from the user. It checks if it's square by calling `is_square()` function and hermitian by calling `is_hermitian()` function. 
# 
# The function also normalize the Hamiltonian which we need for block-encoding by calling `normalized_Hamiltonian()` function. This is an Operator norm defined by $\|A\| = \sup_{x \neq 0} \frac{\|Ax\|}{\|x\|}$, where $A$ is the matrix

# In[6]:


Final.is_square_hermitian()


# Like before, we show that the Hamiltonian matrix is assigned to `hamiltonian` and normalized Hamiltonian matrix is assigned to `normalized_hamiltonian`

# In[7]:


print("The input Hamiltonian matrix is: ")
print(Final.hamiltonian)
print("The normalized Hamiltonian matrix is: ")
print(Final.normalized_hamiltonian)


# To make the function `is_upper_left()` function work correctly, we need to make sure two things - 
# 1. The block-encoded matrix must be related to Operator norm.
# 2. The elements of Hamiltonian matrix and the the upper-left block of block-encoded matrix (`unitary_matrix`) must be equal to two decimal places
# 
# Otherwise, this will raise error.

# In[8]:


Final.is_upper_left()


# The last thing to take input from the user is the hashmap. `valis_hashmap()` calls `get_hashmap()` to take the input and checks if the gate terms belong to Pauli-gate set. 

# In[9]:


Final.valid_hashmap()


# The input is succesfully assigned to the variable `hash_map`

# In[13]:


print("Input hashmap is:", Final.hash_map)


# Depending on the terms and coefficients from the `hash_map`, it takes corresponding pauli-gate matrix defined below
# 
# $$
# I =
# \begin{equation}
# \begin{pmatrix}
# 1 & 0\\
# 0 & 1
# \end{pmatrix}
# \end{equation}
# $$
# $$ X =
# \begin{equation}
# \begin{pmatrix}
# 0 & 1\\
# 1 & 0
# \end{pmatrix}
# \end{equation}
# $$
# $$
# Y =
# \begin{equation}
# \begin{pmatrix}
# 0 & -i\\
# i & 0
# \end{pmatrix}
# \end{equation}
# $$
# $$
# Z =
# \begin{equation}
# \begin{pmatrix}
# 1 & 0\\
# 0 & -1
# \end{pmatrix}
# \end{equation}
# $$
# 
# then it performs **Kronecker rpodect** if there are more than one pauli gate in one term. Below, there's the matrix 
# 
# $$
# H =
# \begin{equation}
# \begin{pmatrix}
# 2 & 1\\
# 1 & -2
# \end{pmatrix}
# \end{equation}
# $$
# 
# for the Hamiltonian expression $H = X + Z$

# In[11]:


Final.hashmap_to_matrix()


# The matrix is assigned to the variable `map_matrix`

# In[14]:


print("The matrix generated from the hashmap is stored in map_matrix")
print(Final.map_matrix)


# For the final check we test if the Hamiltonian matrix (unnormalized) in `hamiltonian` and the matrix generated from the hashmap i.e. `map_matrix` are equal.

# In[10]:


Final.hash_and_hamiltonian()


# ## 2. The user has the hamiltonian matrix and not the block-encoded matrix:

# This section is a subpart of the above. If the user does not have the block-encoded matrix, then he/she starts from inputting the Hamiltonian matrix - the rest is same. The one thing, in this case, we cannot perform is performing `is_upper_left()` as, obviously, we don't have the block-encoded matrix.
# 
# Though we still need the block-encoded matrix for QSP simulation in the later part. So, we block encode the normalized Hamiltonian following the above-mentioned definition of block-encoding. 

# In[28]:


Final.block_encoded_U()


# The block-encoded matrix is then assigned to the variable `U_block`

# In[31]:


Final.U_block


# **Note:** There's a subtle difference between the global variables `unitary_matrix` and `U_block`. 
# 
# 1. `unitary_matrix` stores the block-encoded matrix only if the user has the block-encoded matrix beforehand and inputs it.
# 2. `U_block` stores the block-encoded matrix when we do it for them from the input Hamiltonian
# 
# So, for example, if the user doesn't have the block-encoded matrix beforehand (expected: `unitary_matrix = None`) the `unitary_matrix` will be assigned the same value as `U_block` and vice-versa

# ## 3. The user only has the hashmap:

# This part is now trivial. The user just inputs the Hash map and the code performs the matrix conversion of the hash map, normalizes it and performs the block-encoding and stores it in `U_block`, which as mentioned above is automatically be assigned to the global variable `unitary_matrix`

# In[ ]:




