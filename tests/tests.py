#!/usr/bin/env python
# coding: utf-8

# In[17]:


from Final import *
import unittest


# After combining our individual code segments, we noticed some redundant checks. However, we opted to retain them, understanding that while they lengthen the code slightly, they don't impact its functionality.

# In[19]:


import unittest

class TestUnitarity(unittest.TestCase):

    def test1_unitarity(self):       #tests and shows that it returns True when the matrix is square and unitary under allowed error
        '''
        -> test1_unitarity()  [True case: Unitary, square]
        -> functionailty: Takes a particular matrix (array) and error, checks if it's square and unitary given the allowed error
        -> Return: True (Test Passed)/ test Failed
        -> Description: This shows if our function is_square_unitary() works correctly by calling the function and running it on
        the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''

        inp = np.array([[0.99,0.01],[0.02,0.99]])
        err = 0.1
        self.assertTrue(is_square_unitary(inp, err))

    def test2_not_unitarity(self):    #tests and shows that it returns False when the matrix is square but not unitary under allowed error
        '''
        -> test2_not_unitarity()  [False case: Unitary, True case: square]
        -> functionailty: Takes a particular matrix (array) and error, checks if it's square and unitary given the allowed error
        -> Return: True (Test Passed)/ test Failed
        -> Description: This shows if our function is_square_unitary() works correctly by calling the function and running it on
        the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''

        inp = np.array([[0.99,0.01],[0.02,0.99]])
        err = 0.01
        self.assertFalse(is_square_unitary(inp, err))

    def test3_not_square(self):      #tests and shows that it returns False when the matrix is not square and hence doesn't go on checking unitarity
        '''
        -> test3_not_square()  [False case: square]
        -> functionailty: Takes a particular matrix (array) and error, checks if it's square and unitary given the allowed error
        -> Return: True (Test Passed)/ test Failed
        -> Description: This shows if our function is_square_unitary() works correctly by calling the function and running it on
        the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
        inp = np.array([[0.99,0.01]])
        err = 0.1
        self.assertFalse(is_square_unitary(inp, err))
    def test4_hermitian(self):     #tests and shows that it returns True when the matrix is hermitian
        '''
        -> test4_hermitian() [True case]
        -> functionailty: Takes a particular matrix (array), checks if it's hermitian
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function is_square_hermitian() works correctly by calling is_hermitian() function 
        and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
        inp = np.array([[1,4],[4,1]])
        self.assertTrue(is_hermitian(inp))
    def test5_not_hermitian(self):     #tests and shows that it returns False when the matrix is not hermitian
        '''
        -> test5_not_hermitian()  [False case]
        -> functionailty: Takes a particular matrix (array), checks if it's hermitian
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function is_square_hermitian() works correctly by calling is_hermitian() function 
        and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
    
        inp = np.array([[1,0],[-4,1]])
        self.assertTrue(is_hermitian(inp))
            
    def test6_normalised(self):       #tests and shows that it returns True when the matrix is normalized according to Operator norm
        '''
        -> test6_normalised()  [True case]
        -> functionailty: Takes a particular matrix (array), checks if it's normalized correctly according to Operator norm
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function is_square_hermitian() works correctly by calling the normalized_Hamiltonian()
        function and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise 
        AssertionError.
        '''
        inp = np.array([[1,-6],[-6,4]])
        expected = normalized_Hamiltonian(inp)
        calculated = np.array([[ 0.11514558, -0.69087346],[-0.69087346,  0.4605823 ]])
        self.assertTrue(np.allclose(calculated, expected))
    def test7_not_normalised():     #tests and shows that it returns False when the matrix is not normalized
        '''
        -> test7_normalised()  [False case]
        -> functionailty: Takes a particular matrix (array), checks if it's normalized correctly according to Operator norm
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function is_square_hermitian() works correctly by calling the normalized_Hamiltonian()
        function and running it on the particular matrix. If our function runs correctly then it returns 'True', else raise 
        AssertionError.
        '''
        inp = np.array([[1,-6],[-6,4]])
        expected = normalized_Hamiltonian(inp)
        calculated = np.array([[ 0.10599979, -0.63599873],[-0.63599873,  0.42399915]])
        self.assertTrue(np.allclose(calculated, expected))
    def test8_upper_left(self):     
        '''
        -> test8_upper_left()  [True case]
        -> functionailty: tests if is_upper_left() works correctly
        -> Return: True (Test Passed)/ False
        -> Description: This shows if our function is_upper_left() works correctly by calling the function and running it on
        particular H and U. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
        H = np.array([[1,0],[0,1]])
        U = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.assertTrue(is_upper_left(H, U))
    def test9_not_upper_left():     #tests and shows that it returns False when the hamiltonian matrix is not the upper-left block of U
        '''
        -> test9_upper_left()  [False case]
        -> functionailty: tests if is_upper_left() works correctly
        -> Return: True (Test Passed)/ False
        -> Description: This shows if our function is_upper_left() works correctly by calling the function and running it on
        particular H and U. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
        H = np.array([[1,0],[0,1]])
        U = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,1]])
        expected, _ = is_upper_left(H,U)
        self.assertTrue(is_upper_left(H, U))
    def test10_validmap(self):     #tests and shows that it returns True when user provides valid hashmap
        '''
        -> test10_validmap() [True case]
        -> functionailty: checks if valid_hashmap works correctly
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function is_square_hermitian() works correctly by calling the function and running it on
        the string. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
    
        hashmap = {'XY': 2}
        expected = valid_hashmap(hashmap)
        self.assertTrue(expected)
    def test11_not_validmap():     #tests and shows that it returns False when user doesn't provide valid hashmap
        '''
        -> test11_validmap() [False case]
        -> functionailty: checks if valid_hashmap works correctly
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function is_square_hermitian() works correctly by calling the function and running it on
        the string. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''

        hashmap = {'XR': 2}
        expected = valid_hashmap(hashmap)
        self.assertTrue(expected)
    def test12_hashmap_to_matrix(self):     #tests and shows that it returns True when the hashmap correctly returns the matrix
        '''
        -> test12_hashmap_to_matrix()  [True case]
        -> functionailty: tests hashmap_to_matrix() function
        -> Return: True (Test Passed)/False (test Failed) 
        -> Description: This shows if our function hashmap_to_matrix() works correctly by calling the function and running it on
        the particular string. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
        hashmap = {'XZ': 2}
        expected = hashmap_to_matrix(hashmap)
        calculated = [[0,0,2,0],[0,0,0,-2],[2,0,0,0],[0,-2,0,0]]
        self.assertTrue(np.array_equal(expected,calculated))
    def test13_not_hashmap_to_matrix():     #tests and shows that it returns False when the hashmap doesn't correctly return the matrix
        '''
        -> test13_hashmap_to_matrix()  [False case]
        -> functionailty: tests hashmap_to_matrix() function
        -> Return: True (Test Passed)/False (test Failed) 
        -> Description: This shows if our function hashmap_to_matrix() works correctly by calling the function and running it on
        the particular string. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
        hashmap = {'XZ': 2}
        expected = hashmap_to_matrix(hashmap)
        calculated = [[0,0,4,0],[0,0,0,-2],[2,0,0,0],[0,-2,0,0]]
        self.assertTrue(np.array_equal(expected,calculated))
    def test14_block_encoded_U(self):     #tests and shows that it returns True when the matrix is normalized according to Operator norm
        '''
        -> test14_block_encoded_U()  [True case]
        -> functionailty: tests the block_encoded_U() function
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function block_encoded_U() works correctly by calling the function and running it on
        the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError.
        '''
        Ham = np.array([[1,0],[0,1]])
        expected = block_encoded_U(Ham)
        calculated = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])  
        self.assertTrue(np.allclose(expected,calculated))
    def test15_not_block_encoded_U():     #tests and shows that it returns True when the matrix is normalized according to Operator norm
        '''
        -> test15_block_encoded_U()  [False case]
        -> functionailty: Takes a particular matrix (array), checks if it's square, hermitian and the norm is close to 1
        -> Return: True (Test Passed)/ test Failed (with the message why it fails)
        -> Description: This shows if our function block_encoded_U() works correctly by calling the function and running it on
        the particular matrix. If our function runs correctly then it returns 'True', else raise AssertionError
        '''
        Ham = np.array([[1,0],[0,1]])
        expected = block_encoded_U(Ham)
        calculated = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])  
        self.assertTrue(np.allclose(expected,calculated))


# I've conducted the tests I could think of. I'm uncertain if there are additional tests to consider. I'd appreciate any insights or suggestions on this topic.

# In[25]:
suite = unittest.TestLoader().loadTestsFromTestCase(TestUnitarity)
unittest.TextTestRunner(verbosity=2).run(suite)



# In[ ]:




