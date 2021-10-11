

import math
import numpy as np
import numpy.linalg as nl



def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r<0 or h<0:
        return np.NAN
    else:
        return float(2 * math.pi * r**2 + 2 * math.pi * r * h)

def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    ndarray = np.array([1,1])
    if type(n) == int:

        if n <= 0:
            return None
        elif n == 1:
            return np.array([1])
        elif n == 2:
            return ndarray
        else:
            while(len(ndarray) < n):
                i = len(ndarray)
                ndarray = np.append(ndarray, ndarray[i - 1] + ndarray[i - 2])

            return np.array([ndarray])
    else:
        return None




def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """

    matrix = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    Mdet = nl.det(matrix)
    Mt = np.transpose(matrix)

    if Mdet == 0:
        Minv = np.NaN
    else:
        Minv = nl.inv(matrix)



    return (Minv, Mt, Mdet)

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """

    if n < 0 or m < 0 or type(n) != int or type(m) != int:
        return None
    matrix = np.zeros(shape= (m,n))
    for i in range(0,m):
        for j in range(0,n):
            if i>j:
                matrix[i][j] = i
            else:
                matrix[i][j] = j
    return matrix