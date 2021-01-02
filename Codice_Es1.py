from timeit import default_timer
from random import random
import numpy as np
import matplotlib.pyplot as plt




def generate_sorted_list(n):
    L = list(range(n))
    return L

def generate_random_list(n):
    L = [random() for i in range(n)]
    return L




def test_execution_time_sort(sort_functions):
    out = []
    
    print('random list')
    out.append(test(sort_functions, generate_random_list))
    
    print('sorted list')
    out.append(test(sort_functions, generate_sorted_list))
    
    return out

def test(sort_functions, generate_function):
    n = 1
    max_time = 0
    lengths = np.array([])
    times = np.empty((len(sort_functions), 0))
    while max_time < 16: # can be changed
        n *= 2
        lengths.resize(lengths.size+1)
        lengths[-1] = n
        times = np.hstack((times, np.zeros((times.shape[0], 1))))
        num_tests = 8 # can be changed
        for k in range(num_tests):
            L = generate_function(n)
            for i in range(len(sort_functions)):
                L_copy = list(L)
                times[i][-1] -= default_timer()
                sort_functions[i](L_copy)
                times[i][-1] += default_timer()
        for i in range(times.shape[0]):
            times[i][-1] /= num_tests
            if times[i][-1] > max_time:
                max_time = times[i][-1]
        print(f'    length = {n}')
    return lengths, times




def insertion_sort(A):
    """This function sorts a list in place using the insertion sort algorithm."""
    for j in range(1, len(A)):
        key = A[j]
        i = j-1
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i -= 1
        A[i+1] = key

def merge_sort(A):
    """This function sorts a list in place using the merge sort algorithm."""
    _merge_sort(A, 0, len(A)-1)

def _merge_sort(A, p, r):
    if p < r:
        q = (p+r) // 2
        _merge_sort(A, p, q)
        _merge_sort(A, q+1, r)
        _merge(A, p, q, r)

def _merge(A, p, q, r):
    n1 = q-p+1
    n2 = r-q
    L = []
    R = []
    for i in range(n1):
        L.append(A[p+i])
    for j in range(n2):
        R.append(A[q+j+1])
    i = 0
    j = 0
    for k in range(p, r+1):
        if i == n1 or j == n2:
            break
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
    while i < n1:
        A[k] = L[i]
        i += 1
        k += 1
    while j < n2:
        A[k] = R[j]
        j += 1
        k += 1

def quick_sort(A):
    """This function sort a list in place using the quick sort alorithm."""
    _quick_sort(A, 0, len(A)-1)

def _quick_sort(A, p, r):
    if p < r:
        q = _partition(A, p, r)
        _quick_sort(A, p, q-1)
        _quick_sort(A, q+1, r)

def _partition(A, p, r):
    x = A[r]
    i = p-1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return i+1




test = test_execution_time_sort((insertion_sort, merge_sort))




def plot(test, labels, funcs):
    for i in range(len(test)):
        plt.figure(i+1)
        x = np.logspace(1, np.log2(test[i][0][-1]), num=np.log2(test[i][0][-1]), base=2)
        for j in range(len(test[i][1])):
            plt.plot(test[i][0], test[i][1][j], label=labels[j])
            coeff = test[i][1][j][-1] / funcs[j][i](x[-1])
            y = funcs[j][i](x) * coeff
            plt.plot(x, y, 'k--')

def end_plot(n):
    for i in range(n):
        plt.figure(i+1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('array length')
        plt.ylabel('sorting time')
        plt.legend()

labels = ('insertion sort', 'merge sort')

is_funcs = (lambda x: x**2, lambda x: x)
ms_funcs = (lambda x: x*np.log(x), lambda x: x*np.log(x))
funcs = (is_funcs, ms_funcs)

plot(test, labels, funcs)

end_plot(len(test))
