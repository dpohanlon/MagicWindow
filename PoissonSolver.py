import numpy as np

from tqdm import tqdm

'''
Jacobi method, modified from 'Numerical methods for partial differential equations', by Bernard Knaepen & Yelyzaveta Velizhanina
(https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html)
'''

def l2_diff(a, b):

    return np.sqrt(np.sum((a - b) ** 2)) / len(a)

def solve(image, n, z, max_it = 100000, tol = 1E-10, d = 1):

    nx, ny = image.shape[0] + 2, image.shape[1] + 2

    nI = image / np.sum(image)

    bI = (1. - nI) / ((n - 1) * z)
    bI -= np.mean(bI)

    b = np.zeros((nx, ny))
    b[1:-1, 1:-1] = bI

    dx, dy = d, d

    p0 = np.zeros((nx, ny))
    pnew = p0.copy()

    diff = 1.0

    p = np.empty((nx, ny))

    for i in tqdm(range(max_it)):

        np.copyto(p, pnew)

        pnew[1:-1, 1:-1] = (0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - b[1:-1, 1:-1] * dx ** 2))

        diff = l2_diff(pnew, p)

        if diff < tol:

            print('Final difference:', diff)

            break

    print('Final difference:', diff)

    return pnew
