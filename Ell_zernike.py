#!/usr/bin/env python3

import numpy as np
from zernike import RZern
import matplotlib.pyplot as plt
from scipy.linalg import eig


def Get_Zernike_value(rmax, npix):

    #generate some circular Zernike
    zernike = RZern(rmax)
    xx, yy = np.meshgrid(np.linspace(-1,1,npix), np.linspace(-1,1,npix))
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    zernike.make_cart_grid(xx, yy)

    zern_value = []
    nterms = int((rmax + 1)*(rmax + 2)/2) #number of Zernike terms for a certain radial order
    for i in range(nterms):
        zern_value.append(zernike.Zk(i, rho, theta))

    zern_value = np.array(zern_value/ np.linalg.norm(zern_value)).squeeze()
    # coeff = np.random.random(nterms)
    # zer = zernike.eval_grid(coeff, matrix = True)

    return zern_value


def C_zern(rmax, npix, a, b):
    from scipy import integrate
    zernike_values = Get_Zernike_value(rmax, npix)
    nterms = int((rmax + 1) * (rmax + 2) / 2)

    # Generate the elliptical pupil
    #pupil = generate_elliptical_pupil(npix, a, b)
    #y = np.linspace(-1, 1, npix)
    #C, _ = integrate.nquad(zernike_values[i]*zernike_values[j], [[-a*np.sqrt(1-y**2/b**2), -a*np.sqrt(1-y**2/b**2)], [-b, b]])
    # Initialize the C matrix
    C = np.zeros((nterms, nterms))
    # Calculate the area of each grid cell
    dx = (2 * a) / 10000
    dy = (2 * b) / 10000

    for i in range(nterms):
        for j in range(i, nterms):
            #C, _ = integrate.nquad(zernike_values[i]*zernike_values[j], [[-a*np.sqrt(1-y**2/b**2), -a*np.sqrt(1-y**2/b**2)], [-b, b]])
            product_Zern = np.dot(zernike_values[i],zernike_values[j]) * dx * dy
            #print(product_Zern)
            # Add to the integral result
            # Add the product to the appropriate submatrix in C
            C[i, j] += np.sum(product_Zern)

            # If i and j are not the same, add the product to the corresponding submatrix in the upper triangle of C
            if i != j:
                C[j, i] = C[i, j]
    '''
    eigenvalues, eigenvectors = eig(C)

    # Check for negative or zero eigenvalues
    non_positive_eigenvalues = eigenvalues[eigenvalues <= 0]

    # Analyze the eigenvalues
    if len(non_positive_eigenvalues) == 0:
        print("Matrix C is positive definite.")
    else:
        print("Matrix C is not positive definite.")
        print("Non-positive eigenvalues:", non_positive_eigenvalues)

    plt.imshow(C)
    plt.show()
    '''
    return C

def M_matrix(rmax, npix, a, b):

    C = C_zern(rmax, npix, a, b)
    # Add a small positive constant to the diagonal to regularize
    regularization = 10e-6
    C += regularization * np.eye(C.shape[0])


    Q = np.linalg.cholesky(C)
    QT = np.transpose(Q)
    M = np.linalg.inv(QT)

    return M


def generate_elliptical_pupil(npix, a, b):
    # Create an empty grid
    x, y = np.meshgrid(np.linspace(-1, 1, npix), np.linspace(-1, 1, npix))

    # Calculate the normalized distance from the center
    normalized_distance = (x / a) ** 2 + (y / b) ** 2

    # Create a binary pupil mask (1 inside the ellipse, 0 outside)
    pupil = (normalized_distance <= 1).astype(float)

    return pupil

def calculate_elliptical_zernike(rmax, npix, a, b, l, pupil, ell_pupil = True):
    Z = Get_Zernike_value(rmax, npix)
    M = M_matrix(rmax, npix, a, b)
    E = []  # Initialize a list to store E arrays for each l

    for i in range(1, l + 1):
        E_l = np.zeros(Z[0].shape)  # Initialize E with the same shape as Z[0]
        for j in range(1, i + 1):
            E_l += M[i - 1, j - 1] * Z[j - 1] #* pupil
        E.append(E_l)
        
    E = np.array(E)

    xx, yy = np.meshgrid(np.linspace(-1,1,npix), np.linspace(-1,1,npix))
    E_ell = np.zeros((xx.size, l))

    if ell_pupil:
        E[:, np.logical_not(pupil)] = 0
    for k in range(l):
        coeff = np.random.random(l)
        E_ell[:, k] = np.ravel(E[k])
    phi = np.dot(E_ell, coeff)
    phi = phi.reshape(xx.shape)
    return E, phi



if __name__ == "__main__":
    a = 1
    b = 0.8

    rmax = 7
    npix = 256
    l = 35


    ell_pupil = generate_elliptical_pupil(npix, a, b)
    circular_zern = Get_Zernike_value(rmax, npix)
    # plt.imshow(zer)
    # plt.show()
    #print(circular_zern)

    E, phi = calculate_elliptical_zernike(rmax, npix, a, b, l, ell_pupil)
    # print(E.shape)
    plt.imshow(E[1])
    plt.show()
    plt.imshow(phi)
    plt.show()



