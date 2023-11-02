#!/usr/bin/env python3

import numpy as np
from zernike import RZern
import matplotlib.pyplot as plt
from scipy.linalg import eig

class ZernikeEllipticalaperture:
    def __init__(self, rmax, npix, a, b, l, ell_aperture=True, coeff=None):
        self.rmax = rmax
        self.npix = npix
        self.a = a
        self.b = b
        self.l = l
        self.ell_aperture = ell_aperture  # Specify whether to use the elliptical aperture
        self.coeff = coeff  # Coefficients for the Zernike modes (optional)
        self.ell_aperture_mask = self.GenerateEllipticalAperture()
        self.circular_zern = self.GetCircZernikeValue()
        self.E = self.CalculateEllipticalZernike()

    def GetCircZernikeValue(self):
        zernike = RZern(self.rmax)
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.npix), np.linspace(-1, 1, self.npix))
        rho = np.sqrt(xx**2 + yy**2)
        theta = np.arctan2(yy, xx)
        zernike.make_cart_grid(xx, yy)

        zern_value = []
        nterms = int((self.rmax + 1) * (self.rmax + 2) / 2)
        for i in range(nterms):
            zern_value.append(zernike.Zk(i, rho, theta))

        zern_value = np.array(zern_value / np.linalg.norm(zern_value)).squeeze()
        return zern_value

    def CalculateEllipticalZernike(self):
        Z = self.GetCircZernikeValue()
        M = self.M_matrix()
        E = []  # Initialize a list to store E arrays for each l

        for i in range(1, self.l + 1):
            E_l = np.zeros(Z[0].shape)  # Initialize E with the same shape as Z[0]
            for j in range(1, i + 1):
                E_l += M[i - 1, j - 1] * Z[j - 1]
            E.append(E_l)

        E = np.array(E)
        if self.ell_aperture:
            E[:, np.logical_not(self.ell_aperture_mask)] = 0
        return E

    def M_matrix(self):
        C = self.C_zern()
        regularization = 1e-6  # Small positive constant to regularize
        C += regularization * np.eye(C.shape[0])

        Q = np.linalg.cholesky(C)
        QT = np.transpose(Q)
        M = np.linalg.inv(QT)
        return M

    def C_zern(self):
        nterms = int((self.rmax + 1) * (self.rmax + 2) / 2)
        # Initialize the C matrix
        C = np.zeros((nterms, nterms))
        # Calculate the area of each grid cell
        dx = (2 * self.a) / 10000
        dy = (2 * self.b) / 10000

        for i in range(nterms):
            for j in range(i, nterms):
                product_Zern = np.dot(self.circular_zern[i], self.circular_zern[j]) * dx * dy
                C[i, j] += np.sum(product_Zern)
                if i != j:
                    C[j, i] = C[i, j]

        return C

    def GenerateEllipticalAperture(self):
        x, y = np.meshgrid(np.linspace(-1, 1, self.npix), np.linspace(-1, 1, self.npix))
        normalized_distance = (x / self.a) ** 2 + (y / self.b) ** 2
        aperture = (normalized_distance <= 1).astype(float)
        return aperture

    def EllZernikeMap(self, coeff=None):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.npix), np.linspace(-1, 1, self.npix))
        E_ell = np.zeros((xx.size, self.l))

        for k in range(self.l):
            E_ell[:, k] = np.ravel(self.E[k])

        if coeff is None:
            coeff = np.random.random(self.l)

        phi = np.dot(E_ell, coeff)
        phi = phi.reshape(xx.shape)

        return phi

'''
if __name__ == '__main__':
    a = 1
    b = 0.8

    rmax = 7
    npix = 256
    l = 35

    ell_zern = ZernikeEllipticalaperture(rmax, npix, a, b, l)

    Ell = ell_zern.CalculateEllipticalZernike()
    plt.imshow(Ell[11])
    plt.show()

    phi = ell_zern.EllZernikeMap()
    plt.imshow(phi)
    plt.show()
'''
