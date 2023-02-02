#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# In[2]:


@njit
def laplace_vacuum(r, dr, dz):
    '''
    r: 1d numpy array - radial position of grid points
    dr: 1d numpy array - radial distance of grid points (len(dr) = len(r)-1)
    dz: 1d numpy array - longitudinal distance of grid points (len(dz) = len(z)-1)
    
    Laplace operator in cylindrical coordinates with variable grid-spacing other the 2D r-z plane 
    (r >= 0; -L/2 <= z <= L/2) in Matrix form.
    
    A*phi = laplace(phi) = e*n/eps0
    
    The 2D potential phi is concatenated into a 1D format: phi = (phi_z(r=0), phi_z(r=r1),...)
    At r = 0 a Neumann boundary condition is implemented which sets the derivative of
    the potential to zero in agreement with the symmetry. At the wall a Diriclet boundary is implemented 
    according to the grounded wall.
    
    The Matrix A is in a csc-sparse format generated from col, row and data with the scipy.sparse package:
    A = csc_matrix((data, (col, row)), shape=(res_r*res_z, res_r*res_z))
    '''
    
    res_r = len(r) #Number of radial grid points
    res_z = len(dz)+1 #Number of longitudinal grid points
    col = np.empty(5*res_r*res_z-2*res_r-2*res_z) #Initialize array for column index
    col[:3] = 0 #The first grid point is in the lower left corner and therefor has only two neighboors + himself.
                #That is a bit sad indeed, but I can't help it.
    for i in range(1, res_z-1):        
        col[i*4-1:i*4+3] = i #The grid points in the lowest longitudinal column have at least three neighboors 
    col[4*res_z-5:4*res_z-2] = res_z-1 # + themself, that 
    for n in range(1, res_r-1): #Now I can loop through the grid points that are not at the edge
        j = n*res_z
        l = j+res_z-1
        m = (n-1)*(8+5*res_z-10)+4*res_z-2 #This is the number of indices I gathered so far
        col[m:m+4] = j #Okay, I lied, this one is at the left edge (z = -L/2)
        ii = np.arange(j+1, l, 1) #column indices for one of the longitudinal rows
        for i in range(res_z-2):
            mm = m+4+5*i   #Four column indices for the points with four neighboors + themself
            col[mm:mm+5] = ii[i] #Those are the ones in between(-L/2 < z < L/2)
        col[mm+5:mm+9] = l #And this one is at the right edge (z = L/2)
    col[5*res_r*res_z-2*res_r-6*res_z+2:5*res_r*res_z-2*res_r-6*res_z+5] = (res_r-1)*res_z #This one is in the upper left corner
    ii = np.arange(res_z*(res_r-1)+1,res_z*res_r-1, 1)
    m = 5*res_r*res_z-2*res_r-6*res_z+5
    for i in range(res_z-2):
        col[m+i*4:m+i*4+4] = ii[i] #These grid points adjoin the upper edge  
    col[5*res_r*res_z-2*res_r-2*res_z-3:5*res_r*res_z-2*res_r-2*res_z] = res_r*res_z-1 #Te finel one in the upper right corner

    row = np.empty(5*res_r*res_z-2*res_r-2*res_z) #Initialize array for row index 
    row[0] = 0 #The way the sparse matrix is setup, the column indices are in the outer loop
    row[1] = 1 #and the row indeces go through and the row indices in the inner one:
    row[2] = res_z # A = ((i_col1, j_row1, data1), (i_col1, j_row2, data2), (i_col1, j_row3, , data3), (i_col2, j_row1, data4),... )
    for i in range(1, res_z-1):        
        row[i*4-1] = i-1   
        row[i*4] = i
        row[i*4+1] = i+1
        row[i*4+2] = i+res_z
    row[4*res_z-5] = res_z-2
    row[4*res_z-4] = res_z-1
    row[4*res_z-3] = 2*res_z-1
    for n in range(1, res_r-1):
        j = n*res_z
        l = j+res_z-1
        m = (n-1)*(8+5*res_z-10)+4*res_z-2
        row[m] = j-res_z
        row[m+1] = j
        row[m+2] = j+1
        row[m+3] = j+res_z
        ii = np.arange(j+1, l, 1)
        for i in range(res_z-2):
            mm = m+4+5*i  
            row[mm] = ii[i]-res_z
            row[mm+1] = ii[i]-1
            row[mm+2] = ii[i]
            row[mm+3] = ii[i]+1
            row[mm+4] = ii[i]+res_z
        row[mm+5] = l-res_z
        row[mm+6] = l-1
        row[mm+7] = l
        row[mm+8] = l+res_z
    row[mm+9] = (res_r-2)*res_z
    row[mm+10] = (res_r-1)*res_z
    row[mm+11] = (res_r-1)*res_z+1
    ii = np.arange(res_z*(res_r-1)+1,res_z*res_r-1, 1)
    m = 5*res_r*res_z-2*res_r-6*res_z+5
    for i in range(res_z-2):
        row[m+i*4] = ii[i]-res_z
        row[m+i*4+1] = ii[i]-1
        row[m+i*4+2] = ii[i]
        row[m+i*4+3] = ii[i]+1
    row[5*res_r*res_z-2*res_r-2*res_z-3] = (res_r-1)*res_z-1
    row[5*res_r*res_z-2*res_r-2*res_z-2] = res_r*res_z-2
    row[5*res_r*res_z-2*res_r-2*res_z-1] = res_r*res_z-1

    dz_l = np.empty(res_z-2) #Initialize array for matrix elements according to the second derivative in z
    dz_m = np.empty(res_z-2) #The second spatial derivative of phi is approximated by
    dz_r = np.empty(res_z-2) #d^2(phi)/dz^2 = dz_l*phi(z-dz) + dz_m*phi(z) + dz_r*phi(z+dz)
    for i in range(1, res_z-1): 
        dz_l[i-1] = 2/((dz[i-1] + dz[i])*dz[i-1]) #These coefficients take the variable grid spacing into account
        dz_m[i-1] = -(2/dz[i-1] + 2/dz[i])/(dz[i-1] + dz[i]) 
        dz_r[i-1] = 2/((dz[i-1] + dz[i])*dz[i]) 

    data = np.empty(5*res_r*res_z-2*res_r-2*res_z)
    data[0] = -2/dz[0]**2-2/dr[0]**2 
    data[1] = 2/dz[0]**2 #For z = +/-L/2 the coefficient to the left/right is zero
    data[2] = 2/dr[0]**2 #representing the grounded wall.
    for j in range(res_z-2):
        data[4*j+3] = dz_l[j]
        data[4*j+4] = dz_m[j]-2/dr[0]**2
        data[4*j+5] = dz_r[j]
        data[4*j+6] = 2/dr[0]**2 #Along the left column (r=0) the first derivative in r is zero phi(r-dr) = phi(r+dr)
    data[4*res_z-5] = 1/dz[-1]**2
    data[4*res_z-4] = -2/dz[-1]**2-2/dr[0]**2
    data[4*res_z-3] = 2/dr[0]**2

    for i in range(1, res_r-1):                      #Analogously the second derivative in r is approximated by
        dr_l = (-1/r[i]+2/dr[i-1])/(dr[i-1] + dr[i]) #d^2(phi)/dr^2 = dr_l*phi(r-dr) + dr_m*phi(r) + dr_r*phi(r+dr)
        dr_m = -(2/dr[i-1]+2/dr[i])/(dr[i-1] + dr[i])
        dr_r = (1/r[i]+2/dr[i])/(dr[i-1] + dr[i])
        m = (i-1)*(8+5*res_z-10)+4*res_z-2
        data[m] = dr_l
        data[m+1] = -2/dz[0]**2+dr_m
        data[m+2] = 2/dz[0]**2
        data[m+3] = dr_r
        for j in range(res_z-2):
            mm = m+4+5*j  
            data[mm] = dr_l
            data[mm+1] = dz_l[j]
            data[mm+2] = dz_m[j] + dr_m
            data[mm+3] = dz_r[j]
            data[mm+4] = dr_r
        data[mm+5] = dr_l
        data[mm+6] = dz_l[-1]
        data[mm+7] = dz_m[-1]+dr_m
        data[mm+8] = dr_r
    dr_l = (-1/r[res_r-1]+2/dr[res_r-3])/(dr[res_r-3] + dr[res_r-2])
    dr_m = -(2/dr[res_r-3]+2/dr[res_r-2])/(dr[res_r-3] + dr[res_r-2])
    dr_r = (1/r[res_r-1]+2/dr[res_r-2])/(dr[res_r-3] + dr[res_r-2])
    data[mm+9] = dr_l
    data[mm+10] = -2/dz[0]**2+dr_m
    data[mm+11] = 2/dz[0]**2 
    m = 5*res_r*res_z-2*res_r-6*res_z+5
    for j in range(res_z-2):
        data[m+j*4] = dr_l
        data[m+j*4+1] = dz_l[j]
        data[m+j*4+2] = dz_m[j] + dr_m
        data[m+j*4+3] = dz_r[j]
    data[5*res_r*res_z-2*res_r-2*res_z-3] = dr_l
    data[5*res_r*res_z-2*res_r-2*res_z-2] = 1/dz[-1]**2
    data[5*res_r*res_z-2*res_r-2*res_z-1] = -2/dz[-1]**2+dr_m   
    
    return data, col, row


@njit
def laplace_electrodes(r, dr, dz, mask):
    '''
    r: 1d numpy array - radial position of grid points
    dr: 1d numpy array - radial distance of grid points (len(dr) = len(r)-1)
    dz: 1d numpy array - longitudinal distance of grid points (len(dz) = len(z)-1)
    mask: 2d numpy array - mask with dimension len(r)Xlen(z) which is 1 in the presents 
                           of an electrode and 0 everywhere else.  
    
    Laplace operator in cylindrical coordinates with variable grid-spacing other the 2D r-z plane 
    (r >= 0; 0 <= z <= L/2) in Matrix form.
    
    A*phi = laplace(phi) = e*n/eps0
    
    The 2D potential phi is concatenated into a 1D format: phi = (phi_z(r=0), phi_z(r=r1),...)
    At r = 0  and z = 0 a Neumann boundary condition is implemented which sets the derivative of
    the potential to zero in agreement with the symmetry. At the wall a Diriclet boundary is implemented 
    according to the grounded wall.
    
    The Matrix A is in a csc-sparse format generated from col, row and data with the scipy.sparse package:
    A = csc_matrix((data, (col, row)), shape=(res_r*res_z, res_r*res_z))
    '''
    
    res_r = len(r) #Number of radial grid points
    res_z = len(dz)+1 #Number of longitudinal grid points
    col = np.empty(5*res_r*res_z-2*res_r-2*res_z) #Initialize array for column index
    col[:3] = 0 #The first grid point is in the lower left corner and therefor has only two neighboors + himself.
                #That is a bit sad indeed, but I can't help it.
    for i in range(1, res_z-1):        
        col[i*4-1:i*4+3] = i #The grid points in the lowest longitudinal column have at least three neighboors 
    col[4*res_z-5:4*res_z-2] = res_z-1 # + themself, that 
    for n in range(1, res_r-1): #Now I can loop through the grid points that are not at the edge
        j = n*res_z
        l = j+res_z-1
        m = (n-1)*(8+5*res_z-10)+4*res_z-2 #This is the number of indices I gathered so far
        col[m:m+4] = j #Okay, I lied, this one is at the left edge (z = -L/2)
        ii = np.arange(j+1, l, 1) #column indices for one of the longitudinal rows
        for i in range(res_z-2):
            mm = m+4+5*i   #Four column indices for the points with four neighboors + themself
            col[mm:mm+5] = ii[i] #Those are the ones in between(-L/2 < z < L/2)
        col[mm+5:mm+9] = l #And this one is at the right edge (z = L/2)
    col[5*res_r*res_z-2*res_r-6*res_z+2:5*res_r*res_z-2*res_r-6*res_z+5] = (res_r-1)*res_z #This one is in the upper left corner
    ii = np.arange(res_z*(res_r-1)+1,res_z*res_r-1, 1)
    m = 5*res_r*res_z-2*res_r-6*res_z+5
    for i in range(res_z-2):
        col[m+i*4:m+i*4+4] = ii[i] #These grid points adjoin the upper edge  
    col[5*res_r*res_z-2*res_r-2*res_z-3:5*res_r*res_z-2*res_r-2*res_z] = res_r*res_z-1 #Te finel one in the upper right corner

    row = np.empty(5*res_r*res_z-2*res_r-2*res_z) #Initialize array for row index 
    row[0] = 0 #The way the sparse matrix is setup, the column indices are in the outer loop
    row[1] = 1 #and the row indeces go through and the row indices in the inner one:
    row[2] = res_z # A = ((i_col1, j_row1, data1), (i_col1, j_row2, data2), (i_col1, j_row3, , data3), (i_col2, j_row1, data4),... )
    for i in range(1, res_z-1):        
        row[i*4-1] = i-1   
        row[i*4] = i
        row[i*4+1] = i+1
        row[i*4+2] = i+res_z
    row[4*res_z-5] = res_z-2
    row[4*res_z-4] = res_z-1
    row[4*res_z-3] = 2*res_z-1
    for n in range(1, res_r-1):
        j = n*res_z
        l = j+res_z-1
        m = (n-1)*(8+5*res_z-10)+4*res_z-2
        row[m] = j-res_z
        row[m+1] = j
        row[m+2] = j+1
        row[m+3] = j+res_z
        ii = np.arange(j+1, l, 1)
        for i in range(res_z-2):
            mm = m+4+5*i  
            row[mm] = ii[i]-res_z
            row[mm+1] = ii[i]-1
            row[mm+2] = ii[i]
            row[mm+3] = ii[i]+1
            row[mm+4] = ii[i]+res_z
        row[mm+5] = l-res_z
        row[mm+6] = l-1
        row[mm+7] = l
        row[mm+8] = l+res_z
    row[mm+9] = (res_r-2)*res_z
    row[mm+10] = (res_r-1)*res_z
    row[mm+11] = (res_r-1)*res_z+1
    ii = np.arange(res_z*(res_r-1)+1,res_z*res_r-1, 1)
    m = 5*res_r*res_z-2*res_r-6*res_z+5
    for i in range(res_z-2):
        row[m+i*4] = ii[i]-res_z
        row[m+i*4+1] = ii[i]-1
        row[m+i*4+2] = ii[i]
        row[m+i*4+3] = ii[i]+1
    row[5*res_r*res_z-2*res_r-2*res_z-3] = (res_r-1)*res_z-1
    row[5*res_r*res_z-2*res_r-2*res_z-2] = res_r*res_z-2
    row[5*res_r*res_z-2*res_r-2*res_z-1] = res_r*res_z-1

    dz_l = np.empty(res_z-2) #Initialize array for matrix elements according to the second derivative in z
    dz_m = np.empty(res_z-2) #The second spatial derivative of phi is approximated by
    dz_r = np.empty(res_z-2) #d^2(phi)/dz^2 = dz_l*phi(z-dz) + dz_m*phi(z) + dz_r*phi(z+dz)
    for i in range(1, res_z-1): 
        dz_l[i-1] = 2/((dz[i-1] + dz[i])*dz[i-1]) #These coefficients take the variable grid spacing into account
        dz_m[i-1] = -(2/dz[i-1] + 2/dz[i])/(dz[i-1] + dz[i]) 
        dz_r[i-1] = 2/((dz[i-1] + dz[i])*dz[i]) 

    data = np.empty(5*res_r*res_z-2*res_r-2*res_z)
    data[0] =-2/dz[0]**2-2/dr[0]**2 
    data[1] = 2/dz[0]**2 #For z = +/-L/2 the coefficient to the left/right is zero
    data[2] = 2/dr[0]**2 #representing the grounded wall.
    for j in range(res_z-2):
        data[4*j+3] = dz_l[j]
        data[4*j+4] = dz_m[j]-2/dr[0]**2
        data[4*j+5] = dz_r[j]
        data[4*j+6] = 2/dr[0]**2 #Along the left column (r=0) the first derivative in r is zero phi(r-dr) = phi(r+dr)
    data[4*res_z-5] = 1/dz[-1]**2
    data[4*res_z-4] = -2/dz[-1]**2-2/dr[0]**2
    data[4*res_z-3] = 2/dr[0]**2

    for i in range(1, res_r-1):                      #Analogously the second derivative in r is approximated by
        dr_l = (-1/r[i]+2/dr[i-1])/(dr[i-1] + dr[i]) #d^2(phi)/dr^2 = dr_l*phi(r-dr) + dr_m*phi(r) + dr_r*phi(r+dr)
        dr_m = -(2/dr[i-1]+2/dr[i])/(dr[i-1] + dr[i])
        dr_r = (1/r[i]+2/dr[i])/(dr[i-1] + dr[i])
        m = (i-1)*(8+5*res_z-10)+4*res_z-2
        if mask[i, 0]==1:
            data[m] = 0
            data[m+1] = 1
            data[m+2] = 0
            data[m+3] = 0
        else:
            data[m] = dr_l
            data[m+1] = -2/dz[0]**2+dr_m
            data[m+2] = 2/dz[0]**2
            data[m+3] = dr_r
        for j in range(res_z-2):
            mm = m+4+5*j  
            if mask[i, j]==1: #Where mask = 1 an electrode is present. 
                              #The bias is given by a constrained on the density f = q*n/eps_0 = phi_electrode
                data[mm] = 0
                data[mm+1] = 0
                data[mm+2] = 1
                data[mm+3] = 0
                data[mm+4] = 0
                continue
            data[mm] = dr_l
            data[mm+1] = dz_l[j]
            data[mm+2] = dz_m[j] + dr_m
            data[mm+3] = dz_r[j]
            data[mm+4] = dr_r
        data[mm+5] = dr_l
        data[mm+6] = dz_l[-1]
        data[mm+7] = dz_m[-1]+dr_m
        data[mm+8] = dr_r
    dr_l = (-1/r[-1]+2/dr[-2])/(dr[-1] + dr[-2])
    dr_m = -(2/dr[-1]+2/dr[-2])/(dr[-1] + dr[-2])
    dr_r = (1/r[-1]+2/dr[-1])/(dr[-1] + dr[-2])
    data[mm+9] = dr_l
    data[mm+10] = dz_m[0]+dr_m
    data[mm+11] = 2/dz[0]**2
    m = 5*res_r*res_z-2*res_r-6*res_z+5
    for j in range(res_z-2):
        data[m+j*4] = dr_l
        data[m+j*4+1] = dz_l[j]
        data[m+j*4+2] = dz_m[j] + dr_m
        data[m+j*4+3] = dz_r[j]
    data[5*res_r*res_z-2*res_r-2*res_z-3] = dr_l
    data[5*res_r*res_z-2*res_r-2*res_z-2] = dz_l[-1]
    data[5*res_r*res_z-2*res_r-2*res_z-1] = dz_m[-1]+dr_m   
    
    return data, col, row


@njit(parallel = True)
def array22d(f, res_r, res_z):
    '''
    f: 1d numpy array - Output will be a 2d version of this array
    res_r: Integer - Number of radial grid points
    res_z: Integer - Number of longitudinal grid points
    
    Rearranging the 1d arrays f(i*res_z+j) from the matrix calculation into 2d array f(i, j).
    The index i walks you through radially and j goes longitudinally
    '''
    
    f_new = np.zeros((res_r, res_z)) 
    for i in prange(res_r): 
        for j in range(res_z): 
            f_new[i, j] = f[i*res_z+j]     
    return f_new


# In[4]:


@njit(parallel = True)
def AGM_Dipole(itr, r, z, rz, dz, C,res_r, res_z, R):
    '''
    itr: Integer - Number of iteration for AGM - algorithm (4 should be enough)
    r: 1d numpy array - Radial position of grid points
    z: 1d numpy array - Longitudinal position of grid points
    rz: 3d numpy array - Meshgrid of position of grid points in rz-plane
    dz: 1d numpy array - longitudinal distance of grid points (len(dz) = len(z)-1)
    C: Float - C = mu_0*I/2*pi
    i_ie: Integer - Radial index if the inner edge of the coil 
    i_oe: Integer - Radial index if the outer edge of the coil 
    i_le: Integer - Radial index if the lower edge of the coil 
    i_ue: Integer - Radial index if the upper edge of the coil 
    res_r: Integer - Number of radial grid points
    res_z: Integer - Number of longitudinal grid points
    
    Calculating the magnetic field and magnetic flux of the inner coil from elliptic integrals in the 2D r-z plane
    ("Classical Electrodynamics", J.D. Jackson, John Wiley & Sons, 3 'd
    Edition, 1998, pp. 181-183.)
    The eliptic integrals are evaluated using the arithmetic-geometric mean
    (https://dlmf.nist.gov/19.8)
    '''
    
    Bz = np.empty((res_r, res_z))
    Br = np.empty((res_r, res_z))
    psi = np.empty((res_r, res_z))

    alpha2 = (rz[1] - R)**2 + rz[0]**2 
    beta2 = (rz[1] + R)**2 + rz[0]**2
    beta = np.sqrt(beta2)
    k2 = 4*R*rz[1]/beta2
    k = np.sqrt(k2)

    a_i = np.ones((res_r, res_z))
    b_i = np.sqrt(1 - k2)

    a_i0 = a_i.copy()
    a_i = (a_i + b_i)/2
    b_i = np.sqrt(a_i0*b_i) 
    c = a_i**2
    for n in range(2, itr+2): #Iteration scheme
        a_i0 = a_i.copy()
        a_i = (a_i + b_i)/2
        b_i = np.sqrt(a_i0*b_i)
        c -= 2**(n-1)*(a_i**2-b_i**2)

    K = np.pi/(2*a_i) #Complete elliptic integral of first kind
    E = K*c          #Complete elliptic integral of second kind

    for x in range(res_r):
        for y in range(res_z):
            if k2[x, y] == 1.0: #For E(k=1) this method converges
                E[x, y] = 1.0   #But the value of E is known to be 1

    #The Magnetic field/flux for each grid point is calculated in the following

    Bz = C*((R**2-rz[1]**2-rz[0]**2)*E/alpha2 + K)/np.sqrt(beta2)
    Br = rz[0]*C*((R**2+rz[1]**2+rz[0]**2)*E/alpha2 - K)/(np.sqrt(beta2)*rz[1])
    psi = C*np.sqrt(beta2)*(((R**2+rz[1]**2+rz[0]**2)/beta2)*K - E)
    phi_coil = 4*R*K/beta

    return psi, Bz, Br, phi_coil


@njit
def AGM_0D(itr, r, z, R, C):
    '''
    itr: Integer - Number of iteration for AGM - algorithm (4 should be enough)
    r: 1d numpy array - Radial position of grid points
    z: 1d numpy array - Longitudinal position of grid points
    rz: 3d numpy array - Meshgrid of position of grid points in rz-plane
    dz: 1d numpy array - longitudinal distance of grid points (len(dz) = len(z)-1)
    C: Float - C = mu_0*I/2*pi
    i_ie: Integer - Radial index if the inner edge of the coil 
    i_oe: Integer - Radial index if the outer edge of the coil 
    i_le: Integer - Radial index if the lower edge of the coil 
    i_ue: Integer - Radial index if the upper edge of the coil 
    res_r: Integer - Number of radial grid points
    res_z: Integer - Number of longitudinal grid points
    
    Calculating the magnetic field and magnetic flux of the inner coil from elliptic integrals at a specific grid point
    ("Classical Electrodynamics", J.D. Jackson, John Wiley & Sons, 3 'd
    Edition, 1998, pp. 181-183.)
    The eliptic integrals are evaluated using the arithmetic-geometric mean
    (https://dlmf.nist.gov/19.8)
    '''

    alpha2 = (r - R)**2 + z**2 
    beta2 = (r + R)**2 + z**2
    beta = np.sqrt(beta2)
    k2 = 4*R*r/beta2
    k = np.sqrt(k2)

    a_i = 1
    b_i = np.sqrt(1 - k2)

    a_i0 = a_i
    a_i = (a_i + b_i)/2
    b_i = np.sqrt(a_i0*b_i) 
    c = a_i**2
    for n in range(2, itr+2): #Iteration scheme
        a_i0 = a_i
        a_i = (a_i + b_i)/2
        b_i = np.sqrt(a_i0*b_i)
        c -= 2**(n-1)*(a_i**2-b_i**2)

    K = np.pi/(2*a_i) #Complete elliptic integral of first kind

    if k2 == 1.0: #For E(k=1) this method converges
        E = 1.0   #But the value of E is known to be 1
    else:
        E = K*c          #Complete elliptic integral of second kind

    #The Magnetic field/flux for each grid point is calculated in the following

    dpsi_dr = r*C*((R**2-r**2-z**2)*E/alpha2 + K)/np.sqrt(beta2)
    dpsi_dz = -z*C*((R**2+r**2+z**2)*E/alpha2 - K)/(np.sqrt(beta2))
    psi = C*np.sqrt(beta2)*(((R**2+r**2+z**2)/beta2)*K - E)

    return psi, dpsi_dr, dpsi_dz


@njit
def AGM_wall(itr, R, z, rz, dz, C, H, res_r, res_z):
    '''
    itr: Integer - Number of iteration for AGM - algorithm (4 should be enough)
    R: float - Radius of the outer coil
    z: 1d numpy array - Longitudinal position of grid points
    rz: 3d numpy array - Meshgrid of position of grid points in rz-plane
    dz: 1d numpy array - longitudinal distance of grid points (len(dz) = len(z)-1)
    C: Float - C = mu_0*I/2*pi
    H: float - Height of the outer coil (H=L)
    res_r: Integer - Number of radial grid points
    res_z: Integer - Number of longitudinal grid points
    
    Calculating the magnetic field and magnetic flux of the outer coil from elliptic integrals at z = R_wall
    ("Classical Electrodynamics", J.D. Jackson, John Wiley & Sons, 3 'd
    Edition, 1998, pp. 181-183.)
    The eliptic integrals are evaluated using the arithmetic-geometric mean
    (https://dlmf.nist.gov/19.8)
    '''
    
    Chi_p = rz[0]+H/2
    Chi_m = rz[0]-H/2
    
    alpha2 = (rz[1] - R)**2 +  Chi_p**2 
    beta2 = (rz[1] + R)**2 + Chi_p**2
    beta = np.sqrt(beta2)
    k2 = 4*R*rz[1]/beta2
    k = np.sqrt(k2)
    h2 = 4*R*rz[1]/(rz[1] + R)**2
    
    Q_i = np.ones((res_r, res_z))
    Q_n = Q_i.copy()
    p_i2 = 1 - h2
    a_i = np.ones((res_r, res_z))
    b_i = np.sqrt(1 - k2)
    
    Q_i = Q_i*(p_i2 - a_i*b_i)/(2*(p_i2 + a_i*b_i))
    Q_n += Q_i
    p_i = (p_i2 + a_i*b_i)/(2*np.sqrt(p_i2))
    a_i0 = a_i.copy()
    a_i = (a_i + b_i)/2
    b_i = np.sqrt(a_i0*b_i) 
    c = a_i**2
    for i in range(2, itr+2): #Iteration scheme
        Q_i = Q_i*(p_i**2 - a_i*b_i)/(2*(p_i**2 + a_i*b_i))
        Q_n += Q_i
        p_i = (p_i**2 + a_i*b_i)/(2*p_i)
        a_i0 = a_i.copy()
        a_i = (a_i + b_i)/2
        b_i = np.sqrt(a_i0*b_i)
        c -= 2**(i-1)*(a_i**2-b_i**2)

    K = np.pi/(2*a_i) #Complete elliptic integral of first kind
    E = K*c          #Complete elliptic integral of second kind
    I = np.pi*(2 + Q_n*h2/(1-h2))/(4*a_i)
    for x in range(res_r):
        for y in range(res_z):
            if k2[x, y] == 1.0: #For E(k=1) this method converges
                E[x, y] = 1.0   #But the value of E is known to be 1
    #The Magnetic field/flux for each grid point is calculated in the following
    
    Bz = Chi_p*C*(K + I*(R - rz[1])/(R + rz[1]))/(H*beta)    
    Br = C*((2*R/beta - beta/rz[1])*K + E*beta/rz[1])/(2*H)    
    psi = Chi_p*C*(((rz[1]-R)**2/beta+beta)*K/2 - beta*E/2 - (rz[1]-R)**2*I/(2*beta))/(H)
    
    alpha2 = (rz[1] - R)**2 +  Chi_m**2 
    beta2 = (rz[1] + R)**2 + Chi_m**2
    beta = np.sqrt(beta2)
    k2 = 4*R*rz[1]/beta2
    k = np.sqrt(k2)
    h2 = 4*R*rz[1]/(rz[1] + R)**2
    
    Q_i = np.ones((res_r, res_z))
    Q_n = Q_i.copy()
    p_i2 = 1 - h2
    a_i = np.ones((res_r, res_z))
    b_i = np.sqrt(1 - k2)
    
    Q_i = Q_i*(p_i2 - a_i*b_i)/(2*(p_i2 + a_i*b_i))
    Q_n += Q_i
    p_i = (p_i2 + a_i*b_i)/(2*np.sqrt(p_i2))
    a_i0 = a_i.copy()
    a_i = (a_i + b_i)/2
    b_i = np.sqrt(a_i0*b_i) 
    c = a_i**2
    for i in range(2, itr+2): #Iteration scheme
        Q_i = Q_i*(p_i**2 - a_i*b_i)/(2*(p_i**2 + a_i*b_i))
        Q_n += Q_i
        p_i = (p_i**2 + a_i*b_i)/(2*p_i)
        a_i0 = a_i.copy()
        a_i = (a_i + b_i)/2
        b_i = np.sqrt(a_i0*b_i)
        c -= 2**(i-1)*(a_i**2-b_i**2)

    K = np.pi/(2*a_i) #Complete elliptic integral of first kind
    E = K*c          #Complete elliptic integral of second kind
    I = np.pi*(2 + Q_n*h2/(1-h2))/(4*a_i)
    for x in range(res_r):
        for y in range(res_z):
            if k2[x, y] == 1.0: #For E(k=1) this method converges
                E[x, y] = 1.0   #But the value of E is known to be 1
    #The Magnetic field/flux for each grid point is calculated in the following
    
    Bz -= Chi_m*C*(K + I*(R - rz[1])/(R + rz[1]))/(H*beta)    
    Br -= C*((2*R/beta - beta/rz[1])*K + E*beta/rz[1])/(2*H)    
    psi -= Chi_m*C*(((rz[1]-R)**2/beta+beta)*K/2 - beta*E/2 - (rz[1]-R)**2*I/(2*beta))/(H)
    


    return psi, Bz, Br

@njit(parallel=True)
def norm_psi(f_psi_beta, B_square, M_r):
    '''
    f_psi_beta: 2d numpy array - f = -q*n/eps_0, this array is given in magnetic field coordinates
    B_square: 2d numpy array - Magnetic field squared in magnetic field coordinates
    M_r: 1d numpy array - Desired number of particles on a flux contour
    
    Normalizes the number of particles on a flux contour to match M_r
    '''
        
    for i in prange(len(f_psi_beta)):
        F_psi = 0
        for j in range(len(f_psi_beta[i])):
            F_psi = F_psi + f_psi_beta[i][j]/(B_square[i][j])
        if F_psi == 0:
            f_psi_beta[i] = 0
        else:
            f_psi_beta[i] = M_r[i]*f_psi_beta[i]/F_psi
    return f_psi_beta

@njit(parallel=True)
def F_norm(f_2d, F0, r, z, dr, dz, mask, res_r, res_z): 
    '''
    f_2d: 2d numpy array - f = -q*n/eps_0, this array is given in magnetic field coordinates
    B_square: 2d numpy array - Magnetic field squared in magnetic field coordinates
    M_r: 1d numpy array - Desired number of particles on a flux contour
    
    Normalizes the number of particles on a flux contour to match M_r
    '''
    F = 0
    for i in range(1, res_r-1):
        if mask[i, 0] == 1:
            F = F + np.pi*f_2d[i, 0]*(r[i+1]**2-r[i-1]**2+2*r[i]*(r[i+1]-r[i-1]))*dz[0]/4
        for j in range(1, res_z-1):
            F = F + np.pi*f_2d[i, j]*(r[i+1]**2-r[i-1]**2+2*r[i]*(r[i+1]-r[i-1]))*(z[j+1]-z[j-1])/4
    f_2d = F0*f_2d/F #Normalize density distribution to match number of particles
    
    return f_2d


@njit(parallel = True)
def GTE_coil(phi_coil, dr, dz, r, res_r, res_z, T, F_neg, F_pos, i_ie, i_oe, i_le, i_ue, k, e):
    '''
    dr: 1d numpy array - longitudinal distance of grid points (len(dr) = len(r)-1)
    dz: 1d numpy array - longitudinal distance of grid points (len(dz) = len(z)-1)
    r: 1d numpy array - Radial position of grid points
    res_r: Integer - Number of radial grid points
    res_z: Integer - Number of longitudinal grid points
    T: Float - Plasma temperature
    F_neg: Total negative space charge (F = -qN/eps0)
    F_pos: Total positive space charge
    i_ie: Integer - Radial index if the inner edge of the coil 
    i_oe: Integer - Radial index if the outer edge of the coil 
    i_le: Integer - Radial index if the lower edge of the coil 
    i_ue: Integer - Radial index if the upper edge of the coil 
    k: Float - Boltzmann constant
    e: Float - Elementary charge
    
    Calculates the distribution of an artificial plasma inside the coil that represents the image charge.
    This ensures an equipotential on the coil's surface that can be manipulated by changing the ratio of
    F_neg/F_pos. The temperature should be chosen such that the Debye length is on the order of the grid 
    spacing.
    '''
    f_pos = np.zeros((res_r,res_z)) #Initialize positive charge distribution
    f_neg = np.zeros((res_r,res_z)) #Initialize negative charge distribution
    f_neg_dV = 0 #Auxiliary variable for volumetric integral
    f_pos_dV = 0  
    for j in prange(res_z):
        f_neg[:, j] = np.exp(e*phi_coil[:, j]/(k*T))#Boltzmann factor for negative charge
        f_pos[:, j] = -np.exp(-e*phi_coil[:, j]/(k*T))#Boltzmann factor for positive charge
        f_neg_dV += np.sum(f_neg[:res_r, j]*r[i_ie:i_oe]*dr[i_ie:i_oe])*dz[j+i_le] #Volumetric integral
        f_pos_dV += np.sum(f_pos[:res_r, j]*r[i_ie:i_oe]*dr[i_ie:i_oe])*dz[j+i_le] #of the density distribution
    f_neg_dV = 2*np.pi*f_neg_dV
    f_pos_dV = 2*np.pi*f_pos_dV
    f_neg = f_neg*F_neg/f_neg_dV #Normalize density distribution to match predefined number of particles
    f_pos = f_pos*F_pos/f_pos_dV
    f_coil = f_neg + f_pos  
    return f_coil

@njit(parallel=True)
def inter_psi_to_rz(psi, psi_grid_r, psi_grid_z, r, z, f_psi_beta, psi_r, mask, res_r, res_z):
    f_2d = np.zeros((res_r, res_z))
    n_max = len(psi_grid_r)
    m_max = len(psi_grid_r[0])
    for i in prange(res_r):
        for j in range(res_z):
            if mask[i, j] == 1:
                dn1 = abs(psi_r[0]-psi[i, j])
                for n in range(n_max-1):
                    dn0 = dn1
                    dn1 = abs(psi_r[n+1]-psi[i, j])
                    if dn1>dn0:
                        break
                dm1 = (psi_grid_r[n][0]-r[i])**2 + (psi_grid_z[n][0]-z[j])**2
                for m in range(m_max-1):
                    dm0 = dm1
                    dm1 = (psi_grid_r[n][m+1]-r[i])**2 + (psi_grid_z[n][m+1]-z[j])**2
                    if dm1>dm0:
                        break
                ds0 = (psi_grid_r[n][m]-r[i])**2 + (psi_grid_z[n][m]-z[j])**2
                ds1 = (psi_grid_r[n+1][m]-r[i])**2 + (psi_grid_z[n+1][m]-z[j])**2
                if n == 0:
                    ds2 = 10*ds1
                else:
                    ds2 = (psi_grid_r[n-1][m]-r[i])**2 + (psi_grid_z[n-1][m]-z[j])**2
                ds3 = (psi_grid_r[n][m+1]-r[i])**2 + (psi_grid_z[n][m+1]-z[j])**2
                if m == 0:
                    ds4 = 10*ds3
                else:
                    ds4 = (psi_grid_r[n][m-1]-r[i])**2 + (psi_grid_z[n][m-1]-z[j])**2
                if ds1<ds2:
                    if ds3<ds4:
                        f_2d[i, j] = ds0*f_psi_beta[n][m]*(1/(ds0+ds1)+1/(ds0+ds3)) + ds1*f_psi_beta[n+1][m]/(ds0+ds1) + ds3*f_psi_beta[n][m+1]/(ds0+ds3)
                    elif ds3>ds4:
                        f_2d[i, j] = ds0*f_psi_beta[n][m]*(1/(ds0+ds1)+1/(ds0+ds4)) + ds1*f_psi_beta[n+1][m]/(ds0+ds1) + ds4*f_psi_beta[n][m-1]/(ds0+ds4)
                elif ds1>ds2:
                    if ds3<ds4:
                        f_2d[i, j] = ds0*f_psi_beta[n][m]*(1/(ds0+ds2)+1/(ds0+ds3)) + ds2*f_psi_beta[n-1][m]/(ds0+ds2) + ds3*f_psi_beta[n][m+1]/(ds0+ds3)
                    elif ds3>ds4:
                        f_2d[i, j] = ds0*f_psi_beta[n][m]*(1/(ds0+ds2)+1/(ds0+ds4)) + ds2*f_psi_beta[n-1][m]/(ds0+ds2) + ds4*f_psi_beta[n][m-1]/(ds0+ds4)
    return f_2d


@njit(parallel = True)
def GTE(phi_tot, q, mask, T, F0, r, z, dr, dz, res_r, res_z, i_ie, i_oe, i_le, i_ue): 
    '''
    phi_2d: 2d numpy array - Electrostatic potential in rz-plane
    psi: 2d numpy array - Magnetic flux in rz-plane
    cent: 2d numpy array - Effective mechanical potential in rz-plane
    psi: 2d numpy array - Mask of the confinement region
    omega: Float - Rotation frequency of the fram of reference
    T: Float - Plasma temperature
    F0: Total plasma space charge (F = -qN/eps0)
    r: 1d numpy array - Radial position of grid points
    dr: 1d numpy array - longitudinal distance between grid points (len(dr) = len(r)-1)
    dz: 1d numpy array - longitudinal distance between grid points (len(dz) = len(z)-1)
    res_r: Integer - Number of radial grid points
    res_z: Integer - Number of longitudinal grid points
    i_ie: Integer - Radial index if the inner edge of the coil 
    i_oe: Integer - Radial index if the outer edge of the coil 
    i_le: Integer - Radial index if the lower edge of the coil 
    i_ue: Integer - Radial index if the upper edge of the coil 
    
    Calculates the Boltzmann distribution of the plasma inside the confinement region.
    '''

    f_2d = mask*np.exp(-q*phi_tot/T) #Only consider Boltzmann distribution in the confinement region
    F = 0
    for i in range(1, res_r-1):
        for j in range(res_z-1):
            if mask[i, j] == 1:
                F = F + f_2d[i, j]*np.pi*(r[i+1]**2-r[i-1]**2+2*r[i]*(r[i+1]-r[i-1]))*dz[j]/2
    f_2d = F0*f_2d/F #Normalize density distribution to match number of particles

    return f_2d


# In[8]:


@njit(parallel = True)
def LTE(phi_2d, cent, psi, dpsi, T, mask, s, r, z, dr, dz, res_r, res_z, i_ie, i_oe, i_le, i_ue):     
    phi_psi = np.zeros(np.max(s)-np.min(s)+1)
    for i in prange(i_oe, res_r-1):
        phi_psi[s[i, 0]] = phi_2d[i, 0]
        if s[i, 0]-s[i-1, 0]  == 2:
            phi_psi[s[i-1, 0]+1] = (phi_2d[i, 0] + phi_2d[i-1, 0])/2
        elif s[i, 0]-s[i-1, 0]  == -2:
            phi_psi[s[i-1, 0]-1] = (phi_2d[i, 0] + phi_2d[i-1, 0])/2
    f_2d = np.zeros((res_r, res_z))
    for i in range(1, res_r-1):
        if mask[i, 0] == 1:
            f_2d[i, 0] = np.exp(phi_2d[i, 0]-phi_psi[s[i, 0]]-(phi_psi[s[i, 0]+1]-phi_psi[s[i, 0]])*(psi[i, 0]%dpsi)/dpsi+cent[i, 0]/T)
        for j in range(1, res_z-1):
            if mask[i, j] == 1:
                f_2d[i, j] = np.exp((phi_2d[i, j]-phi_psi[s[i, j]]-(phi_psi[s[i, j]+1]-phi_psi[s[i, j]])*(psi[i, j]%dpsi)/dpsi+cent[i, j])/T)               
    return f_2d


@njit(parallel = True)
def Phi_Psi(phi_2d, s, r, z, dr, dz, i_oe, res_r, res_z):
    
    phi_psi = np.zeros(np.max(s)-np.min(s)+1)
    for i in prange(i_oe, res_r-1):
        phi_psi[s[i, 0]] = phi_2d[i, 0]
        if s[i, 0]-s[i-1, 0]  == 2:
            phi_psi[s[i-1, 0]+1] = (phi_2d[i, 0] + phi_2d[i-1, 0])/2
        elif s[i, 0]-s[i-1, 0]  == -2:
            phi_psi[s[i-1, 0]-1] = (phi_2d[i, 0] + phi_2d[i-1, 0])/2

    return(phi_psi)


# In[12]:


@njit(parallel=True)
def well(phi_conf, res_r, res_z, i_oe, i_mid, i_bottom, frac):
    '''
    phi_tot: 2d numpy array - Total potential in rz-plane (phi + omega*psi - m*omega**2*r**2/2*q)
    r: 1d numpy array - Radial position of grid points
    res_r: Integer - Number of radial grid points
    res_z: Integer - Number of longitudinal grid points
    i_oe: Integer - Radial index if the outer edge of the coil  
    
    Find the confinement region by comparing the height of the potential well for every row and column.
    The lowest value sets the contour line that surrounds the confinement region.
    '''

    i_max_coil = np.argmax(phi_conf[i_oe:i_bottom, i_mid])+i_oe
    if i_max_coil == i_oe:
        i_max_coil = i_max_coil + 1
    phi_r_well_coil = phi_conf[i_max_coil, i_mid]
    phi_r_well_wall = np.max(phi_conf[i_bottom:, i_mid])
    i_min = np.argmin(phi_conf[i_oe:, i_mid])+i_oe
    phi_z_max = np.zeros(res_r-i_oe)
    for i in prange(i_oe, res_r):
        phi_z_max[i-i_oe] = np.max(phi_conf[i, :])
    phi_z_well = np.min(phi_z_max)
    phi_well = np.min(np.array([phi_r_well_coil, phi_r_well_wall, phi_z_well]))
    phi_well = phi_well - frac

    mask = np.zeros((res_r, res_z)) 

    for j in prange(res_z):
        check_r = 0
        for i0 in range(res_r-i_oe-1):
            i = res_r-1-i0
            if phi_conf[i, j] <= phi_well:# and (phi_conf[i, j-1]+phi_conf[i, j-2])/2 <= phi_conf[i, j]:
                check_r = 1
                mask[i, j] = 1
            if check_r == 1 and phi_conf[i, j] > phi_well:
                 break
        if check_r == 0:
            break
            
    mask_sum = np.zeros(res_z)
    for j in prange(res_z):
        mask_sum[j] = np.sum(mask[:, j])
    mask_min = np.min(mask_sum)
    if mask_min > 0:
        j_max = np.argmin(mask_sum)
        for i in prange(res_r):
            for j in range(j_max, res_z):
                mask[i, j] = 0
            
    return mask, phi_well

# In[13]:


@njit(parallel = True)
def capacity(phi_tot, T, mask, r, z, res_r, res_z, i_oe):
    '''
    phi_tot: 2d numpy array - Total potential, consisting of electrostatic, canonical and centrifugal term
    T: float - Plasma temperature
    r: 1d numpy array - Radial position of grid points
    z: 1d numpy array - Longitudinal position of grid points
    res_r: integer - Radial resolution
    res_z: integer - Longitudinial resolution
    i_oe: integer - Radial index of the outer edge of the inner coil
    
    Performs the volume integral of the confinement domain over the Boltzmann factor 
    as a measure for the capacity of the trap. (So more or less the particle number) 
    '''
    
    C = 0
    for i in prange(i_oe, res_r-1):
        for j in range(1, res_z-1):
            C += mask[i, j]*np.exp(phi_tot[i, j]/T)*2*np.pi*r[i]*(r[i+1]-r[i-1])/2*(z[j+1]-z[j-1])/2 
            #Volume integral over Boltzmann factor with vacuum potential as measure for the capacity of the trap
    return(C)


# In[14]:


def plot_2d(n_plasma, phi_tot, mask, r, z, R_coil, R_wall, i_ie, i_oe, i_le, i_ue, res_r, res_z):  
    '''
    n_plasma: 2d numpy array - Plasma density
    phi_tot: 2d numpy array - Total potential, consisting of electrostatic, canonical and centrifugal term
    mask: 2d numpy array - 1 at the confinement region, 0 everywhere else
    r: 1d numpy array - Radial position of grid points
    z: 1d numpy array - Longitudinal position of grid points
    R_coil: float - Radius coil
    R_wall: float - Radius wall
    i_ie: integer - Radial index of the inner edge of the inner coil
    i_oe: integer - Radial index of the outer edge of the inner coil
    i_le: integer - Longitudinal index of the lower edge of the inner coil
    i_ue: integer - Longitudinal index of the upper edge of the inner coi
    
    2D Contour plot of the plasma density (left) and the total potential (right) in the rz-plane. 
    The potential contour surrounding the confinement region and the edge of the coil is highlighted.
    
    '''
    mask_plot = mask.copy()
    #mask_plot[i_ie:i_oe, i_le:i_ue] = 0.5
    contour_mask = np.arange(0, 1.1, 0.1)
    n_plasma = n_plasma/np.max(n_plasma)
    #mag_n = int(round(np.log10(np.max(n_plasma))))-1
    maxi = 1.011#np.max(n_plasma)*1.1
    mini = -0.001#np.min(n_plasma)
    contours_n = np.arange(mini, maxi, (maxi-mini)/100)

    fig, axs = plt.subplots(1, 2, sharey='row', gridspec_kw={'wspace': 0.0}, figsize=(8, 4))
    (ax1, ax2) = axs

    im1 = ax1.contourf(r/R_coil, z/R_coil, np.transpose(n_plasma), contours_n, cmap='nipy_spectral')
    im_mask = ax1.contour(r/R_coil, z/R_coil, np.transpose(mask_plot), contour_mask, colors='w')
    ax1.invert_yaxis()
    ax1.set_xlim(R_wall/R_coil, 0)
    ax1.set_xlabel("r/$R_{coil}$", fontsize = 15)
    ax1.set_ylabel("z/$R_{coil}$", fontsize = 15)
    ax1.tick_params(axis ='both', labelsize=15)
    cb_ax1 = fig.add_axes([0.905, 0.52, 0.015, 0.36])
    cbar1 = fig.colorbar(im1, cax=cb_ax1, ticks = np.arange(0, 1.2, 0.2))
    tick_labels = [str(round(i, 1)) for i in np.arange(0, 1.2, 0.2)]
    cbar1.ax.set_yticklabels(tick_labels, fontsize=15) 
    cbar1.set_label(label = '$n/n_{max}$',fontsize=20)
    ax1.set_title('Density', fontsize=20)

    maxi = 1.2#np.max(phi_tot)*1.1
    mini = 0#np.min(phi_tot)
    contours_phi = np.arange(mini, maxi, (maxi-mini)/50)

    im2 = ax2.contour(r/R_coil, z/R_coil, np.transpose(phi_tot), contours_phi, cmap='inferno')
    im_mask = ax2.contour(r/R_coil, z/R_coil, np.transpose(mask_plot), contour_mask, colors='k')
    ax2.invert_yaxis()
    ax2.set_xlabel("r/$R_{coil}$", fontsize = 15)
    ax2.tick_params(axis ='both', labelsize=15)
    cb_ax2 = fig.add_axes([0.905, 0.125, 0.015, 0.37])
    cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks = np.arange(0, 1.2, 0.2))
    tick_labels = [str(round(i, 1)) for i in np.arange(0, 1.2, 0.2)]
    cbar2.ax.set_yticklabels(tick_labels, fontsize=15) 
    cbar2.set_label(label = '$\phi/\phi_{coil}$',fontsize=20)
    ax2.set_title('Electrostatic Potential', fontsize=20)
    #plt.savefig('Density_Potential.pdf')
    plt.show()
# In[16]:


@njit(parallel=True)
def Rotation(phi, psi, mask, n, T, q, c, me, omega, dr, dz, r, res_r, res_z):
    omega_EXB = np.zeros((res_r, res_z))
    omega_Dia = np.zeros((res_r, res_z))
    omega_cent = np.zeros((res_r, res_z))

    for i in prange(1, res_r-1):
        dr_i = dr[i]+dr[i-1]
        for j in range(1, res_z-1):
            if mask[i, j] == 1:
                dz_j = dz[j]+dz[j-1]
                EzBr = ((psi[i, j+1] - psi[i, j-1])*(phi[i, j+1] - phi[i, j-1]))/dz_j**2
                ErBz = ((psi[i+1, j] - psi[i-1, j])*(phi[i+1, j] - phi[i-1, j]))/dr_i**2
                grad_nzBr = ((psi[i, j+1] - psi[i, j-1])*(n[i, j+1] - n[i, j-1]))/dz_j**2
                grad_nrBz = ((psi[i+1, j] - psi[i-1, j])*(n[i+1, j] - n[i-1, j]))/dr_i**2
                B_square = ((psi[i, j+1] - psi[i, j-1])/dz_j)**2 + ((psi[i+1, j] - psi[i-1, j])/dr_i)**2

                omega_EXB[i, j] = -c*(ErBz + EzBr)/(B_square)
                omega_Dia[i, j] = T*c*(grad_nzBr + grad_nrBz)/(q*n[i, j]*B_square)
                omega_cent[i, j] = c*me*omega**2*(psi[i-1, j] - psi[i+1, j])*r[i]/(q*B_square*dr_i)
            
    omega_cent[0, :] = 0
    
    return omega_EXB, omega_Dia, omega_cent


# In[17]:


def grid(R_coil, R_wall, H, res_r, res_z, spacing_r, spacing_z, norm):
    '''
    R_coil: float - Radius of the coil
    R_wall: float - Radius of the wall
    H: float - Hight of the trap
    res_r: Integer - Radial resolution
    res_z: Integer - Longitudinal resolution
    
    Set up grid in rz-plane. The spacing of the grid points varies according to a supper-gaussian 
    with exponent 4 (it's 2 in the case of a regular gaussian). The minimum of the grid spacing is in the
    equatorial plane in between the inner coil and the wall. Check the grid spacing by using "check_grid()"
    to plot dr(r) and dz(z).
    '''
    
    r0 = np.linspace(0, R_wall, res_r) #Equispaced grid points for reference
    dr = []
    r = []
    for i in range(1, res_r-2): #This is sort of a first order approximation since I need the radius
        r, dr = spacing_r(r, dr, R_wall, R_coil, res_r) #and I need the spacing
    dr = np.array(dr) 
    dr = dr[::-1]
    if norm == 1:
        dr = R_wall*dr/np.sum(dr) #Normalize the spacing to make sure that sum(dr) = R_wall
    r = np.zeros(res_r)

    for i in range(1, res_r): #Recalculate the radial positions with the normalized spacing
        r[i] = r[i-1] + dr[i-1] 
    #Same procedure for the longitudinal spacing.
    z0 = np.linspace(0, H/2, res_z)
    dz = []
    z = []
    for i in range(1, res_z-2):
        z, dz = spacing_z(z, dz, R_coil, H, res_z)
    dz = np.array(dz[::-1])
    if norm == 1:
        dz = (H/2)*dz/np.sum(dz)
    z = np.zeros(res_z)

    for i in range(1, res_z):
        z[i] = z[i-1] + dz[i-1] 
    
    return r, z, dr, dz, r0, z0

# In[18]:


def check_grid(r, z, r0, z0, R_coil, dR_coil, dr, dz, res_r, res_z):
    '''
    r: 1d numpy array - Radial position of grid points (variable spacing)
    z: 1d numpy array - Longitudinal position of grid points (variable spacing)
    r0: 1d numpy array - Radial position of grid points (constant spacing)
    z0: 1d numpy array - Longitudinal position of grid points (constant spacing)
    R_coil: float - Radius of the coil
    dR_coil: float - Width of the coil
    dr: 1d numpy array - longitudinal distance between grid points (len(dr) = len(r)-1)
    dz: 1d numpy array - longitudinal distance between grid points (len(dz) = len(z)-1)
    res_r: Integer - Radial resolution
    res_z: Integer - Longitudinal resolution
    
    Set up grid in rz-plane. The spacing of the grid points varies according to a supper-gaussian 
    with exponent 4 (it's 2 in the case of a regular gaussian). The minimum of the grid spacing is in the
    equatorial plane in between the inner coil and the wall. Check the grid spacing by using "check_grid()"
    to plot dr(r) and dz(z).
    '''
    plt.plot(r0/R_coil, r0/R_coil, label = 'Equidistant')
    plt.plot(r0/R_coil, r/R_coil, 'r', label = 'Gaussian')
    plt.xlabel('$r0/R_{coil}$', fontsize=15)
    plt.ylabel('$r/R_{coil}$', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('r_grid.jpg')
    plt.show()

    plt.plot(r[:res_r-1]/R_coil, dr/R_coil, 'r', label = 'Grid Size')
    plt.vlines(R_coil/R_coil, 0, np.max(dr)/R_coil, label='Coil Center')
    plt.vlines((R_coil-dR_coil/2)/R_coil, 0, np.max(dr)/R_coil, label='Edge', linestyle='dashed')
    plt.vlines((R_coil+dR_coil/2)/R_coil, 0, np.max(dr)/R_coil, linestyle='dashed')
    plt.xlabel('$r/R_{coil}$', fontsize=15)
    plt.ylabel('$dr/R_{coil}$', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.plot(z0/R_coil, z0/R_coil, label = 'Equidistant')
    plt.plot(z0/R_coil, z/R_coil, 'r', label = 'Gaussian')
    #plt.vlines(zw_m*1e3, 0, np.max(z)*1e3, label='Coil Center')
    plt.xlabel('$z0/R_{coil}$', fontsize=15)
    plt.ylabel('$z/R_{coil}$', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('z_grid.jpg')
    plt.show()

    plt.plot(z[:res_z-1]/R_coil, dz/R_coil, 'r', label = 'Grid Size')
    plt.vlines((dR_coil/2)/R_coil, 0, np.max(dz)/R_coil, label='Edge', linestyle='dashed')
    plt.xlabel('$z/R_{coil}$', fontsize=15)
    plt.ylabel('$z/R_{coil}$', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def psi_beta_grid(dpsi, dbeta, psi_max, R_wall, R_coil, dR_coil, r, i_ie, i_oe, C, max_len):
    r_psi = np.empty(max_len)
    psi_r = np.empty(max_len)
    r_psi[0] = R_wall
    dr0 = 0.001
    for n in range(1, max_len):
        psi_n, dpsi_dr, dpsi_dz = AGM_0D(20, r, r_psi[n-1], 0, i_ie, i_oe, C, dR_coil)
        if abs(dpsi/dpsi_dr) > abs(3*dr0):
            r_psi[n] = r_psi[n-1]-abs(3*dr0)
        else:
            r_psi[n] =  r_psi[n-1]-abs(dpsi/dpsi_dr)
            dr0 = dpsi/dpsi_dr
        psi_r[n-1] = psi_n
        if r_psi[n]<R_coil+dR_coil/2:
            break  
    if n>=max_len-1:
        print('More then 5000 grid points in psi required. Did not finish!')
    psi_n, dpsi_dr, dpsi_dz = AGM_0D(20, r, R_coil+dR_coil/2, 0, i_ie, i_oe, C, dR_coil)
    psi_r[n] = psi_n
    r_psi = r_psi[:n]
    r_psi = r_psi[::-1]
    psi_r = psi_r[:n]
    psi_r = psi_r[::-1]
    
    i_psi = np.argmin(abs(psi_r-np.min(psi_max)))+1
    psi_grid_r = [[i] for i in r_psi[i_psi:]]
    psi_grid_z = [[0] for i in r_psi[i_psi:]]
    B_square = []

    for i in range(len(r_psi[i_psi:])):
        psi_n, dpsi_dr, dpsi_dz = AGM_0D(20, r, r_psi[i_psi+i], 0, i_ie, i_oe, C, dR_coil)
        norm = (dpsi_dr**2 + dpsi_dz**2)
        B_square.append([norm/r_psi[i_psi+i]**2])

    mask_grid = np.ones(len(r_psi[i_psi:]))
    z_check = 0
    norm0 = 1
    for n in range(max_len):
        for i in range(len(r_psi[i_psi:])):
            if mask_grid[i] == 0:
                continue
            try:
                psi_n, dpsi_dr, dpsi_dz = AGM_0D(20, r, psi_grid_r[i][n], psi_grid_z[i][n], i_ie, i_oe, C, dR_coil)
            except:
                if n>0 and i>0:
                    r_mean = (psi_grid_r[i][n-1]+psi_grid_r[i-1][n])/2
                    z_mean = (psi_grid_z[i][n-1]+psi_grid_z[i-1][n])/2
                    psi_n, dpsi_dr, dpsi_dz = AGM_0D(20, r, r_mean, z_mean, i_ie, i_oe, C, dR_coil)
            norm0 = norm
            norm = np.nan_to_num(dpsi_dr**2 + dpsi_dz**2)
            if norm == 0:
                norm = norm0
            ds_r = np.nan_to_num(psi_grid_r[i][n]*dbeta*dpsi_dz/norm)
            ds_z = -np.nan_to_num(psi_grid_r[i][n]*dbeta*dpsi_dr/norm)
            psi_grid_r[i].append(psi_grid_r[i][n]+ds_r) 
            psi_grid_z[i].append(psi_grid_z[i][n]+ds_z)
            B_square[i].append(np.nan_to_num(norm/psi_grid_r[i][n]**2))
            if psi_grid_r[i][n] == 0:
                print('r', i , n)
            if psi_grid_z[i][n+1] <= 0:
                z_check = 1
        if z_check == 1:
            break
    if n>=max_len-1:
        print('More then 5000 grid points in beta required. Did not finish!')
    for i in range(len(r_psi[i_psi:])):
        psi_grid_r[i] = np.array(psi_grid_r[i])
        psi_grid_z[i] = np.array(psi_grid_z[i])
        B_square[i] = np.array(B_square[i])
    psi_grid_z = np.array(psi_grid_z)
    psi_grid_r = np.array(psi_grid_r)
    B_square = np.array(B_square)
    
    return psi_grid_r, psi_grid_z, B_square, r_psi


def check_B(Br, Bz, r, z, dr, dz, R_coil, res_r, res_z, i_ie, i_oe):
    dBz_dr = np.zeros((res_r, res_z))
    dBr_dz = np.zeros((res_r, res_z))

    for i in range(1, res_r-2):
        dr_i = dr[i] + dr[i+1]
        for j in range(1, res_z-2):
            dz_j = dz[j]+dz[j+1]
            dBr_dz[i, j] = (Br_wall[i, j+1] - Br_wall[i, j-1])/(dz_j)
            dBz_dr[i, j] = (Bz_wall[i+1, j] - Bz_wall[i-1, j])/(dr_i)

    dBz_dr[i_ie:i_oe, i_le:i_ue] = 0
    dBr_dz[i_ie:i_oe, i_le:i_ue] = 0
    dBz_dr_mean = np.mean(dBz_dr[i_oe:res_r, res_z//2])
    maxi = np.max(dBz_dr[i_oe+1:, :])
    mini = np.min(dBz_dr[i_oe+1:, :])
    contours_dBz_dr = np.arange(mini, maxi, (maxi-mini)/100)

    fig, axs = plt.subplots(1, 2, sharey='row', gridspec_kw={'wspace': 0.0}, figsize=(10, 8))
    (ax1, ax2) = axs

    im1 = ax1.contour(r[1:res_r]/R_coil, z[1:res_z]/R_coil, np.transpose(dBz_dr[1:res_r, 1:res_z]), contours_dBz_dr, cmap='nipy_spectral')
    ax1.invert_yaxis()
    ax1.set_xlim(R_wall/R_coil, 0)
    ax1.set_xlabel("$r/R_{coil}$", fontsize = 15)
    ax1.set_ylabel("$z/R_{coil}$", fontsize = 15)
    ax1.tick_params(axis ='both', labelsize=15)
    cb_ax1 = fig.add_axes([0.905, 0.52, 0.015, 0.36])
    cbar1 = fig.colorbar(im1, cax=cb_ax1, ticks = np.arange(mini, maxi, (maxi-mini)/5))
    tick_labels = [str(round(i, 2)) for i in np.arange(mini, maxi, (maxi-mini)/5)]
    cbar1.ax.set_yticklabels(tick_labels, fontsize=15) 
    cbar1.set_label(label = '$\partial B_z / \partial r$ [T/m]',fontsize=20)
    ax1.set_title('$\partial B_z / \partial r$', fontsize=20)

    im2 = ax2.contour(r[1:res_r]/R_coil, z[1:res_z]/R_coil, np.transpose(dBr_dz[1:res_r, 1:res_z]), contours_dBz_dr, cmap='nipy_spectral')
    ax2.invert_yaxis()
    ax2.set_xlabel("$r/R_{coil}$", fontsize = 15)
    ax2.tick_params(axis ='both', labelsize=15)
    cb_ax2 = fig.add_axes([0.905, 0.125, 0.015, 0.37])
    cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks = np.arange(mini, maxi, (maxi-mini)/5))
    tick_labels = [str(round(i, 2)) for i in np.arange(mini, maxi, (maxi-mini)/5)]
    cbar2.ax.set_yticklabels(tick_labels, fontsize=15) 
    cbar2.set_label(label = '$\partial B_r / \partial z$ [T/m]',fontsize=20)
    ax2.set_title('$\partial B_r / \partial z$', fontsize=20)
    plt.show()


# In[20]:


@njit(parallel=True)
def convolve(phi_psi_2d, len_kernel):
    len_kernel2 = int(len_kernel//2)
    kernel = np.zeros((len_kernel, len_kernel))
    len_r = int(len(phi_psi_2d))
    len_z = int(len(phi_psi_2d[0]))
    for i in prange(len_kernel):
        for j in range(len_kernel):
            kernel[i, j] = np.exp(-(((i-len_kernel2)/len_kernel2)**2+((j-len_kernel2)/len_kernel2)**2))
    kernel = kernel/np.sum(kernel)
    phi_psi_conv = np.zeros((len_r, len_z))
    for i in prange(len_kernel2, len_r-len_kernel2):
        for n in range(len_kernel2):
            phi_psi_conv[i, n] = np.sum(kernel[:, len_kernel2-n:]*phi_psi_2d[i-len_kernel2:i+len_kernel2+1, :len_kernel2+1+n])/np.sum(kernel[:, len_kernel2-n:])
        for j in range(len_kernel2, len_z-len_kernel2):
            phi_psi_conv[i, j] = np.sum(kernel*phi_psi_2d[i-len_kernel2:i+len_kernel2+1, j-len_kernel2:j+len_kernel2+1])
    return(phi_psi_conv)

@njit(parallel=True)
def F_norm(f_2d, F0, r, z, dr, dz, mask, res_r, res_z): 
    F = 0
    for i in range(1, res_r-1):
        if mask[i, 0] == 1:
            F = F + np.pi*f_2d[i, 0]*(r[i+1]**2-r[i-1]**2+2*r[i]*(r[i+1]-r[i-1]))*dz[0]/4
        for j in range(1, res_z-1):
            if mask[i, j] == 1:
                F = F + np.pi*f_2d[i, j]*(r[i+1]**2-r[i-1]**2+2*r[i]*(r[i+1]-r[i-1]))*(z[j+1]-z[j-1])/4
    f_2d = F0*f_2d/F #Normalize density distribution to match number of particles
    
    return f_2d

@njit(parallel=True)
def N_error(d2phi, f, mask, r, z, dz, e, res_r, res_z):
    error = d2phi+f     
    E = 0
    for i in prange(1, res_r-1):
        for j in range(res_z-1):
            E = E + mask[i, j]*np.pi*error[i*res_z+j]*(r[i+1]**2-r[i-1]**2+2*r[i]*(r[i+1]-r[i-1]))*dz[j]/2
    return(E/(4*np.pi*e))

@njit#(parallel = True)
def N_tot(n, r, z, dr, dz, res_r, res_z):
    N = 0
    for i in range(1, res_r-1):
        for j in range(res_z-1):
            N = N + np.pi*n[i, j]*(r[i+1]**2-r[i-1]**2+2*r[i]*(r[i+1]-r[i-1]))*dz[j]/2
    return N