import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import qpsolvers as qp
import sys

def gaussian(x:np.ndarray, mean:np.ndarray, cov:np.ndarray) -> float:
    '''
    Returns the value of the n-dimensional Gaussian function with mean and covariance matrix

    Parameters
    ----------
    x : np array of shape (n,)
        The point at which to evaluate the Gaussian function
    mean : np array of shape (n,)
        The mean of the Gaussian function
    cov : np array of shape (n,n)
        The covariance matrix of the Gaussian function

    Returns
    -------
    float
        The value of the Gaussian function at point x
    '''
    n = len(mean)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    norm = 1/np.sqrt((2*np.pi)**n * det)
    return norm * np.exp(-0.5 * np.dot(np.dot((x-mean), inv), (x-mean).T))

def dissimFunct(Dk,x,gam):
    '''
    Returns the value of the objective function of the QP problem for the given data and parameters

    Parameters
    ----------
    Dk : np array of shape (M,N)
        The training data matrix
    x : np array of shape (N,)
        The extended feature vector
    gam : float
        The value of the gamma parameter

    Returns
    -------
    float
        The value of the objective function of the QP problem

    '''
    # Get P matrix as a square matrix of size 2*N with a diagonal matrix of size N and value 2 in the upper left corner
    P = np.zeros([2*len(Dk), 2*len(Dk)])
    P[:len(Dk), :len(Dk)] = np.eye(len(Dk))*2

    # Matrix P as a sparse matrix
    P = sp.csc_matrix(P)

    # Get q vector of size 2*N+1 with gamma values in the last N elements
    q = np.zeros([2*len(Dk)])
    q[len(Dk):2*len(Dk)] = gam

    # Get G matrix of size 2*N x 2*N with four identity matrices of size N. All identity matrices have negative sign except the upper left corner
    G = np.zeros([2*len(Dk), 2*len(Dk)])
    G[:len(Dk), :len(Dk)] = np.eye(len(Dk))
    G[len(Dk):2*len(Dk), len(Dk):2*len(Dk)] = -np.eye(len(Dk))
    G[len(Dk):2*len(Dk), :len(Dk)] = -np.eye(len(Dk))
    G[:len(Dk), len(Dk):2*len(Dk)] = -np.eye(len(Dk))

    # G as a sparse matrix
    G = sp.csc_matrix(G)

    # Get h vector of size 2*N with zero values
    h = np.zeros([2*len(Dk)])

    # Get A matrix of size M+1 x 2*N with the training data matrix and a row of zeros and a 1 in the last column
    A = np.zeros([len(Dk[0])+1, 2*len(Dk)])
    A[:len(Dk[0]), :len(Dk)] = np.array(Dk).T
    A[len(Dk[0]), :len(Dk)] = 1

    # A as a sparse matrix
    A = sp.csc_matrix(A)
    
    # Create an extended feature vector with a 1
    b = np.hstack([x, 1])

    # Get T that minimizes the QP function using OSQP
    T = qp.solve_qp(P, q.T, G, h, A, b, solver='osqp')

    #check if T is None
    if T is None:
        # set the value of the objective function to infinity
        jx = np.inf
    else:
        # calculate the value of the objective function
        jx = 0.5*np.dot(T, np.dot(P.toarray(), T.T)) + np.dot(q.T, T)
    return jx

#fix the seed
np.random.seed(42)

############################################ PARAMETER SETTING ############################################

# number of points for the random dataset
n = 100

# number of points for Importance Sampling
nIS = 100

# gamma parameter
gam = 1

# c parameter
c = n/8

# grid resolution in points per axis
res = 100

# grid limits
xlim = [-2, 2]
ylim = [-2, 2]

############################################ PROCESSING ############################################

# create a random 2x2 matrix
R= np.random.rand(2,2)

# create an identity matrix
#R= np.eye(2)

# Select a 2D dataset with a uniform distribution inside a centered square of side 2
data = np.random.rand(n, 2) * 2 - 1

# Apply the random 2D transform to the data
dataT = np.dot(data, R)

# get an approximation of the covariance matrix
cov = np.cov(dataT, rowvar=False)

#get an approximation of the mean
mean = np.mean(dataT, axis=0)

# Create a grid of points
x = np.linspace(xlim[0], xlim[1], res)
y = np.linspace(ylim[0], ylim[1], res)
X, Y = np.meshgrid(x, y)

# Compute the value of the Gaussian function at each point of the grid
Zgauss = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Zgauss[i, j] = gaussian(np.array([X[i, j], Y[i, j]]), mean, cov)

# Compute the value of the objective function at each point of the grid using dataT as the training data
jgamma = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        jgamma[i, j] = dissimFunct(dataT, np.array([X[i, j], Y[i, j]]), gam)

# Compute the value of the objective function at each point of the grid using dataT as the training data and the gamma=0
j0 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        j0[i, j] = dissimFunct(dataT, np.array([X[i, j], Y[i, j]]), 0)

# Calculate b as c*sum(jgamma)/sum(j0)
b = c*np.sum(jgamma)/np.sum(j0)

# Calculate upsilon as the equivalent covariance for the gaussian closest to the Jgamma-based probability distribution
upsilon = n/(2*b) * cov

# Create a new dataset with a normal distribution with mean and covariance matrix
dataIS = np.random.multivariate_normal(mean, upsilon, nIS)

# Compute the value of the objective function at each point of dataIS using dataT as the training data
jgammaIS = np.zeros(nIS)
for i in range(nIS):
    jgammaIS[i] = dissimFunct(dataT, dataIS[i], gam)

# Compute the value of the gaussian function at each point of dataIS using mean and upsilon
q = np.zeros(nIS)
for i in range(nIS):
    q[i] = gaussian(dataIS[i], mean, upsilon)


# Calculate exp(-c*jgammaIS)/q for each point of dataIS
exp = np.divide(np.exp(-c*jgammaIS),q)

# Calculate the inverse of F as 1/nIS * sum(exp(-c*jgammaIS)/q)
Finv = 1/nIS * np.sum(exp)

# Calculate the value of F as 1/Finv
F = 1/Finv

# Calculate P as F*exp(-c*j) for each point of the grid
P = F*np.exp(-c*jgamma)

# Calculate the dissimilarity function at each point of dataT using dataT as the training data
jgammaData = np.zeros(n)
for i in range(n):
    jgammaData[i] = dissimFunct(dataT, dataT[i], gam)

# Calculate Pdata as F*exp(-c*j) for each point of dataT
Pdata = F*np.exp(-c*jgammaData)

# Calculate the likelihood ratio as the multiplication of all the values of Pdata
LR = np.prod(Pdata)

# Print the value of the likelihood ratio
print(LR)

############################################ PLOTTING ############################################
        
# create a figure with 4 subplots
fig, axs = plt.subplots(2, 2)

# Plot P in the first subplot
cax1=axs[0, 0].contourf(X, Y, P, 20, cmap='RdGy')
# Add a colorbar to the first subplot
fig.colorbar(cax1, ax=axs[0, 0])

# Add the transformed data points to the plot
axs[0, 0].scatter(dataT[:, 0], dataT[:, 1], c='black', s=10, marker='x')
axs[0, 0].axis('equal')

# Limit the axis to the grid limits
axs[0, 0].set_xlim(xlim)
axs[0, 0].set_ylim(ylim)

# Add a title to the first subplot
axs[0, 0].set_title('Disimilarity function with LR = ' + str(LR))
        
# Plot the Gaussian function in the second subplot
cax2=axs[0, 1].contourf(X, Y, Zgauss, 20, cmap='RdGy')
# Add a colorbar to the second subplot
fig.colorbar(cax2, ax=axs[0, 1])

# Add the transformed data points to the plot
axs[0, 1].scatter(dataT[:, 0], dataT[:, 1], c='black', s=10, marker='x')
axs[0, 1].axis('equal')

# Limit the axis to the grid limits
axs[0, 1].set_xlim(xlim)
axs[0, 1].set_ylim(ylim)

# Add a title to the second subplot
axs[0, 1].set_title('Gaussian function')

# Scatter plot of the original data in the third subplot
axs[1, 0].scatter(data[:, 0], data[:, 1])
axs[1, 0].axis('equal')

# Add a title to the third subplot
axs[1, 0].set_title('Original data')


# Scatter plot of the transformed data in the fourth subplot
axs[1, 1].scatter(dataT[:, 0], dataT[:, 1])
axs[1, 1].axis('equal')

# Add a title to the fourth subplot
axs[1, 1].set_title('Transformed data')
filename="Plots\\LRresults\\gamma" + "{:.2f}".format(gam).replace(".","") + "c" + "{:.2f}".format(c).replace(".","") + ".png"
plt.savefig(filename)