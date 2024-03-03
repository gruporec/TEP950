import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import qpsolvers as qp
import sys
import matplotlib.path as mpath

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

def dissimFunct(Dk,x,gam,gamma2):
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
    P[:len(Dk), :len(Dk)] = np.eye(len(Dk))*2*gamma2

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
        # undo the multiplication by gamma2
        T = T/gamma2
        # calculate the value of the objective function
        jx = 0.5*np.dot(T, np.dot(P.toarray(), T.T)) + np.dot(q.T, T)
    return jx

def sampleReject(F,c,gam,gamma2,mean,cov,n):
    '''
    Computes a sample rejection algorithm to sample from the distribution of the dissimilarity function using a Gaussian distribution as a base distribution

    Parameters
    ----------
    F : float
        The value of the F constant
    c : float
        The value of the c constant
    gam : float
        The value of the gamma parameter. Equivalent gamma will be gam/gamma2
    gamma2 : float
        The value of the gamma2 parameter. Equivalent gamma will be gam/gamma2
    mean : np array of shape (N,)
        The mean of the Gaussian function
    cov : np array of shape (N,N)
        The covariance matrix of the Gaussian function
    n : int
        The number of points to sample

    Returns
    -------
    np array of shape (n,N)
        The sampled points
    '''
    # Create an empty list to store the points
    points = []
    
    # While the list has less points than n
    while len(points) < n:
        # Generate a random point from the Gaussian distribution
        point = np.random.multivariate_normal(mean, cov)
        # Compute the value of the dissimilarity function at the point
        jx = dissimFunct(dataT, point, gam, gamma2)
        # Generate a random number between 0 and 1
        r = np.random.rand()
        
        # get the probability of the point using the Gaussian distribution
        q = gaussian(point, mean, cov)

        # get the probability of the point using the dissimilarity function
        p = F*np.exp(-c*jx)

        # if the random number is less than the ratio of the probabilities
        if r < p/q:
            # add the point to the list
            points.append(point)
    # convert the list to a numpy array
    return np.array(points)

#fix the seed
np.random.seed(42)

############################################ PARAMETER SETTING ############################################

# number of points for the random dataset
n = 100

# number of points for Importance Sampling
nIS = 10000

# number of points for b calculation
nB = 10000

# number of points to generate using the sample rejection algorithm
nSR = 200

# gamma parameter. Equivalent gamma will be gam/gamma2. gamma2 is not really working too well when it comes to getting the value of F
gam = 1
gamma2=1

#c fraction
cf=12

# c parameter
c = n/cf

# grid resolution in points per axis
res = 100

# grid limits
xlim = [-2, 2]
ylim = [-2, 2]

# Name the shape for the filename
shapeName = 'triangle'

# # define take 5 points from a circle of radius 1 1
# t = np.linspace(0, 2*np.pi, 5, endpoint=False)
# x = np.cos(t)
# y = np.sin(t)
# corners1 = np.array([x, y]).T

# # scale the shape and rotate it 360/10 degrees
# corners2 = corners1*0.5
# corners2 = np.dot(corners2, np.array([[np.cos(np.pi/5), -np.sin(np.pi/5)], [np.sin(np.pi/5), np.cos(np.pi/5)]]))

# # merge the two shapes taking points alternatively
# corners = np.zeros([10, 2])
# corners[1::2] = corners1
# corners[0::2] = corners2

# define a triangle
corners = np.array([[-1, -0.5], [1, -0.5], [0, 1*np.sqrt(3)-0.5]])

# flag to only plot the shape
onlyShape = False

# flag to only plot the points in the shape
onlyPoints = False


############################################ PROCESSING ############################################

if onlyShape:
    #plot the shape
    plt.plot(corners[:, 0], corners[:, 1], 'k-')
    # equal aspect ratio
    plt.axis('equal')
    plt.show()
    sys.exit()

# Create a path from the corners
path = mpath.Path(corners)

# create a list of points to use as dataset
dataT = []

# get the rectangle that contains the shape
x0, y0 = np.min(corners, axis=0)
x1, y1 = np.max(corners, axis=0)

# while the list has less points than n
while len(dataT) < n:
    # generate a random point inside the rectangle
    point = np.random.rand(2) * [x1-x0, y1-y0] + [x0, y0]
    # check if the point is inside the shape
    if path.contains_point(point):
        # add the point to the list
        dataT.append(point)

# convert the list to a numpy array
dataT = np.array(dataT)

if onlyPoints:
    #plot the pointsç
    plt.scatter(dataT[:, 0], dataT[:, 1])
    # equal aspect ratio
    plt.axis('equal')
    plt.show()
    sys.exit()


# get an approximation of the covariance matrix
cov = np.cov(dataT, rowvar=False)

#get an approximation of the mean
mean = np.mean(dataT, axis=0)

# Create a grid of points
x = np.linspace(xlim[0], xlim[1], res)
y = np.linspace(ylim[0], ylim[1], res)
X, Y = np.meshgrid(x, y)

# Select some random points using the normal distribution
datab = np.random.multivariate_normal(mean, cov, nB)

# Compute the value of the Gaussian function at each point of the selected points using mean and cov
gaussb = np.zeros(datab.shape[0])
for i in range(datab.shape[0]):
    gaussb[i] = gaussian(datab[i], mean, cov)

# Compute the value of the objective function at each point of the selected points using dataT as the training data
jgammab = np.zeros(datab.shape[0])
for i in range(datab.shape[0]):
    jgammab[i] = dissimFunct(dataT, datab[i], gam, gamma2)

# Compute the value of the objective function at each point of the selected points using dataT as the training data
j0b = np.zeros(datab.shape[0])
for i in range(datab.shape[0]):
    j0b[i] = dissimFunct(dataT, datab[i], 0, gamma2)

# Calculate b as c*sum(jgamma)/sum(j0)
b = c*np.sum(jgammab)/np.sum(j0b)

# Calculate upsilon as the equivalent covariance for the gaussian closest to the Jgamma-based probability distribution
upsilon = n/(2*b) * cov

# Create a new dataset with a normal distribution with mean and covariance matrix
dataIS = np.random.multivariate_normal(mean, upsilon, nIS)

# Compute the value of the objective function at each point of dataIS using dataT as the training data
jgammaIS = np.zeros(nIS)
for i in range(nIS):
    jgammaIS[i] = dissimFunct(dataT, dataIS[i], gam, gamma2)

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

print('F: ', F)

# Compute the value of the objective function at each point of the grid using dataT as the training data
jgamma = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        jgamma[i, j] = dissimFunct(dataT, np.array([X[i, j], Y[i, j]]), gam, gamma2)

# Compute the value of the Gaussian function at each point of the grid
Zgauss = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Zgauss[i, j] = gaussian(np.array([X[i, j], Y[i, j]]), mean, cov)

# Calculate P as F*exp(-c*j) for each point of the grid
P = F*np.exp(-c*jgamma)

# Calculate the dissimilarity function at each point of dataT using dataT as the training data
jgammaData = np.zeros(n)
for i in range(n):
    jgammaData[i] = dissimFunct(dataT, dataT[i], gam, gamma2)

# Calculate Pdata as F*exp(-c*j) for each point of dataT
Pdata = F*np.exp(-c*jgammaData)

# Calculate the likelihood ratio as the multiplication of all the values of Pdata
LR = np.prod(Pdata)

# integrate the values of P over the grid
intP = np.sum(P)*(xlim[1]-xlim[0])*(ylim[1]-ylim[0])/res**2

# Print the value of the integral of P
print('intP: ', intP)

# Print the value of the likelihood ratio
print('LR: ', LR)

# Print the value of c
print('c: ', c)

# Print the value of b
print('b: ', b)

# generate a sample of points using the sample rejection algorithm
dataSR = sampleReject(F, c, gam, gamma2, mean, cov, nSR)

############################################ PLOTTING ############################################
        
# create a figure with 4 subplots
fig, axs = plt.subplots(2, 2)

# Add a title to the figure
fig.suptitle('Likelihood ratio: ' + str(LR))

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
axs[0, 0].set_title('Disimilarity function')
        
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
axs[1, 0].scatter(dataT[:, 0], dataT[:, 1])
axs[1, 0].axis('equal')

# Add a title to the third subplot
axs[1, 0].set_title('Original data')


# Scatter plot of the inferred data in the fourth subplot
axs[1, 1].scatter(dataT[:, 0], dataT[:, 1], c='black', s=10, marker='x')
axs[1, 1].scatter(dataSR[:, 0], dataSR[:, 1], c='red', s=10, marker='x')
axs[1, 1].axis('equal')

# Add a title to the fourth subplot
axs[1, 1].set_title('Inferred data')
filename="Plots\\SRresults\\"+shapeName+"gamma" + "{:.2f}".format(gam/gamma2).replace(".","") + "cf" + "{:.2f}".format(cf).replace(".","") + ".png"

# Save the figure
plt.savefig(filename)