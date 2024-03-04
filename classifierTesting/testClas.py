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

class dissimDistribution:
    '''Implements a class to manage a dissimilarity-function-based distribution'''
    def __init__(self, dataT, gam, gamma2, c, F=None, nIS=None, nB=None):
        '''
        Initializes the dissimilarity-function-based distribution

        Parameters
        ----------
        dataT : np array of shape (M,N)
            The training data matrix
        gam : float
            The value of the gamma parameter. Equivalent gamma will be gam/gamma2
        gamma2 : float
            The value of the gamma2 parameter. Equivalent gamma will be gam/gamma2
        c : float
            The value of the c constant
        F : float
            The value of the F constant. If None, it will be calculated
        '''

        # store the training data
        self.dataT = dataT
        # store the gamma parameter
        self.gam = gam
        # store the gamma2 parameter
        self.gamma2 = gamma2
        # store the c constant
        self.c = c

        # get the dimension of the data
        self.d = len(dataT[0])
        self.N = len(dataT)

        # if nIS is None
        if nIS is None:
            # calculate nIS
            self.nIS = 100**self.d
        else:
            # store nIS
            self.nIS = nIS
        
        # if nB is None
        if nB is None:
            # calculate nB
            self.nB = 100**self.d
        else:
            # store nB
            self.nB = nB

        # get an approximation of the covariance matrix
        self.cov = np.cov(dataT, rowvar=False)

        #get an approximation of the mean
        self.mean = np.mean(dataT, axis=0)
        
        # if F is None
        if F is None:
            # calculate F
            self.F = self.calculateF()
        else:
            # store F
            self.F = F


    def calculateF(self):
        '''
        Calculates the value of the F constant

        Returns
        -------
        float
            The value of the F constant
        '''
        # generate random points using the normal distribution to calculate b
        datab = np.random.multivariate_normal(self.mean, self.cov, self.nB)

        # Compute the value of the objective function at each point of the selected points using dataT as the training data
        jgammab = np.zeros(self.nB)
        for i in range(self.nB):
            jgammab[i] = dissimFunct(self.dataT, datab[i], self.gam, self.gamma2)

        # Compute the value of the objective function at each point of the selected points using dataT as the training data
        j0b = np.zeros(self.nB)
        for i in range(self.nB):
            j0b[i] = dissimFunct(self.dataT, datab[i], 0, 1)

        # Calculate b as c*sum(jgamma)/sum(j0)
        self.b = self.c*np.sum(jgammab)/np.sum(j0b)

        # Calculate upsilon as the equivalent covariance for the gaussian closest to the Jgamma-based probability distribution
        upsilon = self.N/(2*self.b) * self.cov

        # Create a new dataset with a normal distribution with mean and covariance matrix
        dataIS = np.random.multivariate_normal(self.mean, upsilon, self.nIS)

        # Compute the value of the objective function at each point of dataIS using dataT as the training data
        jgammaIS = np.zeros(self.nIS)
        for i in range(self.nIS):
            jgammaIS[i] = dissimFunct(self.dataT, dataIS[i], self.gam, self.gamma2)

        # Compute the value of the gaussian function at each point of dataIS using mean and upsilon
        q = np.zeros(self.nIS)
        for i in range(self.nIS):
            q[i] = gaussian(dataIS[i], self.mean, upsilon)

        # Calculate exp(-c*jgammaIS)/q for each point of dataIS
        exp = np.divide(np.exp(-self.c*jgammaIS),q)

        # Calculate the inverse of F as 1/nIS * sum(exp(-c*jgammaIS)/q)
        Finv = 1/self.nIS * np.sum(exp)

        # Calculate the value of F as 1/Finv
        F = 1/Finv

        return F
    
    def computeP(self, p):
        '''
        Computes the value of the dissimilarity function-based distribution at a point

        Parameters
        ----------
        p : np array of shape (N,)
            The point at which to compute the value of the distribution

        Returns
        -------
        float
            The value of the dissimilarity function-based distribution at the point
        '''

        # Compute the value of the dissimilarity function at the point
        jx = dissimFunct(self.dataT, p, self.gam, self.gamma2)

        # Compute the value of the distribution at the point
        return self.F*np.exp(-self.c*jx)
    
    def computePlist(self, points):
        '''
        Computes the value of the dissimilarity function-based distribution at a list of points

        Parameters
        ----------
        points : np array of shape (n,N)
            The points at which to compute the value of the distribution

        Returns
        -------
        np array of shape (n,)
            The value of the dissimilarity function-based distribution at the points
        '''

        # Create an empty list to store the values of the distribution
        P = []

        # for each point in the list
        for p in points:
            # Compute the value of the distribution at the point
            P.append(self.computeP(p))

        # convert the list to a numpy array
        return np.array(P)
    
    def sample(self, n):
        '''
        Samples from the dissimilarity-function-based distribution

        Parameters
        ----------
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
            point = np.random.multivariate_normal(self.mean, self.cov)

            # Generate a random number between 0 and 1
            r = np.random.rand()

            # get the probability of the point using the Gaussian distribution
            q = gaussian(point, self.mean, self.cov)

            # get the probability of the point using the dissimilarity function
            p = self.computeP(point)

            # if the random number is less than the ratio of the probabilities
            if r < p/q:
                # add the point to the list
                points.append(point)
        # convert the list to a numpy array
        return np.array(points)
    
    def likelyhoodRatio(self, data):
        '''
        Computes the value of the likelihood ratio of the dissimilarity-function-based distribution

        Parameters
        ----------
        data : np array of shape (M,N)
            The data to compute the likelihood ratio

        Returns
        -------
        float
            The value of the likelihood ratio
        '''

        # Compute the value of the distribution at the data
        Pdata = self.computePlist(data)

        # Compute the value of the likelihood ratio
        return np.prod(Pdata)

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

# Create a grid of points
x = np.linspace(xlim[0], xlim[1], res)
y = np.linspace(ylim[0], ylim[1], res)
X, Y = np.meshgrid(x, y)

# create a dissimilarity distribution object to approximate the dataT distribution
dissimDist = dissimDistribution(dataT, gam, gamma2, c)

# Get the value of F
F = dissimDist.F

# get a flattened list of the points in the grid
points = np.array([X.flatten(), Y.flatten()]).T

# Apply the computePlist method to the grid
P = dissimDist.computePlist(points.tolist())

# Reshape the result to the shape of the grid
P = P.reshape(res, res)

# Calculate the value of the Gaussian function over the grid
Zgauss = []
for p in points:
    Zgauss.append(gaussian(p, dissimDist.mean, dissimDist.cov))
Zgauss = np.array(Zgauss).reshape(res, res)

# Calculate the integral of P
intP = np.sum(P)*(x[1]-x[0])*(y[1]-y[0])/(res**2)

# Calculate the likelihood ratio over dataT
LR = dissimDist.likelyhoodRatio(dataT)

# Get the value of b
b = dissimDist.b

# Print the value of F
print('F: ', F)

# Print the value of the integral of P
print('intP: ', intP)

# Print the value of the likelihood ratio
print('LR: ', LR)

# Print the value of c
print('c: ', c)

# Print the value of b
print('b: ', b)

# generate a sample of points using the distribution
dataSR = dissimDist.sample(nSR)

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