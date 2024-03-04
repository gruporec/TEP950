import numpy as np
import scipy.sparse as sp
import qpsolvers as qp

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

def sampleReject(dataT,F,c,gam,gamma2,mean,cov,n):
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
        nIS : int
            The number of points to use for Importance Sampling. If None, it will be stimated from the dimension of the data
        nB : int
            The number of points to use for b calculation. If None, it will be stimated from the dimension of the data
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
    
class dissimClas:
    '''Implements a bayesian classifier based on the dissimilarity-function-based distribution'''
    
    def __init__(self, X, Y, gammak, ck, Fk=None, Pk=None, nISk=None, nBk=None):
        '''
        Initializes the dissimilarity-function-based classifier

        Parameters
        ----------
        X : np array of shape (M,N)
            The training data matrix
        Y : np array of shape (M,)
            Numeric labels of the training data, starting from 0
        gammak : np array of shape (K,) or float
            The value of the gamma parameter for each class. If a float, it will be used for all classes
        ck : np array of shape (K,) or float
            The value of the c constant for each class. If a float, it will be used for all classes
        Fk : np array of shape (K,) or float
            The value of the F constant for each class. If None, it will be calculated. If a float, it will be used for all classes
        Pk : np array of shape (K,)
            The base probability of each class. If None, it will be calculated. If a float, it will be used for all classes
        nISk : np array of shape (K,) or float
            The number of points to use for Importance Sampling for each class. If None, it will be stimated from the dimension of the data. If a float, it will be used for all classes
        nBk : np array of shape (K,) or float
            The number of points to use for b calculation for each class. If None, it will be stimated from the dimension of the data. If a float, it will be used for all classes
        '''
        # store the training data
        self.X = X
        # store the labels of the training data
        self.Y = Y

        # get the dimension of the data
        self.d = len(X[0])

        # get the number of classes from the labels
        self.K = len(np.unique(Y))

        # divide the training data into classes
        self.Xk = []
        for k in range(self.K):
            self.Xk.append(self.X[self.Y==k])

        # if Pk is None
        if Pk is None:
            # calculate Pk
            self.Pk = np.array([len(self.Xk[k])/len(self.X) for k in range(self.K)])

        # if gammak is a float
        if isinstance(gammak, (int, float)):
            # store gammak as an array of the same value for each class
            self.gammak = np.array([gammak]*self.K)
        else:
            # store gammak as the array
            self.gammak = gammak

        # if ck is a float
        if isinstance(ck, (int, float)):
            # store ck as an array of the same value for each class
            self.ck = np.array([ck]*self.K)
        else:
            # store ck as the array
            self.ck = ck

        # if nISk is None
        if nISk is None:
            # calculate nISk
            self.nISk = 100**self.d*np.ones(self.K)
        # if nISk is a float
        elif isinstance(nISk, (int, float)):
            # store nISk as an array of the same value for each class
            self.nISk = np.array([nISk]*self.K)
        else:
            # store nISk as the array
            self.nISk = nISk

        # if nBk is None
        if nBk is None:
            # calculate nBk
            self.nBk = 100**self.d*np.ones(self.K)
        # if nBk is a float
        elif isinstance(nBk, (int, float)):
            # store nBk as an array of the same value for each class
            self.nBk = np.array([nBk]*self.K)
        else:
            # store nBk as the array
            self.nBk = nBk

        # create a list of dissimilarity-function-based distributions for each class
        self.dissimDist = []
        for k in range(self.K):
            self.dissimDist.append(dissimDistribution(self.Xk[k], self.gammak[k], 1, self.ck[k], Fk, self.nISk[k], self.nBk[k]))
        
    def getClassProbabilities(self, x):
        '''
        Computes the probability of each class for a given point

        Parameters
        ----------
        x : np array of shape (N,)
            The point at which to compute the probability of each class

        Returns
        -------
        np array of shape (K,)
            The probability of each class
        '''

        # create an empty list to store the probabilities
        P = []

        # for each class
        for k in range(self.K):
            # compute the value of the dissimilarity-function-based distribution at the point
            Pdissim = self.dissimDist[k].computeP(x)
            # compute the probability of the class
            P.append(Pdissim*self.Pk[k])
        # normalize the probabilities to sum to 1
        return np.array(P)/np.sum(P)
    
    def classify(self, x):
        '''
        Classifies a point

        Parameters
        ----------
        x : np array of shape (N,)
            The point to classify

        Returns
        -------
        int
            The class of the point
        '''

        # get the probabilities of the point
        P = self.getClassProbabilities(x)

        # return the class with the highest probability
        return np.argmax(P)