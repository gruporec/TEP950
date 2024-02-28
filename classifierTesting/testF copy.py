import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import multiprocessing as mp
import tqdm

import qpsolvers
import scipy.sparse as sp

# #add the path to the lib folder to the system path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# # import the isadoralib library
# import isadoralib as isl


def getJ(arg):
    '''Updates the matrices P, q, G, h and A for the training data Dk and the parameter gamma.'''
    Dk,x,gam, initval=arg

    # Get P matrix as a square matrix of size 2*N with a diagonal matrix of size N and value 2 in the upper left corner
    P = np.zeros([2*len(Dk), 2*len(Dk)])
    P[:len(Dk), :len(Dk)] = np.eye(len(Dk))*2
    # Matrix P as a sparse matrix
    #P = sp.csc_matrix(P)

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
    #G = sp.csc_matrix(G)

    # Get h vector of size 2*N with zero values
    h = np.zeros([2*len(Dk)])

    # Get A matrix of size M+1 x 2*N with the training data matrix and a row of zeros and a 1 in the last column
    A = np.zeros([len(Dk[0])+1, 2*len(Dk)])
    A[:len(Dk[0]), :len(Dk)] = np.array(Dk).T
    A[len(Dk[0]), :len(Dk)] = 1
    # A as a sparse matrix
    #A = sp.csc_matrix(A)
    
    # Create an extended feature vector with a 1
    b = np.hstack([x, 1])

    # Get T that minimizes the QP function using OSQP
    T = qpsolvers.solve_qp(P, q.T, G, h, A, b, solver='piqp', initvals=initval)

    #check if T is None
    if T is None:
        # set the value of the objective function to infinity
        jx = 50000000
    else:
        # calculate the value of the objective function
        jx = 0.5*np.dot(T, np.dot(P, T.T)) + np.dot(q.T, T)
    if jx<=0:
        print(T)
    return jx

def getT(arg):
    '''Updates the matrices P, q, G, h and A for the training data Dk and the parameter gamma.'''
    Dk,x,gam=arg

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
    T = qpsolvers.solve_qp(P, q.T, G, h, A, b, solver='clarabel')

    return T


# if name is main
if __name__ == '__main__':
    # Parameters
    mu = 0
    variance = 1
    samp=100
    monteCarloPoints=100

    # get ck as Nk/2, Nk=number of elements in data
    ck = samp/2
    #ck=1
    
    gamm=100

    # get upsilon as Nk*variance/(2*ck)
    upsilon = samp*variance/(2*ck)

    sigma = math.sqrt(variance)


    # Generate normal distribution data
    np.random.seed(0)
    data = np.random.normal(mu, sigma, samp)

    # Create a series of points up to 3 standard deviations or upsilons from the mean; whichever is larger
    largevar = max(3*sigma, 3*math.sqrt(upsilon))
    x = np.linspace(mu - largevar, mu + largevar, 100)

    # get the normal distribution
    y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    #get a random array for the monte carlo simulation. use an uniform distribution in the interval mu-3*sigma, mu+3*sigma
    monteCarlo = np.random.uniform(mu-3*sigma, mu+3*sigma, monteCarloPoints)

    # get a random array for the importance sampling. use a gaussian distribution with mean mu and variance sigma
    importanceSampling = np.random.normal(mu, sigma, monteCarloPoints)

    #order the array points so it does not mess up the plot
    monteCarlo.sort()
    importanceSampling.sort()

    # express data as a list of lists
    datal = data.tolist()
    datal = [[i] for i in datal]

    # get the values of the gaussian distribution at the points of the importance sampling
    yIS = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((importanceSampling - mu) / sigma) ** 2)

    # obtain J for 0 to use as initial value
    initJ = getT((datal, [0], 0))
    initJgamma = getT((datal, [0], gamm))

    # use multi-threading to calculate J0 for every point in the monte carlo array    
    with mp.Pool(mp.cpu_count()) as pool:
    #with mp.Pool(1) as pool:
        J0 = list(tqdm.tqdm(pool.imap(getJ, [(datal, i, 0, initJ) for i in monteCarlo]), total=len(monteCarlo)))
    
    # use multi-threading to calculate JgammaMC for every point in the monte carlo array
    with mp.Pool(mp.cpu_count()) as pool:
        JgammaMC = list(tqdm.tqdm(pool.imap(getJ, [(datal, i, gamm, initJgamma) for i in monteCarlo]), total=len(monteCarlo)))
    
    # J0 to numpy array
    J0 = np.array(J0)
    JgammaMC = np.array(JgammaMC)


    # use multi-threading to calculate Jgamma for every point in the importance sampling array
    with mp.Pool(mp.cpu_count()) as pool:
        Jgam = list(tqdm.tqdm(pool.imap(getJ, [(datal, i, gamm, initJgamma) for i in importanceSampling]), total=len(importanceSampling)))

    # # use multi-threading to calculate JIS for every point in the importance sampling array
    with mp.Pool(mp.cpu_count()) as pool:
        JIS = list(tqdm.tqdm(pool.imap(getJ, [(datal, i, 0, initJ) for i in importanceSampling]), total=len(importanceSampling)))

    # Jgamma to numpy array
    Jgam = np.array(Jgam)
    JIS = np.array(JIS)


    # get the value of Fk for the dissimilarity function classifier as \frac{e^{\frac{1}{2}}}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
    F0 = math.exp(ck/samp)/(math.pow(2*math.pi, 0.5)*math.pow(upsilon, 0.5))

    print(F0)

    # get an alternate value of F0, F1, as 1/N Sum(e^(-ck*J0(x)))*6*sigma
    F1 = np.sum(np.exp(-ck*J0))/len(J0)*6*sigma

    F1gamma = np.sum(np.exp(-ck*JgammaMC))/len(JgammaMC)*6*sigma

    F1 = 1/F1

    if F1gamma!=0:
        F1gamma = 1/F1gamma

    # get the values of np.exp(-ck*JIS)*yIS for each point in the importance sampling
    valIS = np.exp(-ck*JIS)/yIS

    # get yet another value of F0, F2, using importance sampling
    F2 = np.sum(valIS)/len(JIS)

    F2 = 1/F2

    print(F1)

    print(F2)

    # get the value of Fgamma as an importance sampling using the value of Jgamma
    # get the values of np.exp(-ck*JIS)*yIS for each point in the importance sampling
    valISgamma = np.exp(-ck*Jgam)/yIS

    print('yIS: ', yIS)
    print('Jgam: ', Jgam)
    print('ck: ', ck)
    print('exp: ', np.exp(-ck*Jgam))
    print('ValISgamma: ', valISgamma)

    # get the value of Fgamma as an importance sampling using the value of Jgamma
    Fgamma = np.sum(valISgamma)/len(Jgam)

    if Fgamma!=0:
        Fgamma = 1/Fgamma


    # get P0=F0*e^(-ck*J0)
    P0 = F0*np.exp(-ck*J0)

    # get P1=F1*e^(-ck*J0)
    P1 = F1*np.exp(-ck*J0)

    P2 = F2*np.exp(-ck*J0)

    # get Pgamma=Fgamma*e^(-ck*Jgamma)
    Pgamma = Fgamma*np.exp(-ck*Jgam)

    PF1gamma = F1gamma*np.exp(-ck*JgammaMC)

    # Plot J0
    plt.plot(monteCarlo, J0, color='black')

    # Plot JIS
    plt.plot(importanceSampling, JIS, color='red')

    # Plot Jgamma
    plt.plot(importanceSampling, Jgam, color='blue')

    # Plot JgammaMC
    plt.plot(monteCarlo, JgammaMC, color='green')

    # legend
    plt.legend(['J0(x)', 'JIS(x)', 'Jgamma(x)', 'JgammaMC(x)'])
    plt.xlabel('X')
    
    #separate figure
    plt.figure()

    # Plot P0
    plt.plot(monteCarlo, P0, color='red')

    # Plot P1
    plt.plot(monteCarlo, P1, color='blue')

    # Plot P2
    plt.plot(monteCarlo, P2, color='green')

    # Plot PF1gamma
    plt.plot(monteCarlo, PF1gamma, color='yellow')

    # Plot Pgamma
    plt.plot(importanceSampling, Pgamma, color='black')

    #plot the data
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

    # Plot the normal distribution as a dotted line
    plt.plot(x, y, color='black', linestyle='dotted')
    plt.fill_between(x, y, color='black', alpha=0.1)
    plt.fill_between(monteCarlo, P0, color='red', alpha=0.1)
    plt.fill_between(monteCarlo, P1, color='blue', alpha=0.1)
    plt.fill_between(monteCarlo, P2, color='green', alpha=0.1)
    # legend
    plt.legend(['Theorical Normal Distribution', 'Monte Carlo Uniform', 'Monte Carlo Gaussian','Gamma Uniform', 'Gamma Gaussian', 'Gaussian'])
    #plt.legend(['J0(x)', 'JIS(x)', 'Jgamma(x)', 'JgammaMC(x)', 'Theorical Normal Distribution', 'Monte Carlo Uniform', 'Monte Carlo Gaussian','Gamma Uniform', 'Gamma Gaussian', 'Gaussian'])
    plt.xlabel('X')
    plt.ylabel('Probability Density')
    plt.show()
