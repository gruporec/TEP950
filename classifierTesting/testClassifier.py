import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import qpsolvers as qp
import sys
import matplotlib.path as mpath
import dissimClassLib as dcl
import sklearn.discriminant_analysis as sklda
import multiprocessing as mp
import tqdm

if __name__ == '__main__':
    #fix the seed
    np.random.seed(42)

    ############################################ PARAMETER SETTING ############################################

    # number of points for the training datasets
    nT = [100,200]

    # number of points for the validation datasets
    nV = [100,200]

    # number of points for Importance Sampling
    nIS = 10000

    # number of points for b calculation
    nB = 10000

    # number of points to generate using the sample rejection algorithm
    nSR = 200

    # gamma parameter. Equivalent gamma will be gam/gamma2. gamma2 is not really working too well when it comes to getting the value of F
    gam = [0,1]
    gamma2=1

    #c fraction
    cf=[2,12]

    # c parameter = n/cf
    c = [nT[i]/cf[i] for i in range(len(nT))]

    # grid resolution in points per axis
    res = 100

    # grid limits
    xlim = [-2, 2]
    ylim = [-2, 2]

    # Name the shape for the filename
    shapeName = 'triangleAndGaussian3'

    # define a triangle
    corners = np.array([[-1, -1], [1, -1], [0, 1*np.sqrt(3)-1]])

    # define the mean and variance of the Gaussian function
    meanb = np.array([0, 1*np.sqrt(3)-1])
    varb = np.array([[0.1, 0], [0, 1]])

    # flag to only plot the shape
    onlyShape = False

    # flag to only plot the points in the shape
    onlyPoints = False

    ############################################ PROCESSING ############################################

    if onlyShape:
        #plot the shape as a closed polygon, repeat the first point to close the polygon
        plt.plot(np.append(corners[:, 0], corners[0, 0]), np.append(corners[:, 1], corners[0, 1]))
        # plot a point in the mean of the Gaussian function
        plt.scatter(meanb[0], meanb[1], c='red', s=10, marker='x')
        # plot a circle with the variance of the Gaussian function
        circle = plt.Circle(meanb, np.linalg.det(varb), color='red', fill=False)
        plt.gca().add_artist(circle)


        # plot between limits
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
        sys.exit()

    # Create a path from the corners
    path = mpath.Path(corners)

    # create a list of points to use as dataset
    dataTT = []

    # create a list of points to use as dataset
    dataTV = []

    # get the rectangle that contains the shape
    x0, y0 = np.min(corners, axis=0)
    x1, y1 = np.max(corners, axis=0)

    # while the list has less points than nT[0]
    while len(dataTT) < nT[0]:
        # generate a random point inside the rectangle
        point = np.random.rand(2) * [x1-x0, y1-y0] + [x0, y0]
        # check if the point is inside the shape
        if path.contains_point(point):
            # add the point to the list
            dataTT.append(point)

    # while the list has less points than nV[0]
    while len(dataTV) < nV[0]:
        # generate a random point inside the rectangle
        point = np.random.rand(2) * [x1-x0, y1-y0] + [x0, y0]
        # check if the point is inside the shape
        if path.contains_point(point):
            # add the point to the list
            dataTV.append(point)

    # generate a list of points using the Gaussian function
    dataGT = np.random.multivariate_normal(meanb, varb, nT[1])
    dataGV = np.random.multivariate_normal(meanb, varb, nV[1])

    # convert the list to a numpy array
    dataTT = np.array(dataTT)
    dataGT = np.array(dataGT)
    dataTV = np.array(dataTV)
    dataGV = np.array(dataGV)

    if onlyPoints:
        #plot the points
        plt.scatter(dataTT[:, 0], dataTT[:, 1])
        plt.scatter(dataGT[:, 0], dataGT[:, 1])
        # plot between limits
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        #new figure
        plt.figure()
        #plot the points
        plt.scatter(dataTV[:, 0], dataTV[:, 1])
        plt.scatter(dataGV[:, 0], dataGV[:, 1])
        # plot between limits
        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.show()
        sys.exit()

    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], res)
    y = np.linspace(ylim[0], ylim[1], res)
    X, Y = np.meshgrid(x, y)

    # Create a dataset with the points for training
    Xtrain = np.concatenate((dataTT, dataGT))

    # Create a class vector
    ytrain = np.concatenate((np.zeros(nT[0]), np.ones(nT[1])))

    # Create a dataset with the points for validation
    Xval = np.concatenate((dataTV, dataGV))

    # Create a class vector
    yval = np.concatenate((np.zeros(nV[0]), np.ones(nV[1])))

    # create a dissimilarity distribution classifier
    dissimClass = dcl.dissimClas(Xtrain, ytrain, gam, c)

    # create a qda classifier
    qda = sklda.QuadraticDiscriminantAnalysis()
    qda.fit(Xtrain, ytrain)

    # get a flattened list of the points in the grid
    points = np.array([X.flatten(), Y.flatten()]).T

    pDissim=[]
    # Get the class probabilities for each point in the grid using the dissimilarity distribution classifier ussing multiprocessing
    for i in range(len(points)):
        pDissim.append(dissimClass.getClassProbabilities(points[i])[0])
    pDissim = np.array(pDissim)

    # Get the class probabilities for each point in the grid using the qda classifier
    pQda = qda.predict_proba(points)

    # Implement a custom bayesian classifier
    # get the prior probabilities
    prior = [nT[0]/(nT[0]+nT[1]), nT[1]/(nT[0]+nT[1])]

    # calculate class 0 likelihood for each point in the triangle. class 0 is an uniform distribution with the shape of the triangle. The area of the triangle is 1/2*base*height = 1/2*2*sqrt(3) = sqrt(3)
    # check if the point is inside the triangle
    pBayes0 = []
    for i in range(len(points)):
        if path.contains_point(points[i]):
            # if the point is inside the triangle, calculate the probability
            pBayes0.append(1/(np.sqrt(3)))
        else:
            # if the point is outside the triangle, the probability is 0
            pBayes0.append(0)

    # calculate class 1 likelihood for each point in the grid. class 1 is a Gaussian function with meanb and varb
    pBayes1 = []
    for i in range(len(points)):
        pBayes1.append(np.exp(-0.5*np.dot(np.dot((points[i]-meanb), np.linalg.inv(varb)), (points[i]-meanb)))/(2*np.pi*np.sqrt(np.linalg.det(varb))))

    # multiply the prior probabilities by the likelihoods
    pBayes0 = np.array(pBayes0)*prior[0]
    pBayes1 = np.array(pBayes1)*prior[1]

    # get the total probability for each point
    pBayesTotal = pBayes0+pBayes1

    # calculate the posterior probabilities
    pBayes0 = pBayes0/pBayesTotal

    # Reshape the result to the shape of the grid
    pDissim = pDissim.reshape(res, res)
    pQda = pQda[:, 0].reshape(res, res)
    pBayes0 = pBayes0.reshape(res, res)

    # classify the validation data using the dissimilarity distribution classifier
    yPredDissim = []
    for i in range(len(Xval)):
        yPredDissim.append(dissimClass.classify(Xval[i]))
    yPredDissim = np.array(yPredDissim)

    # obtain the confusion matrix
    # get the number of classes
    nClasses = len(np.unique(yval))
    # create a confusion matrix
    confMatDissim = np.zeros((nClasses, nClasses))
    # for each pair of true and predicted class
    for i in range(nClasses):
        for j in range(nClasses):
            # calculate the number of points in the validation set that are of the true class i and were predicted as class j
            confMatDissim[i, j] = np.sum(np.logical_and(yval == i, yPredDissim == j))

    # print the confusion matrix
    print('Confusion matrix for the dissimilarity distribution classifier')
    print(confMatDissim)

    # classify the validation data using the qda classifier
    yPredQda = qda.predict(Xval)

    # obtain the confusion matrix
    # create a confusion matrix
    confMatQda = np.zeros((nClasses, nClasses))
    # for each pair of true and predicted class
    for i in range(nClasses):
        for j in range(nClasses):
            # calculate the number of points in the validation set that are of the true class i and were predicted as class j
            confMatQda[i, j] = np.sum(np.logical_and(yval == i, yPredQda == j))

    # print the confusion matrix
    print('Confusion matrix for the qda classifier')
    print(confMatQda)

    # classify the validation data using the custom bayesian classifier
    # get the probability of class 0 for each point in the validation set by checking if it's inside the triangle
    pBayes0Val = []
    for i in range(len(Xval)):
        if path.contains_point(Xval[i]):
            pBayes0Val.append(1)
        else:
            pBayes0Val.append(0)

    # get the probability of class 1 for each point in the validation set using the Gaussian function
    pBayes1Val = []
    for i in range(len(Xval)):
        pBayes1Val.append(np.exp(-0.5*np.dot(np.dot((Xval[i]-meanb), np.linalg.inv(varb)), (Xval[i]-meanb)))/(2*np.pi*np.sqrt(np.linalg.det(varb))))

    # multiply the prior probabilities by the likelihoods
    pBayes0Val = np.array(pBayes0Val)*prior[0]
    pBayes1Val = np.array(pBayes1Val)*prior[1]

    # get the total probability for each point
    pBayesTotalVal = pBayes0Val+pBayes1Val

    # calculate the posterior probabilities
    pBayes0Val = pBayes0Val/pBayesTotalVal

    # classify the points as class 0 if the probability of class 0 is greater than the probability of class 1, and 1 otherwise
    yPredBayes = np.zeros(len(Xval))
    yPredBayes[pBayes0Val<0.5] = 1

    # obtain the confusion matrix
    # create a confusion matrix
    confMatBayes = np.zeros((nClasses, nClasses))
    # for each pair of true and predicted class
    for i in range(nClasses):
        for j in range(nClasses):
            # calculate the number of points in the validation set that are of the true class i and were predicted as class j
            confMatBayes[i, j] = np.sum(np.logical_and(yval == i, yPredBayes == j))

    # print the confusion matrix
    print('Confusion matrix for the custom bayesian classifier')
    print(confMatBayes)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2)

    # plot the class probabilities of the dissimilarity classifier and add a color bar
    cax0=axs[0, 0].contourf(X, Y, pDissim, levels=20, cmap='RdYlBu')
    fig.colorbar(cax0, ax=axs[0, 0])

    # add a title
    axs[0, 0].set_title('Dissimilarity Distribution')

    # equal aspect ratio
    axs[0, 0].axis('equal')

    # plot the class probabilities of the qda classifier and add a color bar
    cax1=axs[0, 1].contourf(X, Y, pQda, levels=20, cmap='RdYlBu')
    fig.colorbar(cax1, ax=axs[0, 1])

    # add a title
    axs[0, 1].set_title('QDA')

    # equal aspect ratio
    axs[0, 1].axis('equal')

    # plot the points
    axs[1, 0].scatter(dataTT[:, 0], dataTT[:, 1], c='green', s=10, marker='x')
    axs[1, 0].scatter(dataGT[:, 0], dataGT[:, 1], c='orange', s=10, marker='x')

    # add a title
    axs[1, 0].set_title('Training data')

    # equal aspect ratio
    axs[1, 0].axis('equal')

    # plot the class probabilities of the custom bayesian classifier and add a color bar
    cax2=axs[1, 1].contourf(X, Y, pBayes0, levels=20, cmap='RdYlBu')
    fig.colorbar(cax2, ax=axs[1, 1])

    # add a title
    axs[1, 1].set_title('Distribution-based Bayesian Classifier')

    # equal aspect ratio
    axs[1, 1].axis('equal')

    # generate a file name
    filename= 'Plots/disimclasif/'+shapeName+'_n'+str(nT[0])+'_'+str(nT[1])+'_nIS'+str(nIS)+'_nSR'+str(nSR)+'_gam'+str(gam[0])+'_'+str(gam[1])+'_cf'+str(cf[0])+'_'+str(cf[1])

    # save the figure
    plt.savefig(filename+'.png')

    # close the figure
    plt.close()

    # Repeat the plot but use a black and white color scale
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2)

    # plot the class probabilities of the dissimilarity classifier and add a color bar
    cax0=axs[0, 0].contourf(X, Y, pDissim, levels=20, cmap='Greys')
    fig.colorbar(cax0, ax=axs[0, 0])

    # add a title
    axs[0, 0].set_title('Dissimilarity Distribution')

    # equal aspect ratio
    axs[0, 0].axis('equal')

    # plot the class probabilities of the qda classifier and add a color bar
    cax1=axs[0, 1].contourf(X, Y, pQda, levels=20, cmap='Greys')
    fig.colorbar(cax1, ax=axs[0, 1])

    # add a title
    axs[0, 1].set_title('QDA')

    # equal aspect ratio
    axs[0, 1].axis('equal')

    # plot the points
    axs[1, 0].scatter(dataTT[:, 0], dataTT[:, 1], c='green', s=10, marker='x')
    axs[1, 0].scatter(dataGT[:, 0], dataGT[:, 1], c='orange', s=10, marker='x')

    # add a title
    axs[1, 0].set_title('Training data')

    # equal aspect ratio
    axs[1, 0].axis('equal')

    # plot the class probabilities of the custom bayesian classifier and add a color bar
    cax2=axs[1, 1].contourf(X, Y, pBayes0, levels=20, cmap='Greys')
    fig.colorbar(cax2, ax=axs[1, 1])

    # add a title
    axs[1, 1].set_title('Distribution-based Bayesian Classifier')

    # equal aspect ratio
    axs[1, 1].axis('equal')

    # save the figure
    plt.savefig(filename+'_bw.png')

    # obtain a vector with the predicted class, equal to 0 if the probability of class 0 is greater than the probability of class 1, and 1 otherwise
    yPredDissim = np.zeros(len(points))
    yPredDissim[pDissim.flatten()<0.5] = 1

    yPredQda = np.zeros(len(points))
    yPredQda[pQda.flatten()<0.5] = 1

    yPredBayes = np.zeros(len(points))
    yPredBayes[pBayes0.flatten()<0.5] = 1

    # shape the vector to the shape of the grid
    yPredDissim = yPredDissim.reshape(res, res)
    yPredQda = yPredQda.reshape(res, res)
    yPredBayes = yPredBayes.reshape(res, res)

    # find the points in the grid that are a border between the classes
    borderDissim = np.zeros((res, res))
    borderQda = np.zeros((res, res))
    borderBayes = np.zeros((res, res))

    # for each point in the grid
    for i in range(1, res-1):
        for j in range(1, res-1):
            # if the point is a border, set the value to 1
            if yPredDissim[i, j] != yPredDissim[i-1, j] or yPredDissim[i, j] != yPredDissim[i+1, j] or yPredDissim[i, j] != yPredDissim[i, j-1] or yPredDissim[i, j] != yPredDissim[i, j+1]:
                borderDissim[i, j] = 1
            if yPredQda[i, j] != yPredQda[i-1, j] or yPredQda[i, j] != yPredQda[i+1, j] or yPredQda[i, j] != yPredQda[i, j-1] or yPredQda[i, j] != yPredQda[i, j+1]:
                borderQda[i, j] = 1
            if yPredBayes[i, j] != yPredBayes[i-1, j] or yPredBayes[i, j] != yPredBayes[i+1, j] or yPredBayes[i, j] != yPredBayes[i, j-1] or yPredBayes[i, j] != yPredBayes[i, j+1]:
                borderBayes[i, j] = 1

    # plot the borders as an image
    # create a plot with 4 subplots
    fig, axs = plt.subplots(2, 2)

    # plot the borders of the dissimilarity classifier as an image over the grid positions
    axs[0, 0].imshow(borderDissim, cmap='Greys', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')

    # add a title
    axs[0, 0].set_title('Dissimilarity Distribution')

    # plot the borders of the qda classifier
    axs[0, 1].imshow(borderQda, cmap='Greys', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')

    # add a title
    axs[0, 1].set_title('QDA')

    # plot the training data
    axs[1, 0].scatter(dataTT[:, 0], dataTT[:, 1], c='k', s=10, marker='+')
    axs[1, 0].scatter(dataGT[:, 0], dataGT[:, 1], c='k', s=10, marker='x')

    # add a title
    axs[1, 0].set_title('Training data')


    # plot the borders of the custom bayesian classifier
    axs[1, 1].imshow(borderBayes, cmap='Greys', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')

    # add a title
    axs[1, 1].set_title('Distribution-based Bayesian Classifier')

    # save the figure
    plt.savefig(filename+'_borders.png')

    # close the figure
    plt.close()