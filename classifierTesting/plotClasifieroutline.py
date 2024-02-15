import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis as sklda
import sklearn.metrics as skmetrics
import sklearn.decomposition as skdecomp
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import seaborn as sns
import multiprocessing as mp
import tqdm
import sys
import os
#add the path to the lib folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import the isadoralib library
import isadoralib as isl


# create a function to apply the classifier to the training and test data in parallel
def classify_probs(arg):
    (x,clf)=arg
    ret=clf.classifyProbs(x)
    return ret


if __name__ == '__main__':

    sns.set(rc={'figure.figsize':(11.7,8.27)})

    #use agg when intended for saving the image and not for showing it
    #plt.switch_backend('agg')

    # ---CONFIGURATION---

    #dataset file path
    datasetpath=os.path.dirname(__file__)+"\\dbv2\\"

    # grid step
    gridstep=0.01

    #margin for the plot
    margin=0.1

    #accuracy margin
    acc_margin=0.1

    # clasifier types to use. Available types: "lda", "qda", "krigingfun","kriginglam"
    # "lda" and "qda" are the linear and quadratic discriminant analysis from sklearn
    # "krigingfun", "kriginglam" and "krigingqda" are the kriging based classifiers from isadoralib
    clasifs=["dissimilarityWckCal","dissimilarityWck", "dissimilarity","qda"]
    clasifs=["dissimilarityWckCal","dissimilarityWck", "dissimilarity"]
    clasifs=["dissimilarityWckCalHC"]

    # gamma for kriging 
    kr_gamma=1

    fileprefix=""

    # calibration/training split
    train_split=0.5

    #databases to plot
    files=[3]

    # ---END OF CONFIGURATION---
    for file in files:
        dataset=datasetpath+str(file)+".csv"
        # load the database taking into account that there is no index
        db=pd.read_csv(dataset,index_col=False)

        # split the database into the data and the labels
        Xtrain=db[["a","b"]].to_numpy()
        Ytrain=db["Y"].to_numpy()
        
        # get the limits in X
        xmin=np.min(Xtrain[:,0])
        xmax=np.max(Xtrain[:,0])
        ymin=np.min(Xtrain[:,1])
        ymax=np.max(Xtrain[:,1])

        # Calculate the limits of the plot adding a proportional margin
        xmin=(xmin-margin*(xmax-xmin))
        xmax=(xmax+margin*(xmax-xmin))
        ymin=(ymin-margin*(ymax-ymin))
        ymax=(ymax+margin*(ymax-ymin))

        # calculate the steps in x and y
        xstep=(xmax-xmin)*gridstep
        ystep=(ymax-ymin)*gridstep

        # create a grid of points with the limits and the grid step
        x=np.arange(xmin,xmax,xstep)
        y=np.arange(ymin,ymax,ystep)
        xx,yy=np.meshgrid(x,y)

        # create a list with the grid points
        Xtest=np.c_[xx.ravel(),yy.ravel()]


        for clasif in clasifs:
            # select the classifier according to clasif
            match clasif:
                # case "lda":
                #     clf=sklda.LinearDiscriminantAnalysis()

                #     # train the classifier
                #     clf.fit(Xtrain,Ytrain)

                #     # apply the classifier to the training and test data
                #     Ytrain_pred=clf.predict(Xtrain)
                #     Ytest_pred=clf.predict(Xtest)

                case "qda":
                    clf=sklda.QuadraticDiscriminantAnalysis()

                    # train the classifier
                    clf.fit(Xtrain,Ytrain)

                    # apply the classifier to the training and test data to obtain the probabilities
                    Ytrain_pred=clf.predict_proba(Xtrain)
                    Ytest_pred=clf.predict_proba(Xtest)
                    #print(Ytrain_pred.shape)


                
                # case "customqda":
                #     clf=isl.qdaClassifier(Xtrain.T, Ytrain)

                #     # apply the classifier to the training and test data
                #     Ytrain_pred=np.empty(Xtrain.shape[0])
                #     for i in range(Xtrain.shape[0]):
                #         Ytrain_pred[i]=clf.qda_classifier(Xtrain[i])

                #     Ytest_pred=np.empty(Xtest.shape[0])
                #     for i in range(Xtest.shape[0]):
                #         Ytest_pred[i]=clf.qda_classifier(Xtest[i])

                case "kriging":
                    # calculate the number of classes
                    nclasses=len(np.unique(Ytrain))

                    # calculate the number of data per class in the training data
                    Nk=np.array([np.sum(Ytrain==i) for i in range(nclasses)])

                    # calculate alpha
                    alphak=[Nk[i]/2 for i in range(nclasses)]
                    # alphak=None

                    # create the classifier
                    clf=isl.KrigBayesian(Xtrain.T,krig_gamma=kr_gamma, alphak=alphak, Fk=None, ytrain=Ytrain)

                    # create a matrix of size nclases x Xtrain.shape[0] to store the results
                    Ytrain_pred=np.empty([nclasses,Xtrain.shape[0]])

                    # apply the classifier to the training and test data
                    for i in range(Xtrain.shape[0]):
                        # store the results in the matrix.
                        tempres=clf.class_prob(Xtrain[i])
                        Ytrain_pred[:,i]=tempres

                    # create a matrix of size nclasses x Xtest.shape[0] to store the results
                    Ytest_pred=np.empty([nclasses,Xtest.shape[0]])
                    for i in range(Xtest.shape[0]):
                        # store the results in the matrix.
                        tempres=clf.class_prob(Xtest[i])
                        Ytest_pred[:,i]=tempres
                    # Transpose the results to match the shape of the grid
                    Ytrain_pred=Ytrain_pred.T
                    Ytest_pred=Ytest_pred.T
                    #print(Ytrain_pred.shape)

                case "krigingOpt":
                    # calculate the number of classes
                    nclasses=len(np.unique(Ytrain))

                    # create the classifier
                    clf=isl.KrigOpt(Xtrain.T,krig_gamma=kr_gamma, alphak=None, Fk=None, ytrain=Ytrain)

                    # create a matrix of size nclases x Xtrain.shape[0] to store the results
                    Ytrain_pred=np.empty([nclasses,Xtrain.shape[0]])

                    # apply the classifier to the training and test data
                    for i in range(Xtrain.shape[0]):
                        # store the results in the matrix.
                        tempres=clf.class_prob(Xtrain[i])
                        Ytrain_pred[:,i]=tempres

                    # create a matrix of size nclasses x Xtest.shape[0] to store the results
                    Ytest_pred=np.empty([nclasses,Xtest.shape[0]])
                    for i in range(Xtest.shape[0]):
                        # store the results in the matrix.
                        tempres=clf.class_prob(Xtest[i])
                        Ytest_pred[:,i]=tempres
                    # Transpose the results to match the shape of the grid
                    Ytrain_pred=Ytrain_pred.T
                    Ytest_pred=Ytest_pred.T
                    #print(Ytrain_pred.shape)
                
                case "dissimilarityQDA":
                    # calculate ck as the number of elements of each class divided by 2
                    ck=[np.sum(Ytrain==i)/2 for i in range(len(np.unique(Ytrain)))]
                    # calculate fk as e^(1/2)/((2*pi)^(d/2)*Ek^(1/2)) where d is the number of dimensions of the data and E is the determinant of the covariance matrix of each class
                    fk=[np.exp(1/2)/(np.power(2*np.pi,Xtrain.shape[1]/2)*np.sqrt(np.linalg.det(np.cov(Xtrain[Ytrain==i].T)))) for i in range(len(np.unique(Ytrain)))]
                    #create the classifier
                    clf=isl.DisFunClass(Xtrain.T, Ytrain,ck=ck,Fk=fk)
                    
                    #apply the classifier to the training and test data to obtain the probabilities. these loops can be parallelized
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytrain_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtrain]), total=len(Xtrain)))
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytest_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtest]), total=len(Xtest)))
                    
                    #convert the list to a numpy array
                    Ytrain_pred = np.array(Ytrain_pred)
                    Ytest_pred = np.array(Ytest_pred)
                case "dissimilarityWckCalHC":

                    # separate the training data into training and calibration data
                    Xtrain, Xcal, Ytrain, Ycal = train_test_split(Xtrain, Ytrain, test_size=train_split, random_state=42, stratify=Ytrain)

                    #get the value of ck for the dissimilarity function classifier
                    ck=[np.sum(Ycal==i)/2 for i in range(len(np.unique(Ycal)))]

                    # get the value of Fk for the dissimilarity function classifier as \frac{e^{\frac{1}{2}}}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
                    Fk = [np.exp(0.5)/(np.power(2*np.pi, Xtrain.shape[1]/2)*np.sqrt(np.linalg.det(np.cov(Xtrain[Ytrain==i].T))) ) for i in range(len(np.unique(Ytrain)))]
                    # create the dissimilarity function classifier
                    clf=isl.DisFunClass(Xtrain.T, Ytrain, Xcal=Xcal.T, ycal=Ycal,ck=ck,Fk=None, gam=kr_gamma, ck_init=ck, Fk_init=Fk)
                    #apply the classifier to the training and test data to obtain the probabilities. these loops can be parallelized
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytrain_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtrain]), total=len(Xtrain)))
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytest_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtest]), total=len(Xtest)))

                    #convert the list to a numpy array
                    Ytrain_pred = np.array(Ytrain_pred)
                    Ytest_pred = np.array(Ytest_pred)

                case "dissimilarityWckCal":

                    # separate the training data into training and calibration data
                    Xtrain, Xcal, Ytrain, Ycal = train_test_split(Xtrain, Ytrain, test_size=train_split, random_state=42, stratify=Ytrain)

                    # calculate ck as the number of elements of each class divided by 2
                    ck=[np.sum(Ytrain==i)/2 for i in range(len(np.unique(Ytrain)))]
                    #create the classifier
                    clf=isl.DisFunClass(Xtrain.T, Ytrain, Xcal=Xcal.T, ycal=Ycal, ck=ck, Fk=None, gam=kr_gamma)
                    
                    #apply the classifier to the training and test data to obtain the probabilities. these loops can be parallelized
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytrain_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtrain]), total=len(Xtrain)))
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytest_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtest]), total=len(Xtest)))

                    
                    #convert the list to a numpy array
                    Ytrain_pred = np.array(Ytrain_pred)
                    Ytest_pred = np.array(Ytest_pred)
                case "dissimilarityWck":
                    # calculate ck as the number of elements of each class divided by 2
                    ck=[np.sum(Ytrain==i)/2 for i in range(len(np.unique(Ytrain)))]
                    #create the classifier
                    clf=isl.DisFunClass(Xtrain.T, Ytrain,ck=ck,Fk=None, gam=kr_gamma)
                    
                    #apply the classifier to the training and test data to obtain the probabilities. these loops can be parallelized
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytrain_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtrain]), total=len(Xtrain)))
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytest_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtest]), total=len(Xtest)))

                    
                    #convert the list to a numpy array
                    Ytrain_pred = np.array(Ytrain_pred)
                    Ytest_pred = np.array(Ytest_pred)
                case "dissimilarity":
                    #create the classifier
                    clf=isl.DisFunClass(Xtrain.T, Ytrain,ck=None,Fk=None)
                    
                    #apply the classifier to the training and test data to obtain the probabilities. these loops can be parallelized
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytrain_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtrain]), total=len(Xtrain)))
                    with mp.Pool(mp.cpu_count()) as pool:
                        Ytest_pred = list(tqdm.tqdm(pool.imap(classify_probs, [(x, clf) for x in Xtest]), total=len(Xtest)))

                    
                    #convert the list to a numpy array
                    Ytrain_pred = np.array(Ytrain_pred)
                    Ytest_pred = np.array(Ytest_pred)
                case _:
                    # if the clasifier is not valid, print an error message
                    print("unvalid classifier")
            # If the number of classes is 2, add a column of zeros to the probabilities
            if Ytrain_pred.shape[1]==2:
                Ytrain_pred=np.c_[Ytrain_pred,np.zeros(Ytrain_pred.shape[0])]
                Ytest_pred=np.c_[Ytest_pred,np.zeros(Ytest_pred.shape[0])]
            #print(Ytest_pred)
            

            colors=["r","g","b"]
            colors_pred = [colors[col] for col in Ytrain]

            # reshape the results to the grid shape
            Ytest_pred=Ytest_pred.reshape([xx.shape[0],xx.shape[1],3])

            # create an array to store the class with the highest probability
            Ytest_pred_class=np.argmax(Ytest_pred,axis=2)

            # create an array to find borders of the classes
            Ytest_pred_class_border=np.ones(Ytest_pred_class.shape)
            for i in range(1,Ytest_pred_class.shape[0]-1):
                for j in range(1,Ytest_pred_class.shape[1]-1):
                    if Ytest_pred_class[i,j]!=Ytest_pred_class[i-1,j] or Ytest_pred_class[i,j]!=Ytest_pred_class[i,j-1] or Ytest_pred_class[i,j]!=Ytest_pred_class[i-1,j-1] or Ytest_pred_class[i,j]!=Ytest_pred_class[i+1,j]:
                        Ytest_pred_class_border[i,j]=0
            
            # plot the borders of the classes using imshow
            plt.imshow(Ytest_pred_class_border,extent=(xmin,xmax,ymin,ymax),origin="lower")

            # Plot the training data with the colors defined by the colors array
            plt.scatter(Xtrain[:,0],Xtrain[:,1],c=colors_pred,marker="x",s=10)
            
            #check if the plots folder exists and create it if it doesn't
            if not os.path.exists("Plots"):
                os.makedirs("Plots")

            # Form the name of the file
            filename="Plots\\borders\\"+fileprefix+clasif
            # If the classifier isn't qda, add the lambda value to the name; put two decimal places and remove the decimal point
            if clasif!="qda":
                filename+="Lambda"+"{:.2f}".format(kr_gamma).replace(".","")

            # If the classifier is a calibrated dissimilarity function, add the calibration split to the name
            if clasif=="dissimilarityWckCal" or clasif=="dissimilarityWckCalHC":
                filename+="Cal"+"{:.2f}".format(train_split).replace(".","")
            # Add the database number to the name
            filename+="_"+str(file)+".png"

            print("Saving plot to "+filename)
            
                
            # save the plot using the clasifier name and the database name as the name of the file
            plt.savefig(filename)
            #close the plot
            plt.close()