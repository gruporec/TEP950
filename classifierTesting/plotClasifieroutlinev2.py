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

    # fix the limits of the plot
    plotlims=(-2,2,-2,2)
    #plotlims=None

    # grid step
    gridstep=0.005

    #margin for the plot
    margin=0.1

    #accuracy margin
    acc_margin=0.1

    # clasifier types to use. Available types: "qda", "dissimilarityWckCalHC", "dissimilarity"
    clasifs=["dissimilarityWckCalHC"]

    # gamma for kriging 
    kr_gammas=[0]

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

        

        # separate the training data into training and calibration data
        Xtrain, Xcal, Ytrain, Ycal = train_test_split(Xtrain, Ytrain, test_size=train_split, random_state=42, stratify=Ytrain)
        
        #if the limits of the plot are not fixed, calculate them
        if plotlims==None:
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
        else:
            xmin=plotlims[0]
            xmax=plotlims[1]
            ymin=plotlims[2]
            ymax=plotlims[3]

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
            if clasif=="qda":
                clf=sklda.QuadraticDiscriminantAnalysis()

                # train the classifier
                clf.fit(Xtrain,Ytrain)

                # apply the classifier to the training and test data to obtain the probabilities
                Ytrain_pred=clf.predict_proba(Xtrain)
                Ytest_pred=clf.predict_proba(Xtest)

                # If the number of classes is 2, add a column of zeros to the probabilities
                if Ytrain_pred.shape[1]==2:
                    Ytrain_pred=np.c_[Ytrain_pred,np.zeros(Ytrain_pred.shape[0])]
                    Ytest_pred=np.c_[Ytest_pred,np.zeros(Ytest_pred.shape[0])]
                #print(Ytest_pred)
                

                # reshape the results to the grid shape
                Ytest_pred=Ytest_pred.reshape([xx.shape[0],xx.shape[1],3])

                # create an array to store the class with the highest probability
                Ytest_pred_class=np.argmax(Ytest_pred,axis=2)

                # create an array to find borders of the classes
                Ytest_pred_class_border=np.ones(Ytest_pred_class.shape)
                for i in range(1,Ytest_pred_class.shape[0]-1):
                    for j in range(1,Ytest_pred_class.shape[1]-1):
                        if Ytest_pred_class[i,j]!=Ytest_pred_class[i-1,j] or Ytest_pred_class[i,j]!=Ytest_pred_class[i+1,j] or Ytest_pred_class[i,j]!=Ytest_pred_class[i,j-1] or Ytest_pred_class[i,j]!=Ytest_pred_class[i,j+1]:
                            Ytest_pred_class_border[i,j]=0
                #change the ones to nans
                Ytest_pred_class_border[Ytest_pred_class_border==1]=np.nan
                
                # plot the borders of the classes using imshow
                plt.imshow(Ytest_pred_class_border,extent=(xmin,xmax,ymin,ymax),origin="lower")

                # Plot the training data for class 0 in grey with shape x
                plt.scatter(Xtrain[Ytrain==0,0],Xtrain[Ytrain==0,1],color="grey",marker="x",s=10)

                # Plot the training data for class 1 in grey with shape o
                plt.scatter(Xtrain[Ytrain==1,0],Xtrain[Ytrain==1,1],color="grey",marker="o",s=10)
                
                #check if the plots folder exists and create it if it doesn't
                if not os.path.exists("Plots"):
                    os.makedirs("Plots")

                # Form the name of the file
                filename="Plots\\borders\\"+fileprefix+clasif

                # Add the database number to the name and create another filename for the plot without the scatter
                scatterlessfilename=filename+"_"+str(file)+"_scatterless.png"
                filename+="_"+str(file)+".png"

                print("Saving plot to "+filename)
                # save the plot using the clasifier name and the database name as the name of the file
                plt.savefig(filename)
                #close the plot
                plt.close()

                #plot the borders of the classes using imshow
                plt.imshow(Ytest_pred_class_border,extent=(xmin,xmax,ymin,ymax),origin="lower")
                
                # Save the plot without the scatter
                plt.savefig(scatterlessfilename)
                #close the plot
                plt.close()

                # Save the probabilities to a file
                np.savetxt(filename.replace(".png",".csv"),Ytest_pred[:,:,0],delimiter=",")

                # Save the probabilities to a file
                np.savetxt(filename.replace(".png","_train.csv"),Ytrain_pred,delimiter=",")
                
                #Save the border limits
                np.savetxt(filename.replace(".png","_limits.csv"),np.array([xmin,xmax,ymin,ymax]),delimiter=",")
            else:
                for kr_gamma in kr_gammas:
                    # select the classifier according to clasif
                    match clasif:
                        case "dissimilarityWckCalHC":

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
                        case "dissimilarity":
                            
                            # separate the training data into training and calibration data
                            Xtrain, Xcal, Ytrain, Ycal = train_test_split(Xtrain, Ytrain, test_size=train_split, random_state=42, stratify=Ytrain)

                            #get the value of ck for the dissimilarity function classifier
                            ck=[np.sum(Ycal==i)/2 for i in range(len(np.unique(Ycal)))]

                            # get the value of Fk for the dissimilarity function classifier as \frac{e^{\frac{1}{2}}}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
                            Fk = [np.exp(0.5)/(np.power(2*np.pi, Xtrain.shape[1]/2)*np.sqrt(np.linalg.det(np.cov(Xtrain[Ytrain==i].T))) ) for i in range(len(np.unique(Ytrain)))]
                            # create the dissimilarity function classifier
                            clf=isl.DisFunClass(Xtrain.T, Ytrain, Xcal=Xcal.T, ycal=Ycal,ck=ck,Fk=Fk, gam=kr_gamma, ck_init=ck, Fk_init=Fk)
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
                        
                    print("ck and Fk for",clasif)
                    print(clf.ck)
                    print(clf.Fk)
        

                    # reshape the results to the grid shape
                    Ytest_pred=Ytest_pred.reshape([xx.shape[0],xx.shape[1],3])

                    # create an array to store the class with the highest probability
                    Ytest_pred_class=np.argmax(Ytest_pred,axis=2)

                    # create an array to find borders of the classes
                    Ytest_pred_class_border=np.ones(Ytest_pred_class.shape)
                    for i in range(1,Ytest_pred_class.shape[0]-1):
                        for j in range(1,Ytest_pred_class.shape[1]-1):
                            if Ytest_pred_class[i,j]!=Ytest_pred_class[i-1,j] or Ytest_pred_class[i,j]!=Ytest_pred_class[i+1,j] or Ytest_pred_class[i,j]!=Ytest_pred_class[i,j-1] or Ytest_pred_class[i,j]!=Ytest_pred_class[i,j+1]:
                                Ytest_pred_class_border[i,j]=0
                    #change the ones to nans
                    Ytest_pred_class_border[Ytest_pred_class_border==1]=np.nan
                    
                    # plot the borders of the classes using imshow
                    plt.imshow(Ytest_pred_class_border,extent=(xmin,xmax,ymin,ymax),origin="lower")

                    # Plot the training data for class 0 in grey with shape x
                    plt.scatter(Xtrain[Ytrain==0,0],Xtrain[Ytrain==0,1],color="grey",marker="x",s=10)

                    # Plot the training data for class 1 in grey with shape o
                    plt.scatter(Xtrain[Ytrain==1,0],Xtrain[Ytrain==1,1],color="grey",marker="o",s=10)
                    
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
                    # Add the database number to the name and create another filename for the plot without the scatter
                    scatterlessfilename=filename+"_"+str(file)+"_scatterless.png"
                    filename+="_"+str(file)+".png"

                    print("Saving plot to "+filename)
                    
                    # save the plot using the clasifier name and the database name as the name of the file
                    plt.savefig(filename)
                    #close the plot
                    plt.close()

                    #plot the borders of the classes using imshow
                    plt.imshow(Ytest_pred_class_border,extent=(xmin,xmax,ymin,ymax),origin="lower")
                    
                    # Save the plot without the scatter
                    plt.savefig(scatterlessfilename)
                    #close the plot
                    plt.close()

                    # Save the probabilities to a file
                    np.savetxt(filename.replace(".png",".csv"),Ytest_pred[:,:,0],delimiter=",")

                    # Save the probabilities to a file
                    np.savetxt(filename.replace(".png","_train.csv"),Ytrain_pred,delimiter=",")

                    # If the classifier is a dissimilarity function, save the values of ck and Fk
                    if clasif=="dissimilarityWckCalHC" or clasif=="dissimilarityWckCal" or clasif=="dissimilarityWck" or clasif=="dissimilarity":
                        np.savetxt(filename.replace(".png","_ck.csv"),clf.ck,delimiter=",")
                        np.savetxt(filename.replace(".png","_Fk.csv"),clf.Fk,delimiter=",")
                    
                    #Save the border limits
                    np.savetxt(filename.replace(".png","_limits.csv"),np.array([xmin,xmax,ymin,ymax]),delimiter=",")