import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Number of classes
n=2

# Number of samples per class
m=[100,100,100,100,100]

# Center of the classes
c=[[0,0],[1,1],[0,1],[1,0],[0.5,0.5]]

# Standard deviation of the classes
s=[0.3,0.3,0.3,0.3,0.3]

#size of the shapes
size=[0.5,0.5,0.5,0.5,0.5]

# Shape of the classes
shape=["dot","circle"]

#Get the folder where the script is
folder=os.path.dirname(__file__)

# Create an empty dataframe
data=pd.DataFrame()

# For each class
for i in range(n):

    # Match case for the shape of the class
    match shape[i]:
        case "circle":
            # create a empty dataframe
            df=pd.DataFrame()
            # For each sample of the class
            for j in range(m[i]):
                # Create a random angle
                angle=np.random.uniform(0,2*np.pi,1)
                # Create a random radius with s as standard deviation and size as mean
                radius=np.random.normal(size[i],s[i],1)
                # Get a point that is the radius away from the center in the angle direction
                point=np.array(c[i])+radius*np.array([np.cos(angle),np.sin(angle)])
                # Add the point to the dataframe
                df=pd.concat([df,pd.DataFrame(point,columns=["a","b"])])
                print(df)
                sys.exit()
        case _: # dot
            # Create a dataframe with the values of the class
            df=pd.DataFrame(np.random.normal(c[i],s[i],(m[i],len(c[i]))),columns=["a","b"])

    # Add a column with the class
    df["Y"]=i

    # Add the dataframe to the data
    data=pd.concat([data,df])

print(data)
# Create a list with the colors of the classes
colors=["red","blue","green","yellow","black"]

# Plot the data
plt.scatter(data["a"],data["b"],c=data["Y"].apply(lambda x: colors[x]))

# show the plot
plt.show()

count=0
saved=0
while(saved==0):
    # Check if count.csv exists in this folder
    if os.path.isfile(folder+"/db/"+str(count)+".csv"):
        # If it exists, add 1 to the count
        count=count+1
    # Else, save the dataframe as count.csv
    else:
        data.to_csv(folder+"/db/"+str(count)+".csv",index=False)
        saved=1