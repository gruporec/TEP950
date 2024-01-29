import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Number of classes
n=2

# Number of samples
m=1000

# Center of the shapes
c=[[0,0],[1,0]]

#size of the shapes
size=[1,1]

# Shape of the classes
shape=["circle", "square"]

#Get the folder where the script is
folder=os.path.dirname(__file__)

# Create a dataframe with m random points of the last class
data=pd.DataFrame(np.random.normal(0,1,(m,2)),columns=["a","b"])

# Add a column with the class
data["Y"]=n-1
print(data)

# Create a plot storing the axis
fig,ax=plt.subplots()

# Create a list of figures
figs=[]

# For each class except the last
for i in range(n-1):
    # Match case for the shape of the class
    match shape[i]:
        case "circle":
            # For each sample
            for sample in data.iterrows():
                # Check if the sample is in the circle
                if np.linalg.norm(sample[1][["a","b"]]-c[i])<size[i]:
                    # If it is, change the class of the sample
                    data.at[sample[0],"Y"]=i
            
            # Add the circle to the plot
            figs.append(plt.Circle(c[i],size[i],color="black",fill=False))
            ax.add_patch(figs[i])
        case "square":
            # For each sample
            for sample in data.iterrows():
                # Check if the sample is in the square
                if np.max(np.abs(sample[1][["a","b"]]-c[i]))<size[i]:
                    # If it is, change the class of the sample
                    data.at[sample[0],"Y"]=i
            # Add the square to the plot
            figs.append(plt.Rectangle([c[i][0]-size[i],c[i][1]-size[i]],2*size[i],2*size[i],color="black",fill=False))
            ax.add_patch(figs[i])
        case _: # dot
            # Create a dataframe with the values of the class
            df=pd.DataFrame(np.random.normal(c[i],s[i],(m[i],len(c[i]))),columns=["a","b"])


print(data)
# Create a list with the colors of the classes
colors=["red","blue","green","yellow","black"]

# Create a list with the styles of the classes
styles=["^","x","*","o","s"]

# Add the data to the plot using the colors and styles per class
for i in range(n):
    ax.scatter(data[data["Y"]==i]["a"],data[data["Y"]==i]["b"],color=colors[i],marker=styles[i])

# show the plot
#plt.show()

count=0
saved=0
while(saved==0):
    # Check if count.csv exists in this folder
    if os.path.isfile(folder+"/dbv2/"+str(count)+".csv"):
        # If it exists, add 1 to the count
        count=count+1
    # Else, save the dataframe as count.csv and the plot as count.png
    else:
        data.to_csv(folder+"/dbv2/"+str(count)+".csv",index=False)
        fig.savefig(folder+"/dbv2/"+str(count)+".png")
        saved=1