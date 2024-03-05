import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import qpsolvers as qp
import sys
import matplotlib.path as mpath
import dissimClassLib as dcl

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
shapeName = 'triangleAndGaussian'

# define a triangle
corners = np.array([[0, -0.5], [2, -0.5], [1, 1*np.sqrt(3)-0.5]])

# define the mean and variance of the Gaussian function
meanb = np.array([-1, 0])
varb = 1

# flag to only plot the shape
onlyShape = True

# flag to only plot the points in the shape
onlyPoints = False

############################################ PROCESSING ############################################

if onlyShape:
    #plot the shape as a closed polygon, repeat the first point to close the polygon
    plt.plot(np.append(corners[:, 0], corners[0, 0]), np.append(corners[:, 1], corners[0, 1]))
    # plot a point in the mean of the Gaussian function
    plt.scatter(meanb[0], meanb[1], c='red', s=10, marker='x')
    # plot a circle with the variance of the Gaussian function
    circle = plt.Circle(meanb, varb, color='r', fill=False)
    plt.gca().add_artist(circle)

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
dissimDist = dcl.dissimDistribution(dataT, gam, gamma2, c)

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
    Zgauss.append(dcl.gaussian(p, dissimDist.mean, dissimDist.cov))
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