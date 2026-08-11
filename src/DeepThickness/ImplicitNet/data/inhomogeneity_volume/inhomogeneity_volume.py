'''
Created on Wednesday - May 13 2020, 10:02:54

@author: Michele Svanera, University of Glasgow

See: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

Code to create 1-D, 2-D, and 3-D inhomogeneity augmented data.

'''

################################################################################################################
## Imports


import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.stats import multivariate_normal

################################################################################################################
## Paths and Constants

#Path_out = './'


################################################################################################################
## Main

path = "/analyse/Project0235/segmentator/data/multi_site_data/T1/1000_FCP_ABIDE_Olin_sub-0050121_T1w_256iso.nii.gz"
T1 = nib.load(path)
#GT = nib.load(Path_in_GT)

# load the volume
t1 = T1.get_data()#[:,:,:,0]
#gt = GT.get_data()


#################################### 1-D case ###################################
#x = np.linspace(0, 1, t1.shape[0] * 2, endpoint=True)
#y = multivariate_normal.pdf(x, mean=0.5, cov=0.25)
#
## Standardisation
#y -= np.mean(y)
#
## Random crop (select just a part of it)
#x_1 = np.random.randint(0, t1.shape[0]-1, size=1)[0]
#y_1 = y[x_1 : x_1 + t1.shape[0]]
#x_1 = x[x_1 : x_1 + t1.shape[0]]
#
## plot
#plt.plot(x, y, '')
#plt.plot(x_1, y_1, '*')
#plt.axis([-0.1, 1.1, -1.1, 1.1])
#
#
#################################### 2-D case ###################################
#x_1_linear = np.linspace(0, 1, t1.shape[0]*2, endpoint=True)
#x_2_linear = np.linspace(0, 1, t1.shape[1]*2, endpoint=True)
#x_1,x_2 = np.meshgrid(x_1_linear,x_2_linear)
#pos = np.dstack((x_1, x_2))
#
## create distrs
#mean = [0.5, 0.5]
#cov = [[2.0, 0.3], [0.3, 2.0]]
#rv = multivariate_normal(mean, cov, allow_singular=True)
#y = rv.pdf(pos)             # derive distribution
#y -= np.mean(y)             # standardisation (I want across zero)
#y = y / max(y.max(), np.abs(y.min())) * 0.25    # max (or min) = 0.25 (-0.25)
#
## Random crop (select just a part of it)
#x_1_corner = np.random.randint(0, t1.shape[0]-1, size=1)[0]
#x_2_corner = np.random.randint(0, t1.shape[1]-1, size=1)[0]
#y_1 = y[x_2_corner : x_2_corner + t1.shape[1], x_1_corner : x_1_corner + t1.shape[0]].T
#
#assert y_1.shape == (t1.shape[0],t1.shape[1])
#
## plot
#fig, (ax1, ax2) = plt.subplots(ncols=2)
#
#posit = ax1.imshow(y.T, origin='lower', vmin=y.min(), vmax=y.max())
#fig.colorbar(posit, ax=ax1)
#
#rect = patches.Rectangle((x_2_corner, x_1_corner), t1.shape[1], t1.shape[0],
#                         linewidth=1,edgecolor='r',facecolor='none') # Create a Rectangle
#ax1.add_patch(rect)         # Add the patch to the Axes
#
#posit = ax2.imshow(y_1, origin='lower', vmin=y.min(), vmax=y.max())
#
#fig.colorbar(posit)
#
## Plot 3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x_1, x_2, y, cmap=cm.viridis, linewidth=0)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#

################################### 3-D case ###################################
molt_factor = 2
max_value = 1. #0.5                     # max (or min) = 0.25 (-0.25)

# create axis
x_1 = np.linspace(0, 1, int(256*molt_factor), endpoint=True)
x_2 = np.linspace(0, 1, int(256*molt_factor), endpoint=True)
x_3 = np.linspace(0, 1, int(256*molt_factor), endpoint=True)
x_1,x_2,x_3 = np.meshgrid(x_1,x_2,x_3)
pos = np.stack((x_1, x_2, x_3),axis=-1)

# create distrs
mean = [0.5, 0.5, 0.5]
cov = [[2.0, 0.3, 0.3], [0.3, 2.0, 0.3], [0.3, 0.3, 2.0]]
rv = multivariate_normal(mean, cov, allow_singular=True)
y = rv.pdf(pos)             # derive distribution
y -= np.mean(y)             # standardisation (I want across zero)
y = y / max(y.max(), np.abs(y.min())) * max_value
y = np.transpose(y, (1, 0, 2))      # swap axes

## Random crop (select just a part of it)
x_1 = np.random.randint(0, int(t1.shape[0])-1, size=1)[0]
x_2 = np.random.randint(0, int(t1.shape[1])-1, size=1)[0]
x_3 = np.random.randint(0, int(t1.shape[2])-1, size=1)[0]
y_1 = y[x_1 : x_1 + t1.shape[0], x_2 : x_2 + t1.shape[1], x_3 : x_3 + t1.shape[2]]

assert y_1.shape == t1.shape


# Save the 'npy' volume
np.save('./inhomogeneity.npy', y)

# Save the nifti volume (for visualisation only)
vol = vol = nib.Nifti1Image(y_1, affine=T1.get_sform(), header=T1.header)
nib.save(vol, './inhomogeneity.nii.gz')

