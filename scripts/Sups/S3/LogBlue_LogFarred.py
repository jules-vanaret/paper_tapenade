import tifffile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.stats import linregress
from skimage.measure import block_reduce
from tapenade.preprocessing._preprocessing import change_array_pixelsize
import scipy.ndimage as ndi

folder = ...
fig,ax = plt.subplots(1,figsize=(10,7))
all_blue = []
all_farred = []
subfolder = f'hoechst_and_draq5'
scale=(1,0.62,0.62) 
def flatten_image(image,mask,scale,blocksize,threshold):
    image = image.astype(float)
    image = change_array_pixelsize(image, scale)
    mask = change_array_pixelsize(mask, scale,order=0)
    mask = ndi.binary_erosion(mask,iterations=5)
    image[mask==False]=np.nan
    image_log = np.log1p(image)
    image_blockreduced = block_reduce(image_log, block_size=(blocksize,blocksize,blocksize), func=np.nanmean)
    image_blockreduced[image_blockreduced<threshold]=np.nan
    image_flat = image_blockreduced.flatten()
    return(image_flat)

for num in tqdm(range(1)):
    image = tifffile.imread(Path(folder) /f'S3_wavelength/{subfolder}/{num+1}.tif')
    farred = image[:,3,:,:].astype(float)
    blue = image[:,0,:,:].astype(float)
    mask = tifffile.imread(Path(folder) /f'S3_wavelength/{subfolder}/masks/{num+1}.tif')
    blue_flat = flatten_image(blue,mask,scale =scale,blocksize = 10, threshold =0)
    farred_flat = flatten_image(farred,mask,scale =scale,blocksize = 10, threshold =4)
    valid_indices = ~np.isnan(blue_flat) & ~np.isnan(farred_flat)
    all_blue.extend(blue_flat[valid_indices])
    all_farred.extend(farred_flat[valid_indices])
ax.scatter(all_blue, all_farred, s=1, alpha=0.5,color='black')

slope, intercept, r_value, p_value, std_err = linregress(all_blue, all_farred)
x_fit = np.array([min(all_blue), max(all_blue)])
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")

plt.xlabel('log(Blue detection channel)',fontsize=30)
plt.ylabel('log(Far-red detection channel)',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.legend()
plt.savefig(Path(folder) /f'S3e_plot.svg')
plt.show()