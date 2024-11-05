import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
import tifffile
import napari
from tapenade.preprocessing import change_arrays_pixelsize, local_image_equalization
from time import time
from skimage.filters import threshold_otsu


data = tifffile.imread(
    '/home/jvanaret/data/data_paper_tapenade/morphology/ag6.lsm'
)[:,0]

data = change_arrays_pixelsize(image=data, reshape_factors=(1,1/2,1/2))
data = (data/np.max(data)).astype(np.float32)
datae = local_image_equalization(data, box_size=2, perc_low=1, perc_high=99)



t0 = time()
gf = gaussian_filter(data, sigma=1)
print('Gaussian filter:', time()-t0)
t0 = time()
mf = median_filter(data, size=2)
print('Median filter:', time()-t0)

t0 = time()
dat = np.abs(mf - gf)/gf
print('Difference:', time()-t0)

t0 = time()
threshold = threshold_otsu(dat)
print('Threshold:', time()-t0)

t0 = time()
gfe = gaussian_filter(datae, sigma=1)
print('Gaussian filter:', time()-t0)
t0 = time()
mfe = median_filter(datae, size=2)
print('Median filter:', time()-t0)

t0 = time()
date = np.abs(mfe - gfe)/gfe
print('Difference:', time()-t0)

t0 = time()
thresholde = threshold_otsu(date)
print('Threshold:', time()-t0)

viewer = napari.Viewer()
viewer.add_image(data, name='data', colormap='gray')
viewer.add_image(datae, name='datae', colormap='gray')
viewer.add_image(gf, name='gaussian', colormap='gray')
viewer.add_image(mf, name='median', colormap='gray')
viewer.add_image(dat, name='dat', colormap='gray')
viewer.add_image(gfe, name='gaussian', colormap='gray')
viewer.add_image(mfe, name='median', colormap='gray')
viewer.add_image(date, name='dat', colormap='gray')

viewer.add_image(dat>threshold, name='threshold', colormap='gray')
viewer.add_image(date>thresholde, name='threshold', colormap='gray')

napari.run()


