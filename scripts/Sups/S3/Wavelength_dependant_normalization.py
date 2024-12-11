import tifffile
import numpy as np
import napari
import matplotlib.pyplot as plt
from tapenade.preprocessing._preprocessing import change_array_pixelsize,_masked_smooth_gaussian
from tapenade.preprocessing._intensity_normalization import _optimize_sigma,_nans_outside_mask,_find_reference_plane_from_medians
from pathlib import Path

def normalize(
    array: np.ndarray,
    ref_array: np.ndarray,
    sigma=None,
    mask: np.ndarray = None,
    labels: np.ndarray = None,
    exponentiation_coeff: float = 1,
    width=3,
):
    """
    MODIFIED FROM TAPENADE PACKAGE TO PLOT DIFFERENT VALUES OF RATIO exponentiation_coeff
    Normalize the intensity of an array based on a reference array assumed to have
    ideally homogeneous signal (e.g DAPI).

    Parameters:
    - array (ndarray): The input array to be normalized.
    - ref_array (ndarray): The reference array used for normalization.
    - sigma (float, optional): The standard deviation for Gaussian smoothing of the reference array. Default is None,
        but a value is required right now since setting 'sigma' to None is not implemented.
    - mask (ndarray, optional): A binary mask of the sample. Default is None.
    - labels (ndarray, optional): An array of labels indicating the instances in which the reference
        signal is expressed, e.g nuclei. Default is None.
    - image_wavelength (float, optional): direct value for exponentiation of the reference array. Default is1
    - width (int, optional): The number of neighboring planes to consider for reference plane calculation. Default is 5.

    Returns:
    - array_norm (ndarray): The normalized input array.
    - ref_array_norm (ndarray): The normalized reference array.
    """

    num_z_slices = array.shape[0]

    labels_mask = None if labels is None else labels.astype(bool)

    # apply wavelength correction
    if exponentiation_coeff is not None:
        
        ref_array = ref_array ** exponentiation_coeff

    # compute smoothed reference array for normalization
    if sigma is None:
        sigma = _optimize_sigma(ref_array, mask, labels_mask)
        print("sigma = ", sigma)
    ref_array_smooth = _masked_smooth_gaussian(
        ref_array, sigmas=sigma, mask_for_volume=labels_mask, mask=mask
    )

    if mask is not None:
        array = _nans_outside_mask(array, mask)
        ref_array = _nans_outside_mask(ref_array, mask)
        ref_array_smooth = _nans_outside_mask(ref_array_smooth, mask)

        mask_divide = mask
    else:
        mask_divide = True

    # normalize array and reference array
    array_norm = np.divide(array, ref_array_smooth, where=mask_divide)
    ref_array_norm = np.divide(ref_array, ref_array_smooth, where=mask_divide)

    if mask is not None:
        array_norm = _nans_outside_mask(array_norm, mask)
        ref_array_norm = _nans_outside_mask(ref_array_norm, mask)

    # rectify median intensity in both normalized arrays
    # to that of the median of the brightest consecutive planes
    z_ref = _find_reference_plane_from_medians(ref_array)
    z_ref_norm = _find_reference_plane_from_medians(ref_array_norm)

    sl = slice(max(0, z_ref - width), min(num_z_slices, z_ref + width))
    sl_norm = slice(
        max(0, z_ref_norm - width), min(num_z_slices, z_ref_norm + width)
    )

    array_normalization_factor = np.nanmedian(
        array_norm[sl_norm]
    ) / np.nanmedian(array[sl])

    array_norm = array_norm / array_normalization_factor

    return array_norm


def intensity_along_axis(image, mask,axis):
    image = image.astype(float)
    image[mask == 0] = np.nan
    return np.nanmean(image, axis=axis)

def plot_and_save(path,xlabel,ylabel,yticks):
    plt.legend()
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(ylabel,fontsize=30)
    plt.xticks([0,100,200,300],fontsize=30)
    plt.yticks(yticks,fontsize=30)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(path)
    plt.show()

folder = Path(__file__).parents[3] / 'data'
num=1 #num sample
scale = (1,0.6,0.6)

im = tifffile.imread(Path(folder) / f'S3_wavelength/hoechst_and_draq5/{num}.tif')
mask = tifffile.imread(Path(folder) / f'S3_wavelength/hoechst_and_draq5/masks/{num}.tif')
farred = im[:,3,:,:]
blue = im[:,0,:,:]
blue_iso = change_array_pixelsize(array=blue,input_pixelsize=scale)
farred_iso = change_array_pixelsize(array=farred,input_pixelsize=scale)
mask_iso = change_array_pixelsize(array=mask,input_pixelsize=scale,order=0)

#S3f_1
fig = plt.figure(figsize=(10,7))
farred_norm_048 = normalize(array = farred_iso,ref_array = blue_iso, exponentiation_coeff=0.48,sigma=15,mask=mask_iso,labels=None,width=3)
farred_norm_1 = normalize(array = farred_iso,ref_array = blue_iso, exponentiation_coeff=1,sigma=15,mask=mask_iso,labels=None,width=3)
plt.plot(intensity_along_axis(farred_iso,mask=mask_iso,axis=(1,2)),label='unnormalized signal',color='black',linewidth=4,linestyle='--')
plt.plot(intensity_along_axis(farred_norm_1,mask=mask_iso,axis=(1,2)),label='r=1',color='#CC3311',linewidth=4)
plt.plot(intensity_along_axis(farred_norm_048,mask=mask_iso,axis=(1,2)),label='r=0.48',color='#009988', linewidth=4)
plot_and_save(Path(folder) / f'S3f_1_plot.svg',xlabel='Depth (µm)',ylabel='Intensity in the plane \n (far-red channel) (A.U.)',yticks=[500,1000,1500,2000])

#S3f_2
fig = plt.figure(figsize=(10,7))
farred_norm_053 = normalize(array = farred_iso,ref_array = blue_iso,exponentiation_coeff=0.53,sigma=15,mask=mask_iso,labels=None,width=3)
farred_norm_043 = normalize(array = farred_iso,ref_array = blue_iso,exponentiation_coeff=0.43, sigma=15,mask=mask_iso,labels=None,width=3)
plt.plot(intensity_along_axis(farred_norm_043,mask=mask_iso,axis=(1,2)),label='r=0.43',color="#33BBEE",linewidth=4)
plt.plot(intensity_along_axis(farred_norm_048,mask=mask_iso,axis=(1,2)),label='r=0.48',color='#009988', linewidth=4)
plt.plot(intensity_along_axis(farred_norm_053,mask=mask_iso,axis=(1,2)),label='r=0.53',color='#0077BB',linewidth=4)
plot_and_save(Path(folder) / f'S3f_2_plot.svg',xlabel='Depth (µm)',ylabel='Intensity in the plane \n (far-red channel) (A.U.)',yticks=[500,1000,1500,2000])

#S3f_3
viewer=napari.Viewer()
viewer.add_image(farred_iso,colormap='gray_r',name='unnormalized')
viewer.add_image(farred_norm_1,colormap='gray_r',name='r=1')
viewer.add_image(farred_norm_043,colormap='gray_r',name='r=0.43')
viewer.add_image(farred_norm_048,colormap='gray_r',name='r=0.48')
viewer.add_image(farred_norm_053,colormap='gray_r',name='r=0.53')
for l in viewer.layers:
    l.data = np.transpose(l.data, (1, 0, 2))
viewer.grid.enabled = True
viewer.grid.shape = (2, 3)
viewer.grid.stride = -1
napari.run()

#S3g_1
fig = plt.figure(figsize=(10,7))
blue_norm_1 = normalize(array = blue_iso,ref_array = blue_iso, exponentiation_coeff=1, sigma=15,mask=mask_iso,labels=None,width=3)
plt.plot(intensity_along_axis(blue_iso,mask=mask_iso,axis=(1,2)),label='unnormalized signal',color='black',linewidth=4,linestyle='--')
plt.plot(intensity_along_axis(blue_norm_1,mask=mask_iso,axis=(1,2)),label='r=1',color='#CC3311',linewidth=4)
plot_and_save(Path(folder) / f'S3g_plot.svg',xlabel='Depth (µm)',ylabel='Intensity in the plane \n (blue channel) (A.U.)',yticks=[250,500,750,1000])

#S3g_2
viewer=napari.Viewer()
viewer.add_image(blue_iso,colormap='gray_r',name='unnormalized')
viewer.add_image(blue_norm_1,colormap='gray_r',name='r=1')
for l in viewer.layers:
    l.data = np.transpose(l.data, (1, 0, 2))
viewer.grid.enabled = True
viewer.grid.shape = (2, 1)
viewer.grid.stride = -1
napari.run()

#S3h
plt.plot(intensity_along_axis(farred_norm_048,mask=mask_iso,axis=(1,2)),label='along Z axis',color='#009988',linestyle='solid',linewidth=4)
plt.plot(intensity_along_axis(farred_norm_048,mask=mask_iso,axis=(2,0)),label='along Y axis',color='#009988',linestyle='dotted',linewidth=4)
plt.plot(intensity_along_axis(farred_norm_048,mask=mask_iso,axis=(0,1)),label='along X axis',color='#009988',linestyle = 'dashed',linewidth=4)
plot_and_save(Path(folder) / f'S3h_plot.svg',xlabel='Axis length(µm)',ylabel='Intensity in the plane \n (far-red channel) (A.U.)',yticks=[500,1000,1500,2000])
