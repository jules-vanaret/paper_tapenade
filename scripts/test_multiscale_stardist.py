import numpy as np
import tifffile
from tifffile import TiffFile
# import napari
import matplotlib.pyplot as plt

from stardist.models import StarDist3D
from tqdm import tqdm
from organoid.preprocessing.preprocessing import local_image_normalization
from organoid.preprocessing.preprocessing import make_array_isotropic
from skimage.transform import resize




def predict_lennedist(array, zoom_factors, normalize=False):

    is_temporal = array.ndim == 4

    if normalize:
        if is_temporal:
            perc_low, perc_high = np.percentile(array, (1, 99), axis=(1, 2, 3))
            array = (array - perc_low[:, None, None, None]) / (perc_high - perc_low)[:, None, None, None]
        else:
            perc_low, perc_high = np.percentile(array, (1, 99))
            array = (array - perc_low) / (perc_high - perc_low)
        
        array = np.clip(array, 0, 1)

    # isotropize to reach target object size
    if not all(zf == 1 for zf in zoom_factors):
        array = make_array_isotropic(image=array, zoom_factors=zoom_factors, order=1)
    print(array.min(), array.max())

    model = StarDist3D(None, name='lennedist_3d_grid222_rays64', basedir='/data1/lennedist_data/models')
    model.config.use_gpu=True

    if is_temporal:
        labels = np.zeros(array.shape, dtype=np.uint16)

        for index_t, im in enumerate(tqdm(array)):
            labs, _ = model.predict_instances(im, n_tiles=model._guess_n_tiles(im))
            labels[index_t] = labs

    else:
        labels, _ = model.predict_instances(array, n_tiles=model._guess_n_tiles(array))

    # stretch by a factor of 2 in all dims to account for binning, plus 
    # initial zoom_factors
    if not all(zf == 1 for zf in zoom_factors):
        second_zoom_factors = [1/zf for zf in zoom_factors]
        labels = make_array_isotropic(labels, zoom_factors=second_zoom_factors, order=0)

    return labels

path_to_data = '/data1/project_egg/raw/fusion3'
data = tifffile.imread(f'{path_to_data}/rescaled_normalized_t30.tif')
shape = data.shape

# rescale_factors = [1/3, 1/2.5, 1/2, 1/1.5, 1, 1.5, 2, 2.5, 3]
rescale_factors = np.logspace(np.log10(1/2), np.log10(2), 9)


all_labels = np.zeros((len(rescale_factors), *shape), dtype=np.uint16)



for i,f in enumerate(rescale_factors):

    new_shape = [int(s*f) for s in shape]
    print(shape, new_shape)

    image = resize(data, new_shape, order=1, preserve_range=True)

    labels_norm = predict_lennedist(image, (1,1,1), normalize=False)

    labels_norm = resize(labels_norm, shape, order=0, preserve_range=True)

    all_labels[i] = labels_norm


tifffile.imwrite(f'{path_to_data}/multiscale_stardist_labels2.tif', all_labels)

