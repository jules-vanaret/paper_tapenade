import numpy as np
import napari


_,img= np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))

viewer = napari.Viewer()

viewer.add_image(img, colormap='reds', gamma=1.5, translate=[0,10])
viewer.add_image(img, colormap='reds')

viewer.grid.enabled = True
viewer.grid.shape = (1, 2) 

napari.run()
