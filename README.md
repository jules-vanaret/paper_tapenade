# Requirements

Running the scripts requires setting up a proper Python3 environment. The following packages are required:
 - tapenade
 - napari (see [documentation](https://napari.org/dev/tutorials/fundamentals/installation.html))
 - tifffile
 - pandas
 - seaborn
 - jupyter (only to run the notebooks)
 - stardist (only to train a custom stardist model, see [documentation](https://stardist.net/))

They can be installed using [pip](https://pip.pypa.io/en/stable/getting-started/).

# Before running the scripts

Before running the main scripts (e.g to generate the figures), our datasets needs to be downloaded from Zenodo. 
To do so, we recommend running the script `download_data_from_zenodo.py`, which will automatically download the data and extract it in the `data` folder. 

# Running the scripts

After running the script `download_data_from_zenodo.py`, all scripts can be run without any modification. Scripts are located in folders named after the main figure they (partly) generate.


