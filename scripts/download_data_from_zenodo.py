import requests
import zipfile
from tqdm import tqdm
import pathlib
import os
    


if __name__ == '__main__':
    script_dir = pathlib.Path(__file__).parents[1] / 'data'
    print(script_dir)

    # url of the data
    url=requests.get('...', stream=True)

    # Sizes in bytes
    block_size = 1024

    # Create the zip file
    filename = "file.zip"

    with tqdm(total=0, unit='B', unit_scale=True) as pbar:
        with open(script_dir / filename, mode="wb") as localfile:
            for data in url.iter_content(block_size):
                pbar.update(len(data))
                localfile.write(data)

    with zipfile.ZipFile(script_dir / filename, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(script_dir, './'))

    # remove initial folder
    folder_path = script_dir / filename
    folder_path.unlink()
    