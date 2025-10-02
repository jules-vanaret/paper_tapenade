import requests
import zipfile
from tqdm import tqdm
import pathlib


if __name__ == "__main__":
    script_dir = pathlib.Path(__file__).parents[1] / "data"
    print(script_dir)

    # url of the data
    url = requests.get(
        "https://zenodo.org/records/17249972/files/zenodo_tapenade_data.zip?download=1",
        stream=True,
    )

    # Sizes in bytes
    block_size = 1024

    # Create the zip file
    filename = "file.zip"

    with tqdm(
        unit="B",
        unit_scale=True,
        total=int(9.41e9),
        desc="Downloading data from Zenodo:",
    ) as pbar:
        with open(script_dir / filename, mode="wb") as localfile:
            for data in url.iter_content(block_size):
                pbar.update(len(data))
                localfile.write(data)

    with zipfile.ZipFile(script_dir / filename, "r") as zip_ref:
        for file in tqdm(
            zip_ref.namelist(),
            total=len(zip_ref.namelist()),
            desc="Extracting data from zip file:",
        ):
            zip_ref.extract(member=file, path=script_dir)

    # remove initial folder
    folder_path = script_dir / filename
    folder_path.unlink()
