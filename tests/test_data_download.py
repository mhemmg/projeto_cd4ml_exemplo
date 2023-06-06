import os

from src.data.download_dataset import Download

data_raw_path = 'data/raw'

def test_download_dataset():

    download = Download(destination_path = data_raw_path)
    response = download.download()
    assert os.path.exists(response[0]) == True




