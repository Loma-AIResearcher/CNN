import os
import urllib.request
import tarfile
import socket
from tqdm import tqdm

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
data_dir = "./data"
tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
expected_size = 171783168

def download_file(url, path):
    print("Downloading dataset...")
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.info().get('Content-Length').strip())
            block_size = 1024  
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(path, 'wb') as file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    file.write(buffer)
                    progress_bar.update(len(buffer))
            progress_bar.close()
        print("Download complete.")
    except (urllib.error.URLError, socket.timeout) as e:
        print(f"Network error: {e}. Please check your internet connection and try again.")
        if os.path.exists(path):
            os.remove(path)
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(path):
            os.remove(path)
        raise

def is_download_complete(path, expected_size):
    return os.path.exists(path) and os.path.getsize(path) == expected_size

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not is_download_complete(tar_path, expected_size):
    if os.path.exists(tar_path):
        os.remove(tar_path)
    try:
        download_file(url, tar_path)
    except Exception as e:
        print("Download failed. Exiting.")
        exit(1)

try:
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
        print("Dataset extracted.")
except tarfile.TarError as e:
    print(f"Error extracting tar file: {e}")
    if os.path.exists(tar_path):
        os.remove(tar_path)
