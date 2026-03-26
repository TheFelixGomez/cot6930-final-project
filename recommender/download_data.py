import os
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-1m.zip")
    extract_path = os.path.join(DATA_DIR, "ml-1m")

    if os.path.exists(extract_path):
        print("Data already exists.")
        return

    print("Downloading MovieLens 1M...")
    urllib.request.urlretrieve(
        "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        zip_path
    )
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print("Done.")

if __name__ == "__main__":
    download()