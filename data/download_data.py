import urllib.request
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
save_path = "data/arrhythmia.data"

os.makedirs("data", exist_ok=True)
print("Downloading Arrhythmia dataset...")
urllib.request.urlretrieve(url, save_path)
print("Download complete! File saved at:", save_path)
