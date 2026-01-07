import pandas as pd
import requests
import os
import time
from tqdm import tqdm  # Progress bar

# --- CONFIGURATION ---
API_KEY = "pk.eyJ1Ijoicml0ZXNoaWl0ciIsImEiOiJjbWp6bjYydWI2YjZkM2ZzNWRnZWZ4bzRmIn0.6Jh2ozJzSwCr-I3jihYI0w" 
INPUT_FILE = "C:/Users/singh/Codes/CDC_PROJECT/gemini/test.csv"  # 👈 changed from train_original.csv
OUTPUT_DIR = "satellite_images_test"  # 👈 separate folder for test images
BASE_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image(lat, lon, house_id):
    filename = f"{house_id}.jpg"
    file_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(file_path):
        return True

    url = f"{BASE_URL}/{lon},{lat},17,0,0/600x600?access_token={API_KEY}"

    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"Error {response.status_code} for House ID {house_id}")
            return False
    except Exception as e:
        print(f"Connection failed for House ID {house_id}: {e}")
        return False

def main():
    print(f"Reading {INPUT_FILE}...")

    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'.")
        return

    # Drop rows missing required fields
    df = df.dropna(subset=["lat", "long", "id"])

    print(f"Dataset loaded. Processing ALL {len(df)} rows from test.csv.")

    iterator = zip(df['lat'], df['long'], df['id'])

    print("Starting download...")

    for lat, lon, house_id in tqdm(iterator, total=len(df)):
        success = download_image(lat, lon, str(house_id))
        if success:
            time.sleep(0.15)

    print("\n------------------------------------------")
    print(f"Done! Test images saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
