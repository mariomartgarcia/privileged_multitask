import os
import requests
import zipfile
import io
import pandas as pd
from scipy.io import arff
from river.datasets import Phishing
import re 

# Define the list of file links and their desired output names
file_links = {
    "https://archive.ics.uci.edu/static/public/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip": "obesity.csv",
    "https://archive.ics.uci.edu/static/public/186/wine+quality.zip": "wine_uci_186.csv",
    "https://www.openml.org/data/download/22044302/diabetes.arff": "diabetes.arff",  # Will convert to CSV later
    "https://www.openml.org/data/download/1592281/php8Mz7BG": "phoneme.arff",
    "https://www.openml.org/data/download/53929/mozilla4.arff": "mozilla4.arff",
    "https://www.openml.org/data/download/53254/abalone.arff": "abalone.arff",
    "https://www.openml.org/data/download/53381/wind.arff": "wind.arff"
}

# Folder where the files will be saved
output_folder = "data"

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def convert_arff_to_csv(arff_path, csv_path):
    print(f"Converting {arff_path} to CSV...")
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"File saved as {csv_path}")

def download_file(url, new_name, folder_path):
    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    file_path = os.path.join(folder_path, new_name)
    
    with open(file_path, 'wb') as file:
        file.write(response.content)
    
    # Convert only the diabetes dataset from .arff to .csv
    if new_name == "diabetes.arff":
        csv_path = file_path.replace(".arff", ".csv")
        convert_arff_to_csv(file_path, csv_path)
        
        # Delete the .arff file after conversion
        os.remove(file_path)
        print(f"Deleted {file_path} after conversion to CSV.")
        
        # Process the diabetes dataset and clean the 'Outcome' column
        df = pd.read_csv(csv_path)
        df['Outcome'] = df['Outcome'].apply(lambda x: re.sub(r"b'(\d+)'", r"\1", str(x)))
        df['Outcome'] = df['Outcome'].astype('int64')
        df.to_csv(csv_path, index=False)

    # Special case for obesity dataset (handling zip)
    elif "obesity.csv" in new_name:
        obesity_zip_url = url
        response = requests.get(obesity_zip_url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_contents = zip_file.namelist()
            obesity_file = "ObesityDataSet_raw_and_data_sinthetic.csv"
            if obesity_file in zip_contents:
                print(f"Extracting {obesity_file} and renaming it to {new_name}...")
                with zip_file.open(obesity_file) as file:
                    df = pd.read_csv(file)
                    output_path = os.path.join(folder_path, new_name)
                    df.to_csv(output_path, index=False)
                    print(f"Obesity dataset saved as {output_path}")
            else:
                print(f"Error: {obesity_file} not found in ZIP.")
                    
    # Special case for wine dataset (handling zip and merging CSVs)
    elif "wine_uci_186.csv" in new_name:
        wine_zip_url = url
        response = requests.get(wine_zip_url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_contents = zip_file.namelist()
            # Look for CSV files in the ZIP, excluding any files with "names" in the name
            csv_files = [f for f in zip_contents if f.endswith(".csv") and "names" not in f]
            if not csv_files:
                print(f"Error: No CSV files found in {wine_zip_url}")
                return
            
            # List to store DataFrames of each CSV
            data_frames = []
            for csv_file in csv_files:
                print(f"Extracting and reading {csv_file}...")
                with zip_file.open(csv_file) as file:
                    df = pd.read_csv(file, sep=";")  # UCI wine dataset uses ";"
                    data_frames.append(df)
            
            # Merge all CSV files into one DataFrame
            merged_data = pd.concat(data_frames, ignore_index=True)
            
            # Transform the 'quality' column into the new column '0'
            merged_data['0'] = merged_data['quality'].apply(lambda x: 1 if x > 5 else 0)
            
            # Drop the 'quality' column
            merged_data = merged_data.drop(columns=['quality'])
            
            # Save the merged CSV
            output_path = os.path.join(folder_path, new_name)
            merged_data.to_csv(output_path)
            print(f"Transformed and merged file saved as {output_path}")
        
    else:
        # Handle other files
        if new_name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_contents = zip_file.namelist()
                if new_name in zip_contents:
                    print(f"Extracting {new_name}...")
                    zip_file.extract(new_name, folder_path)
                    print(f"File saved as {os.path.join(folder_path, new_name)}")
                else:
                    print(f"Error: {new_name} not found in ZIP.")
        else:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File saved as {file_path}")

# Function to download and process the Phishing dataset from River
def download_phishing_dataset(folder_path):
    print("Downloading Phishing dataset from River...")
    
    dataset = Phishing()
    X = pd.DataFrame()
    y = []

    for xx, yy in dataset.take(5000):  # Take up to 5000 samples
        X = pd.concat([X, pd.DataFrame([xx])], ignore_index=True)
        y.append(yy)

    y = pd.Series(y) * 1  # Convert labels to 0 and 1
    df = pd.concat([X, y], axis=1)

    output_path = os.path.join(folder_path, "phishing.csv")
    df.to_csv(output_path)
    print(f"Phishing dataset saved as {output_path}")

# Create the folder if it does not exist
create_folder(output_folder)

# Download and process the files
for url, new_name in file_links.items():
    download_file(url, new_name, output_folder)

# Download the Phishing dataset from River
download_phishing_dataset(output_folder)

print("All downloads completed.")
