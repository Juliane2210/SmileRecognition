

import shutil
import os
import pandas as pd


m_rootFolder = ".\\data\\SMILE PLUS Training Set\\SMILE PLUS Training Set\\"
m_csvData = m_rootFolder + "annotations.csv"


m_outputFolder = ".\\data\\testimages"

# Define column names for data
m_columns = ["FILENAME", "EMOTION"]


def loadCSVData():
    data = pd.read_csv(m_csvData, sep=',', header=None, names=m_columns)
    # Strip leading and trailing spaces from all fields
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return data


def copy_file_to_folder(file_name, output_folder):
    # Ensure that the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Check if the file exists
    if os.path.isfile(file_name):
        # Extract the file name from the file path
        _, file_base_name = os.path.split(file_name)
        # Generate the output file path
        output_file_path = os.path.join(output_folder, file_base_name)

        # Copy the file to the output folder
        shutil.copy(file_name, output_file_path)
        print(f"File '{file_name}' copied to '{output_file_path}'")
    else:
        print(f"Error: File '{file_name}' does not exist")


if __name__ == "__main__":
    data = loadCSVData()

    # Iterate through rows
    for index, row in data.iterrows():
        imageFile = row["FILENAME"]
        emotion = row["EMOTION"]

        fullPathImage = m_rootFolder + imageFile

        classificationFolder = "\\neutral\\"
        if (emotion == "happy"):
            classificationFolder = "\\happy\\"

        copy_file_to_folder(
            fullPathImage, m_outputFolder + classificationFolder)
