import os
from PIL import Image
import numpy as np
import pandas as pd

def folders_to_csv(parent_folder_path, nb_folders, output_path):
    data = []
    for i in range(0,nb_folders):
        folder_path = f"{parent_folder_path}{i}"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path).convert("L")
            vector = np.array(img).flatten()
            vector = np.append(vector, i)
            data.append(vector)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

folders_to_csv("data/test/", 10, "data/test.csv")
folders_to_csv("data/train/", 10, "data/train.csv")


            
