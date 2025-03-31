import os
from PIL import Image
import numpy as np
import pandas as pd


data = []
for i in range(0,10):
    folder_path = f"data/test/{i}"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = Image.open(file_path).convert("L")
        vector = np.array(img).flatten()
        vector = np.append(vector, i)
        data.append(vector)

df = pd.DataFrame(data)
df.to_csv("data/mnist.csv")
        
