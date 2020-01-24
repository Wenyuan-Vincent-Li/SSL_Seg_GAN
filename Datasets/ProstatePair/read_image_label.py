from PIL import Image
import numpy as np
label = Image.open("train_label/0000_sementic.png")
label = np.array(label)
print(np.unique(label))