import os

import imquality.brisque as brisque
import PIL.Image

def get_quality(path):
    img = PIL.Image.open(path)
    b = brisque.score(img)
    print(b)

for file in os.listdir("/Users/holzmano/Downloads"):
    if file.endswith(".jpg"):
        path = os.path.join("/Users/holzmano/Downloads",file)
        print(file)
        get_quality(path)
