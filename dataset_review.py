# verisetimizi inceledik ve kabaca algoritmamızı olusturduk

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

images_dir = 'images'
images_path = os.listdir(images_dir)

for image_name in images_path:
    image_path = os.path.join(images_dir, image_name) # tam dosya yolunu oluşturduk
    image = cv2.imread(image_path)  # resimleri yükledik
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR'den RGB'ye dönüştür
    image = cv2.resize(image,(500,500)) #resimleri yeniden boyutlandırdık
    plt.imshow(image) # görüntüyü gösterelim
    plt.show()