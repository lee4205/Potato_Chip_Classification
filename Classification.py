# --------------------------------------------------------------------------------------------
# 作業を始める前に、データセットが入ってるかを確認する
# --------------------------------------------------------------------------------------------
# ライブラリーから必要なモジュールをインポートする
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import random
import os

# 定数の定義
FAST_RUN = False
IMAGE_WIDTH=960
IMAGE_HEIGHT=1280
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

# 学習データの準備
filename = []
taste = []
flavors = os.listdir("/content/Potato_Chip_Classification/potato-chips")
for flavor in flavors:
    images = os.listdir(f"/content/Potato_Chip_Classification/potato-chips/{flavor}")
    for image in images:
      taste.append(flavor)
      filename.append(image)
    !cp /content/Potato_Chip_Classification/potato-chips/$flavor/*.jpg /content/Potato_Chip_Classification/dataset/
df = pd.DataFrame({'filename' : filename, 'taste' : taste})
df.to_csv('/content/Potato_Chip_Classification/potato_img.csv')