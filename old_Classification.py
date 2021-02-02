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
flavor_type = 0
flavor_img = []
img_dir = []
flavor = []
potato_file = os.listdir("/content/Potato_Chip_Classification/potato-chips")
!mkdir /content/Potato_Chip_Classification/potato_img
for flavor_file in potato_file:
    flavor_dir = os.listdir(f"/content/Potato_Chip_Classification/potato-chips/{flavor_file}")
    for potato_img in flavor_dir:
      img_dir.append(flavor_file)
      flavor_img.append(potato_img)
      flavor.append(flavor_type)
    !cp /content/Potato_Chip_Classification/potato-chips/$flavor_file/*.jpg /content/Potato_Chip_Classification/potato_img/
    flavor_type += 1
df = pd.DataFrame({'filename' : flavor_img, 'flavor' : img_dir, 'category' : flavor})
df.to_csv('/content/Potato_Chip_Classification/potato_img.csv')
