# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gitからプールしない場合
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# データセットをダウンロードする
!wget https://sagacentralstorage.blob.core.windows.net/dataset/potato-chips.zip

# zipファイルを解凍する
!unzip potato-chips.zip
# いらないファイルを削除する
!rm -r potato-chips.zip
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gitからプールする場合
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# レポシトリをクローンする
!git clone https://[username]:[passsword]@github.com/lee4205/Potato_Chip_Classification.git

# メールとユーザ名の設定
!git config --global user.email "[email@gmail.com]"
!git config --global user.name "[username]"

# ワークスペースに移動する
cd [Potato_Chip_Classificationのパス]

# masterブランチからプールする
!git pull origin master

# 作業ブランチを作成する
!git branch [ブランチ名]

# 作業ブランチに切り替える
!git checkout [ブランチ名]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# --------------------------------------------------------------------------------------------
# 作業を始める（データセットが入ってるかを確認する）
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

# 
FAST_RUN = False
IMAGE_WIDTH=960
IMAGE_HEIGHT=1280
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

#
flavor_type = 0
flavor_img = []
img_dir = []
flavor = []
potato_file = os.listdir("/content/Potato_Chip_Classification/potato-chips")
print(potato_file)
!mkdir /content/Potato_Chip_Classification/potato_img
for flavor_file in potato_file:
    flavor_dir = os.listdir(f"/content/Potato_Chip_Classification/potato-chips/{flavor_file}")
    for potato_img in flavor_dir:
      img_dir.append(flavor_file)
      flavor_img.append(potato_img)
      flavor.append(flavor_type)
    !cp /content/Potato_Chip_Classification/potato-chips/$flavor_file/*.jpg /content/Potato_Chip_Classification/potato_img/
    flavor_type += 1
filenames = os.listdir("/content/Potato_Chip_Classification/potato_img")
df = pd.DataFrame({'filename' : flavor_img, 'flavor' : img_dir, 'category' : flavor})
df.to_csv('/content/Potato_Chip_Classification/potato_img.csv')
