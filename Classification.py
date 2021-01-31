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
