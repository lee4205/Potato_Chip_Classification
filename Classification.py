# --------------------------------------------------------------------------------------------
# 作業を始める前に、データセットが入ってるかを確認する
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# ライブラリーから必要なモジュールをインポートする
# --------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import random
import os
# --------------------------------------------------------------------------------------------
# 定数の定義
# --------------------------------------------------------------------------------------------
FAST_RUN = False
IMAGE_WIDTH=960
IMAGE_HEIGHT=1280
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
# --------------------------------------------------------------------------------------------
# 学習データの準備
# --------------------------------------------------------------------------------------------
# 現在ワークスペースのディレクトリを設定する
cwd = os.getcwd()
# 学習用データセットファイルの作成
!mkdir $cwd/dataset
# 2つのパラメータを設定する
filename = []
taste = []
# 各味のディレクトリをリスト化する
flavors = os.listdir(cwd + "/potato-chips")
for flavor in flavors:
    # 各味の画像ファイルをリスト化する
    images = os.listdir(cwd + f"/potato-chips/{flavor}")
    for image in images:
        # 各画像に情報を付ける
        taste.append(flavor)
        filename.append(image)
    # 各味の画像ファイルを学習用データセットファイルにコピーする
    !cp $cwd//potato-chips/$flavor/*.jpg $cwd/dataset
# pandasを持ちてデータフレームの作成
df = pd.DataFrame({'filename' : filename, 'taste' : taste})
