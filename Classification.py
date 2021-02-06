# --------------------------------------------------------------------------------------------
# 作業を始める前に、データセットが入ってるかを確認する
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# モジュールのインポート
# --------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb
import subprocess
import random
import os
# デバッグ用
pd.set_option('display.max_rows', None)
# --------------------------------------------------------------------------------------------
# 定数の定義
# --------------------------------------------------------------------------------------------
FAST_RUN = False
IMAGE_WIDTH = 480 #960
IMAGE_HEIGHT = 640 #1280
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
# --------------------------------------------------------------------------------------------
# データセット準備
# --------------------------------------------------------------------------------------------
# 現在ワークスペースのディレクトリを設定する
cwd = os.getcwd()
# 学習用データセットフォルダの作成
# !mkdir $cwd/dataset
subprocess.run(["mkdir " + cwd + "/dataset"], shell=True)
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
    # !cp $cwd/potato-chips/$flavor/*.jpg $cwd/dataset
    subprocess.run(["cp " + cwd + "/potato-chips/" + flavor + "/*.jpg " + cwd + "/dataset"], shell=True)
# pandasを持ちてデータフレームの作成
df = pd.DataFrame({"filename" : filename, "taste" : taste})
# デバッグ用
print(df)
print(df.shape)
df['taste'].value_counts().plot.bar()
# --------------------------------------------------------------------------------------------
# 解析オプション
# --------------------------------------------------------------------------------------------
# 味の選択
analyse = input("以下の味を1つ選択して入力する\n\
consomme-punch, kyusyu-shoyu, norishio, norishio-punch, shiawase-butter, shoyu-mayo, usushio\n\
選択した味：")
# 選択した味以外、全部0に与える
flavor_data = dict.fromkeys(flavors, 0)
flavor_data.update({analyse: 1})
# データフレームのデータを書き換える
df["taste"] = df["taste"].replace(flavor_data) 
# デバッグ用
print(df)
print(df.shape)
df['taste'].value_counts().plot.bar()
# --------------------------------------------------------------------------------------------
# モデル作成
# --------------------------------------------------------------------------------------------
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# モデルをまとめる
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# モデルの詳細
model.summary()
# --------------------------------------------------------------------------------------------
# 過学習の場合
# --------------------------------------------------------------------------------------------
# 過学習を防ぐために，10エポック後に学習を停止し，val_lossの値を減らさないようにする
earlystop = EarlyStopping(patience=10)
# 関数の呼び出す
callbacks = [earlystop]
# --------------------------------------------------------------------------------------------
# 学習と検証用のデータ準備
# --------------------------------------------------------------------------------------------
df["taste"] = df["taste"].replace({0: "others", 1: analyse}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=25
# デバッグ用
train_df['taste'].value_counts().plot.bar()
validate_df['taste'].value_counts().plot.bar()
# --------------------------------------------------------------------------------------------
# 学習の仕組み
# --------------------------------------------------------------------------------------------
# 学習用データを生成する
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)
# 学習のやり方を定義する
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    cwd + "/dataset/", 
    x_col="filename",
    y_col="taste",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=batch_size)
# --------------------------------------------------------------------------------------------
# 検証の仕組み
# --------------------------------------------------------------------------------------------
# 検証用データを生成する
validation_datagen = ImageDataGenerator(rescale=1./255)
# 検証のやり方を定義する
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    cwd + "/dataset/", 
    x_col="filename",
    y_col="taste",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=batch_size)
# --------------------------------------------------------------------------------------------
# モデルの適合
# --------------------------------------------------------------------------------------------
# エポックを指定する
epochs=10 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks)
# モデルを保存する
model.save_weights("model.h5")
# 学習結果を出す
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))
ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------
# サンプル用
# --------------------------------------------------------------------------------------------
# 写真の試し
sample = random.choice(filename)
sample_image = load_img(cwd + "/dataset/" + sample)
plt.imshow(sample_image)
# 仕組みの試し
sample_df = train_df.sample(n=1).reset_index(drop=True)
sample_generator = train_datagen.flow_from_dataframe(
    sample_df, 
    cwd + "/dataset/", 
    x_col='filename',
    y_col='taste',
    target_size=IMAGE_SIZE,
    class_mode="categorical")
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in sample_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
