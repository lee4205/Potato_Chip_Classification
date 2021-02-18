import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

image_width = 1280 // 10
image_height = 960 // 10
image_size = (image_width, image_height)
image_channel = 1

# ファイルパスの取得
cwd = os.getcwd()
flavors = os.listdir(cwd + "/potato-chips")

# dataframe用のpixel列名
pixels = []
pixel = 0
for x in range(image_width * image_height):
    pixel += 1
    pixels.append('pixel' + str(pixel))

# csvの入力
with open("potato_chips.csv", 'a') as c:
    csv_input = csv.writer(c)
    header = ["image", "flavor"]
    header.extend(pixels)
    csv_input.writerow(header)

    for flavor in flavors:
        print("loading image from " + flavor + " ...")
        images = os.listdir(cwd + f"/potato-chips/{flavor}")
        for image in images:
            rgb_image = Image.open(cwd + f"/potato-chips/{flavor}/" + image)
            grey_image = rgb_image.convert('L').resize((image_width, image_height))
            pixel_data = np.asarray(grey_image.getdata(), dtype=np.int).reshape((grey_image.size[1], grey_image.size[0]))
            pixel_data = pixel_data.flatten()
            image_data = [image, flavors.index(flavor)]
            image_data.extend(pixel_data)
            csv_input.writerow(image_data)

# csvの読み込み
df = pd.read_csv(cwd + "/potato_chips.csv")

# 学習、検証とテスト用のデータ準備
train_df, validate_df = train_test_split(df, train_size=0.6, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# デバッグ用
# train_df
# validate_df
# test_df
# plt.figure(figsize=(15, 15))
# sns.set_style("darkgrid")
# sns.countplot(train_df['flavor'])
# sns.countplot(validate_df['flavor'])
# sns.countplot(test_df['flavor'])


y_train = train_df['flavor']
y_validate = validate_df['flavor']
y_test = test_df['flavor']
y = test_df['flavor']

del train_df['image'], train_df['flavor']
del validate_df['image'], validate_df['flavor']
del test_df['image'], test_df['flavor']

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_validate = label_binarizer.fit_transform(y_validate)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_validate = validate_df.values
x_test = test_df.values

x_train = x_train / 255
x_validate = x_validate / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, image_height, image_width, 1)
x_validate = x_validate.reshape(-1, image_height, image_width, 1)
x_test = x_test.reshape(-1, image_height, image_width, 1)

# デバッグ用
# f, ax = plt.subplots(3, 3)
# f.set_size_inches(10, 10)
# k = 0
# for i in range(3):
#     for j in range(3):
#         ax[i, j].imshow(x_train[k].reshape(image_height, image_width), cmap="gray")
#         k += 1
#     plt.tight_layout()

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False)
datagen.fit(x_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(image_width, image_height, image_channel)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(datagen.flow(x_train, y_train, batch_size=5),
                    epochs=500,
                    validation_data=(x_validate, y_validate),
                    callbacks=[learning_rate_reduction])

print("Accuracy of the model : ", model.evaluate(x_validate, y_validate)[1] * 100, "%")

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(range(1, len(history.history["accuracy"]) + 1), history.history["accuracy"])
axs[0].plot(range(1, len(history.history["val_accuracy"]) + 1), history.history["val_accuracy"])
axs[0].set_title("Model Accuracy")
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("Epoch")
axs[0].set_xticks(np.arange(1, len(history.history["accuracy"]) + 1), len(history.history["accuracy"]) / 10)
axs[0].legend(["train", "val"], loc="best")
axs[1].plot(range(1, len(history.history["loss"]) + 1), history.history["loss"])
axs[1].plot(range(1, len(history.history["val_loss"]) + 1), history.history["val_loss"])
axs[1].set_title("Model Loss")
axs[1].set_ylabel("Loss")
axs[1].set_xlabel("Epoch")
axs[1].set_xticks(np.arange(1, len(history.history["loss"]) + 1), len(history.history["loss"]) / 10)
axs[1].legend(["train", "val"], loc="best")
plt.show()

predictions = model.predict_classes(x_test)
for i in range(len(predictions)):
    if predictions[i] >= 9:
        predictions[i] += 1

# デバッグ用
# print(predictions)

classes = ["Class " + str(i) for i in range(8) if i != 0]
print(classification_report(y, predictions, target_names=classes))

cm = confusion_matrix(y, predictions)
cm = pd.DataFrame(cm, index=[i for i in range(8) if i != 0], columns=[i for i in range(8) if i != 0])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='')

correct = (predictions == y).to_numpy().nonzero()[0]
i = 0
plt.figure(figsize=(10, 10))
for c in correct[:9]:
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[c].reshape(image_height, image_width), cmap="gray", interpolation='none')
    plt.title("Predicted Class {}, Actual Class {}".format(flavors[predictions[c]], flavors[y[c]]))
    i += 1