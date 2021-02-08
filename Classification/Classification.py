import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import random
import os

pd.set_option("display.max_rows", None)

FAST_RUN = False
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 160
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

cwd = os.getcwd()
subprocess.run(["mkdir " + cwd + "/dataset"], shell=True)

filename = []
taste = []
flavors = os.listdir(cwd + "/potato-chips")
for flavor in flavors:
    images = os.listdir(cwd + f"/potato-chips/{flavor}")
    for image in images:
        taste.append(flavor)
        filename.append(image)
    subprocess.run(["cp " + cwd + "/potato-chips/" + flavor + "/*.jpg " + cwd + "/dataset"], shell=True)
df = pd.DataFrame({"filename": filename, "taste": taste})

# ～～～デバッグ～～～
print(df)
print(df.shape)
df["taste"].value_counts().plot.bar()
# ～～～～～～～～～～

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# ～～～デバッグ～～～
model.summary()
# ～～～～～～～～～～

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy",
                                            patience=2,
                                            verbose=1,
                                            factor=0.3,
                                            min_lr=0.0001)
callbacks = [earlystop, learning_rate_reduction]

train_df, validate_df = train_test_split(df, train_size=0.6, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ～～～デバッグ～～～
train_df["taste"].value_counts().plot.bar()
validate_df["taste"].value_counts().plot.bar()
test_df["taste"].value_counts().plot.bar()
# ～～～～～～～～～～

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
total_test = test_df.shape[0]
batch_size = 15

analyse = input("以下の味を1つ選択して入力する\n\
consomme-punch, kyusyu-shoyu, norishio, norishio-punch, shiawase-butter, shoyu-mayo, usushio\n\
選択した味：")

new_taste = dict.fromkeys(flavors, 0)
new_taste.update({analyse: 1})

train_df["taste"] = train_df["taste"].replace(new_taste)
train_df["taste"] = train_df["taste"].replace({0: "others", 1: analyse})
validate_df["taste"] = validate_df["taste"].replace(new_taste)
validate_df["taste"] = validate_df["taste"].replace({0: "others", 1: analyse})
test_df["taste"] = test_df["taste"].replace(new_taste)
test_df["taste"] = test_df["taste"].replace({0: "others", 1: analyse})

# ～～～デバッグ～～～
train_df["taste"].value_counts().plot.bar()
validate_df["taste"].value_counts().plot.bar()
test_df["taste"].value_counts().plot.bar()
# ～～～～～～～～～～

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    cwd + "/dataset/",
    x_col="filename",
    y_col="taste",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=batch_size)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    cwd + "/dataset/",
    x_col="filename",
    y_col="taste",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    batch_size=batch_size)

epochs = 10 if FAST_RUN else 50
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks)
model.save_weights("model.h5")

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

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    cwd + "/dataset/",
    x_col="filename",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=True)

predict = model.predict(test_generator, steps=np.ceil(total_test / batch_size))
test_df["taste"] = np.argmax(predict, axis=-1)
label_map = dict((v, k) for k, v in train_generator.class_indices.items())
test_df["taste"] = test_df["taste"].replace(label_map)
test_df["taste"].value_counts().plot.bar()

sample = random.choice(filename)
sample_image = load_img(cwd + "/dataset/" + sample)
print(sample)
plt.imshow(sample_image)

sample_df = train_df.sample(n=1).reset_index(drop=True)
sample_generator = train_datagen.flow_from_dataframe(
    sample_df,
    cwd + "/dataset/",
    x_col="filename",
    y_col="taste",
    target_size=IMAGE_SIZE,
    class_mode="categorical")
plt.figure(figsize=(12, 12))
for i in range(0, 9):
    plt.subplot(3, 3, i + 1)
    for X_batch, Y_batch in sample_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

sample_test = test_df.head(25)
sample_test.head()
plt.figure(figsize=(24, 24))
for index, row in sample_test.iterrows():
    filename = row["filename"]
    category = row["taste"]
    img = load_img(cwd + "/dataset/" + filename, target_size=IMAGE_SIZE)
    plt.subplot(5, 5, index + 1)
    plt.imshow(img)
    plt.xlabel("{}".format(category) + "(" + filename + ")")
plt.tight_layout()
plt.show()
