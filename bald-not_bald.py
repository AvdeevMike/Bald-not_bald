Версия на проверку списка фотографий
-------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from google.colab import drive
import os

# Подключаем Google Drive
drive.mount('/content/drive')
data_path = '/content/drive/My Drive/HairHealth/'

# Определяем пути к папкам с фотографиями
train_dir = '/content/drive/My Drive/HairHealth/train'
test_dir = '/content/drive/My Drive/HairHealth/test'

# Создаем генераторы изображений для обучения и тестирования
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Загружаем обучающие и тестовые изображения с помощью генераторов
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['nb', 'bald']
)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['nb', 'bald']
)

# Создаем модель нейронной сети
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])

# Компилируем модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator))

# Загружаем изображение для классификациии
# Путь к папке с фотографиями
photos_dir = '/content/drive/My Drive/HairHealth/chek/'

# Получаем список файлов в папке
photo_files = os.listdir(photos_dir)

# Классифицируем каждую фотографию
for photo_file in photo_files:
    # Полный путь к фотографии
    photo_path = os.path.join(photos_dir, photo_file)

    # Загружаем и предобрабатываем фотографию
    img = image.load_img(photo_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255

    # Классифицируем фотографию
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print(photo_file, " - Лысый человек")
    else:
        print(photo_file, " - Не лысый человек")

     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Found 160 images belonging to 2 classes.
Found 40 images belonging to 2 classes.
Epoch 1/10
8/8 [==============================] - 26s 3s/step - loss: 0.7274 - accuracy: 0.5312 - val_loss: 0.6771 - val_accuracy: 0.5000
Epoch 2/10
8/8 [==============================] - 20s 3s/step - loss: 0.6467 - accuracy: 0.6687 - val_loss: 0.6177 - val_accuracy: 0.5000
Epoch 3/10
8/8 [==============================] - 20s 3s/step - loss: 0.5874 - accuracy: 0.7000 - val_loss: 0.4666 - val_accuracy: 0.8500
Epoch 4/10
8/8 [==============================] - 20s 2s/step - loss: 0.4659 - accuracy: 0.7937 - val_loss: 0.4405 - val_accuracy: 0.8500
Epoch 5/10
8/8 [==============================] - 23s 3s/step - loss: 0.2798 - accuracy: 0.8875 - val_loss: 0.3811 - val_accuracy: 0.8750
Epoch 6/10
8/8 [==============================] - 20s 2s/step - loss: 0.2109 - accuracy: 0.9187 - val_loss: 0.4397 - val_accuracy: 0.8750
Epoch 7/10
8/8 [==============================] - 20s 2s/step - loss: 0.3130 - accuracy: 0.8625 - val_loss: 0.3183 - val_accuracy: 0.8750
Epoch 8/10
8/8 [==============================] - 23s 3s/step - loss: 0.2370 - accuracy: 0.8750 - val_loss: 0.2799 - val_accuracy: 0.8750
Epoch 9/10
8/8 [==============================] - 22s 3s/step - loss: 0.1963 - accuracy: 0.9250 - val_loss: 0.5340 - val_accuracy: 0.8500
Epoch 10/10
8/8 [==============================] - 21s 3s/step - loss: 0.0982 - accuracy: 0.9688 - val_loss: 0.4131 - val_accuracy: 0.8500
1/1 [==============================] - 0s 138ms/step
bald002.jpg  - Лысый человек
1/1 [==============================] - 0s 33ms/step
bald003.jpg  - Лысый человек
1/1 [==============================] - 0s 36ms/step
bald004.jpg  - Лысый человек
1/1 [==============================] - 0s 37ms/step
bald001.jpg  - Лысый человек
1/1 [==============================] - 0s 46ms/step
bald005.jpg  - Лысый человек
1/1 [==============================] - 0s 38ms/step
bald100.jpg  - Не лысый человек
1/1 [==============================] - 0s 41ms/step
bald099.jpg  - Не лысый человек
1/1 [==============================] - 0s 38ms/step
nb106.jpg  - Не лысый человек
1/1 [==============================] - 0s 32ms/step
nb102.jpg  - Не лысый человек
1/1 [==============================] - 0s 35ms/step
nb103.jpg  - Не лысый человек
1/1 [==============================] - 0s 37ms/step
nb105.jpg  - Лысый человек
1/1 [==============================] - 0s 44ms/step
nb104.jpg  - Не лысый человек
1/1 [==============================] - 0s 39ms/step
nb108.jpg  - Не лысый человек
1/1 [==============================] - 0s 37ms/step
nb101.jpg  - Не лысый человек
1/1 [==============================] - 0s 46ms/step
nb107.jpg  - Не лысый человек
1/1 [==============================] - 0s 38ms/step
bald101.jpg  - Не лысый человек




----------------------------------------------------------------------------------------------------------------
Версия на проверку одной фотографии
----------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from google.colab import drive

# Подключаем Google Drive
drive.mount('/content/drive')
data_path = '/content/drive/My Drive/HairHealth/'

# Определяем пути к папкам с фотографиями
train_dir = '/content/drive/My Drive/HairHealth/train'
test_dir = '/content/drive/My Drive/HairHealth/test'

# Создаем генераторы изображений для обучения и тестирования
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Загружаем обучающие и тестовые изображения с помощью генераторов
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['nb', 'bald']
)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['nb', 'bald']
)

# Создаем модель нейронной сети
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])

# Компилируем модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator))

# Загружаем изображение для классификациии
img_path = '/content/drive/My Drive/HairHealth/nb108.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255

# Классифицируем изображение
prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("Лысый человек")
else:
    print("Не лысый человек")


     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Found 160 images belonging to 2 classes.
Found 40 images belonging to 2 classes.
Epoch 1/10
8/8 [==============================] - 22s 2s/step - loss: 0.8217 - accuracy: 0.5188 - val_loss: 0.6872 - val_accuracy: 0.5000
Epoch 2/10
8/8 [==============================] - 20s 2s/step - loss: 0.6869 - accuracy: 0.6062 - val_loss: 0.6560 - val_accuracy: 0.5000
Epoch 3/10
8/8 [==============================] - 19s 2s/step - loss: 0.6640 - accuracy: 0.5500 - val_loss: 0.5463 - val_accuracy: 0.8750
Epoch 4/10
8/8 [==============================] - 19s 2s/step - loss: 0.5578 - accuracy: 0.7250 - val_loss: 0.4780 - val_accuracy: 0.8000
Epoch 5/10
8/8 [==============================] - 19s 2s/step - loss: 0.4062 - accuracy: 0.8500 - val_loss: 0.6941 - val_accuracy: 0.7500
Epoch 6/10
8/8 [==============================] - 19s 2s/step - loss: 0.2904 - accuracy: 0.8875 - val_loss: 0.3332 - val_accuracy: 0.8500
Epoch 7/10
8/8 [==============================] - 19s 2s/step - loss: 0.2015 - accuracy: 0.9375 - val_loss: 0.3424 - val_accuracy: 0.9000
Epoch 8/10
8/8 [==============================] - 19s 2s/step - loss: 0.1154 - accuracy: 0.9500 - val_loss: 0.4791 - val_accuracy: 0.8750
Epoch 9/10
8/8 [==============================] - 19s 2s/step - loss: 0.0484 - accuracy: 0.9875 - val_loss: 0.5715 - val_accuracy: 0.8750
Epoch 10/10
8/8 [==============================] - 19s 2s/step - loss: 0.0251 - accuracy: 0.9937 - val_loss: 0.7762 - val_accuracy: 0.8250
1/1 [==============================] - 0s 148ms/step
Не лысый человек