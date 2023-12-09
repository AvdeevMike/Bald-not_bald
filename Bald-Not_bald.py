

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


--------------------------------------------------------------------------------------------------------------
Вариант со списком фото
---------------------------------------------------------------------------------------------------------------

# Путь к папке с фотографиями
photos_dir = '/content/drive/My Drive/chek/'

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
