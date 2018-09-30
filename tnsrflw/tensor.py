import tensorflow as tf
from tensorflow import keras

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist # загружаем датасет mnist с картинками цифр от 0 до 9 размера 28x28

(x_train, y_train),(x_test, y_test) = mnist.load_data() # открываем датасет в указанные переменные
x_train = tf.keras.utils.normalize(x_train, axis=1) # в x_train и x_test данные хранились в диапазлне [0;255]
x_test = tf.keras.utils.normalize(x_test, axis=1) # нормалихуем их до [0;1] с целью упрощения
# строим модель NN ( используем модель Sequential) и определяем слои NN
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten()) # первый ( input ) слой

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))   # второй слой. Выбираем Dense с функцией активации relu
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))   # и указываем первым параметром количество входных данных(128)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # последний слой. Количество нейронов = количеству classifictaion
                                                               # ( количество цифр в диапазоне [0;9])
# перед тем как начать обучать сеть, необходимо указать то, как именно она должна реагировать на данные
# optimizer = 'adam' - то, как NN должна реагировать на данные, loss - определяем фукнцию потерь, которую будем стараться
# минимизировать, metrics - контроль обучения ( правильные предсказания)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# "фидим" модель необходимыми данными и указываем количество "прокруток"
model.fit(x_train, y_train, epochs=5)
model.save('epic_num_reader.model') # сохраняем модель
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test) # вызываем функцию с параметром тестовых данных, что,s определить предсказание

print(np.argmax(predictions[2])) # берем максимум от значения, чтобы получить результат с наибольшей вероятностью ( он же prediction)
plt.imshow(x_test[2],cmap=plt.cm.binary) # проверяем
plt.show()