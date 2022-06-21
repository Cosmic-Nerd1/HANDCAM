import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[16], cmap = plt.cm.binary)
plt.show()
print("\n\n*********************************The Number displayed in the image is " + str(y_train[16])+"*********************************\n\n")

model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())


model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test, y_test)
plt.imshow(x_test[17], cmap = plt.cm.binary)
plt.show()  
print("\n\n*********************************The Number displayed in the image is " + str(y_test[17])+"*********************************\n\n")
