import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=(x_train.shape[1:]),  return_sequences=True)) #i wanna sequences because there will be another LSTM
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.CuDNNLSTM(128 )) #i wanna sequences because there will be another LSTM
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5),
              metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)


