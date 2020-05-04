import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf

EPOCHS = 10
BATCH_SZIE =64
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'
MODEL_NAME = f'{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'


def classify(current, future):
    if float(future)> float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col!= 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])#not take target
        if len(prev_days)==SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target ==1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


main_df = pd.DataFrame()

ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={'close': f'{ratio}_close', 'volume': f'{ratio}_volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[[f'{ratio}_close', f'{ratio}_volume']]
    if len(main_df)==0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
print(main_df['target'].head())

times = sorted(main_df.index.values)
last5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last5pct)]
main_df = main_df[(main_df.index < last5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)
print(len(train_x), len(validation_x))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=(train_x.shape[1:]),  return_sequences=True)) #i wanna sequences because there will be another LSTM
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=(train_x.shape[1:]),  return_sequences=True)) #i wanna sequences because there will be another LSTM
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tf_board = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{MODEL_NAME}')

filepath = 'RNN_Final-{epoch:02d}-{val_acc:.3f}'
checkpoint = tf.keras.callbacks.ModelCheckpoint('models/{}.model'.format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
history = model.fit(
    train_x,
    train_y,
    batch_size=BATCH_SZIE,
    validation_data=(validation_x, validation_y),
    callbacks=[tf_board, checkpoint],
    epochs=EPOCHS


)

