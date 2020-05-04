import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = 50


test_data = np.load('test_data.npy', allow_pickle=True)


model = tf.keras.models.load_model('3-conv-32-layersize-1-dense-1588071568')
fig = plt.figure()

for num, data in enumerate(test_data[-12:]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    img_data = img_data/255
    #print(img_data)
    data = img_data.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])
    print(model_out)
#     if np.argmax(model_out)==1:
#         str_label = 'how how doge'
#     else:
#         str_label = 'miau mia kitku'
#     y.imshow(orig, cmap='gray')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_xaxis().set_visible(False)
# plt.show()