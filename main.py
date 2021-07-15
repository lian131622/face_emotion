from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from model import conv2D


from helper import var_info

feature_raw = np.load('feature_raw.npy')
label = np.load('label.npy')
feature_edge = np.load('feature_edge.npy')
label_edge = np.load('label_edge.npy')


total_feature = len(feature_raw)
total_label = len(label)

if total_label != total_feature:
    raise Exception('label and feature not pair', total_label, total_feature)

X = np.squeeze(feature_raw)
X = np.expand_dims(X, 3)
Y = np.vstack(label)

print('X的特征尺寸是：', X.shape)
print('Y的特征尺寸是：', Y.shape)

# k 折
nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=202)
input_dim = (48, 48, 1)
i = 0
history = []
for train_index, valid_index in kf.split(X, Y):
    print('\nFold {}'.format(i + 1))
    train_x, val_x = X[train_index], X[valid_index]
    train_y, val_y = Y[train_index], Y[valid_index]
    train_x = train_x.reshape(-1, 48, 48, 1)
    val_x = val_x.reshape(-1, 48, 48, 1)
    train_y = to_categorical(train_y)
    val_y = to_categorical(val_y)
    model = conv2D(input_dim, 7)
    h = model.fit(train_x, train_y, epochs=140, batch_size=60, validation_data=(val_x, val_y))
    model.save_weights('first_try.h5')
    history.append(h)
    i += 1

# plot the accuracy and val_accuracy
plt.figure()
plt.plot(history[-1].history['accuracy'])
plt.plot(history[-1].history['val_accuracy'])
plt.legend()
plt.show()
