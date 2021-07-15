from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers, Model, optimizers
from tensorflow.python.keras.optimizer_v2.adam import Adam


def conv2D(input_dim, cat):
    model = Sequential()
    model.add(Convolution2D(128, (5, 5), padding="same", input_shape=input_dim, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(cat, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def conv1D(input_dim):
    model = Sequential()
    model.add(Conv1D(16, 3, padding="same", activation="relu", input_shape=input_dim))  # 卷积层
    model.add(Conv1D(16, 3, padding="same", activation="relu"))  # 卷积层
    model.add(Conv1D(16, 3, padding="same", activation="relu"))  # 卷积层
    model.add(BatchNormalization())  # BN层
    model.add(Dropout(0.52, seed=66))
    model.add(Flatten())  # 展开
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.52, seed=66))
    model.add(Dense(20, activation="softmax"))  # 输出层：20个units输出20个类的概率
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def model11(input_dim):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding="same", activation="tanh", input_shape=input_dim))  # 卷积层
    model.add(MaxPool2D(pool_size=(3, 5)))  # 最大池化
    model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))  # 卷积层
    model.add(MaxPool2D(pool_size=(3, 5)))  # 最大池化层

    model.add(Dropout(0.6))
    model.add(Flatten())  # 展开
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(20, activation="softmax"))  # 输出层：20个units输出20个类的概率

    # 编译模型，设置损失函数，优化方法以及评价标准
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

