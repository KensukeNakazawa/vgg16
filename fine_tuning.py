# coding: utf-8

import os, sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from tensorflow.keras import utils
# from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.vgg import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers


from VGG.VGG16 import NetWork
from utils.input_image2 import InputImageV2 as InputImage
from utils.data_augmentation import DataAugmentation
from utils.line_notify import LineNotify
from utils.save_model import save_model
from utils.save_pickle import save_pickle
from utils.set_seed import set_random_seed

# set magical number
NUM_CLASSES = 4
CHANNEL = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.005
IMAGE_SIZE = 224

SEED = 19990109


def main():
    """

    :return:
    """

    set_random_seed(SEED)

    image = InputImage(image_size=(IMAGE_SIZE, IMAGE_SIZE), train_size=0.75)
    X_train, X_test, y_train, y_test = image.get_test_train_data()

    da = DataAugmentation()
    X_train, y_train = da.rotation(X_train, y_train)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    base_vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL))
    model = models.Sequential()
    model.add(base_vgg16)
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, acrivation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    # VGG16の図の青色の部分は重みを固定（frozen）
    for layer in model.layers[:15]:
        layer.trainable = False

    # 多クラス分類を指定
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum=0.9),
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test),
    )

    save_model(model)

    predict = model.predict(X_test)

    save_pickle(file_name='predict', object=predict)
    save_pickle(file_name='history', object=history)
    save_pickle(file_name='y_test', object=y_test)

    notify = LineNotify()
    message = '終了しました。'
    notify.post_linenotify(message)


if __name__ == '__main__':
    main()
