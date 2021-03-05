# coding: utf-8
import os, sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from tensorflow.keras import utils
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import cifar100


from VGG.VGG16 import NetWork
from utils.input_image import InputImage
from utils.line_notify import LineNotify
from utils.save_model import save_model
from utils.save_pickle import save_pickle
from utils.set_seed import set_random_seed


# set magical number
NUM_CLASSES = 100
CHANNEL = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

SEED = 19990109


def main():
    """

    :return:
    """
    # 乱数シードの設定
    set_random_seed(SEED)
    # 画像データの取得
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    # 画像の正規化
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # one-hot ラベル化
    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)
    # model の定義
    model = NetWork(IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE).model()
    # 学習

    start_time = time.time()

    history = model.fit(
        X_train, y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, verbose=1,validation_data=(X_test, y_test)
    )
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    # データの保存
    save_model(model)
    save_pickle(file_name='history', object=history.history)

    predict = model.predict(X_test)
    save_pickle(file_name='predict', object=predict)
    save_pickle(file_name='y_test', object=y_test)

    notify = LineNotify()
    message = "{}分で終了しました。".format(execution_time)
    notify.post_linenotify(message)


if __name__ == '__main__':
    main()