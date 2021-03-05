# coding: utf-8
import os, sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from tensorflow.keras import utils
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold

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


def cross_validation(X, y):
    """交差検証を行う
    引数で与えられた画像データとラベルデータを訓練データと検証データに分け
    k-分割交差検証を行う

    Args:
        X(ndarray): 画像データ
        y(ndarray): ラベルデータ

    Returns:
        history_list(list): 学習のhistoryの配列(辞書)
        cross_valid_scores(list): 交差検証を行った時の各学習結果の評価値
        model(sequential): 最後に学習を行ったモデル

    References:
        url: https://qiita.com/agumon/items/0df9f008a255796b5a94
    """

    fold_num = 5
    kfold = KFold(n_splits=fold_num, random_state=SEED)
    cross_valid_scores = []

    i = 1
    for train_index, test_index in kfold.split(X, y):
        model = NetWork(IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE).model()
        history = model.fit(X[train_index], y[train_index],
                            batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(X[test_index], y[test_index]), verbose=0)
        scores = model.evaluate(X[test_index], y[test_index], verbose=0)
        cross_valid_scores.append(scores)
        save_name = "{}_history".format(i)
        save_pickle(file_name=save_name, object=history.history)
        i += 1
    return cross_valid_scores, model


def step_decay(epoch: int):
    """

    Args:
        epoch: 現在のエポック

    Returns:
        learning_rate: 更新した学習率
    """
    learning_rate = LEARNING_RATE
    if epoch >= 10:
        learning_rate = learning_rate / 10
    if epoch >= 30:
        learning_rate = learning_rate / 100

    return learning_rate


def main():
    """

    :return:
    """

    set_random_seed(SEED)

    # 画像データとラベルデータの取得
    image = InputImage(image_size=(IMAGE_SIZE, IMAGE_SIZE), train_size=0.75)
    X_train, X_test, y_train, y_test = image.get_test_train_data()

    # データ拡張
    da = DataAugmentation()
    X_train, y_train = da.rotation(X_train, y_train)

    # 正規化
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # one hot label
    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    model = NetWork(IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE).model()

    # lr_decay = LearningRateScheduler(step_decay, verbose=1)

    history = model.fit(
        X_train, y_train, batch_size=BATCH_SIZE,
        epochs=EPOCHS, verbose=1,validation_data=(X_test, y_test),
        # callbacks=[lr_decay]
    )
    # cross_valid_scores, model = cross_validation(X_train, y_train)
    predict = model.predict(X_test)

    # データの保存
    save_model(model)
    save_pickle(file_name='predict', object=predict)
    save_pickle(file_name='y_test', object=y_test)

    notify = LineNotify()
    message = '終了しました。'
    notify.post_linenotify(message)


if __name__ == '__main__':
    main()
