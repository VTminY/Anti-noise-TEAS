import os
import pandas as pd
import numpy as np
import random
import tensorflow
import glob
from collections import Counter
from keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.python.keras.layers import *
from tqdm import tqdm
import librosa
import librosa.display


def call_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tensorflow.compat.v1.GPUOptions(allow_growth=True)
    sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(
        gpu_options=gpu_options))


def display_gpu():
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos]
    for d in devices:
        print(d)


def set_seeds(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)


def extract_features(parent_dir, sub_dirs, max_file=10, feature_y=64, file_ext="*.wav"):
    n_fft = int((3.2 * 8000/feature_y)*4)
    hop_length = int(3.2 * 8000/feature_y)
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):
            label_name = fn.split('\\')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn, sr=8000)
            mels = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
            log_mel_spectrogram = librosa.power_to_db(mels)
            feature.extend([log_mel_spectrogram])
    return [feature, label]


def extract_features_test(test_dir, feature_y=64, file_ext="*.wav"):
    n_fft = int((3.2 * 8000/feature_y)*4)
    hop_length = int(3.2 * 8000/feature_y)
    feature = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]):
        # X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
        X, sample_rate = librosa.load(fn, sr=8000)
        mels = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        log_mel_spectrogram = librosa.power_to_db(mels)
        feature.extend([log_mel_spectrogram])
    return feature


def generate_train(parent_dir, feature_y=64):
    sub_dirs = np.array(['BePatrolling', 'ComeBack', 'FollowMe', 'HoldOn', 'MoveOn', 'TurnLeft',
                         'TurnRight', 'TheDrone', 'TheNurse', 'TheSpider'])
    temp = extract_features(parent_dir, sub_dirs, feature_y=feature_y, max_file=1000)
    temp = np.array(temp, dtype=object)
    data = temp.transpose()
    np.save('CAM_train_mfcc_128', data)


def generate_test(path, envi, feature_y=64):
    X_test = extract_features_test(path, feature_y=feature_y)
    temp = np.array(X_test)
    np.save('./CAM_test_' + envi + '_mfcc_128', temp)


def acc_loss_graph(acc, loss, name):
    Data_acc = pd.DataFrame(acc)
    Data_acc.to_csv('acc.csv')
    Data_loss = pd.DataFrame(loss)
    Data_loss.to_csv('loss.csv')
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy', linestyle='-', marker='o')
    # plt.plot(val_acc, label='Validation Accuracy', linestyle='-.', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss', linestyle='-', marker='o')
    # plt.plot(val_loss, label='Validation Loss', linestyle='-.', marker='s')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("CAM_Acc_Loss_{}.jpg".format(name))
    plt.close()
    # plt.show()



def model_train(epochs=50, batch_size=10, drop_out=0.2, name='Nobody'):
    print('\n', 'The Voice_Mel_model is training: ')
    train_data = np.load('./CAM_train_mfcc_128.npy', allow_pickle=True)
    X = train_data[:, 0]
    Y = np.array(train_data[:, 1])
    Y = to_categorical(Y)
    X_train = np.vstack(X[:])
    X_train = X_train.reshape(len(Y), X[0].shape[0], X[0].shape[1], 1)
    print('The shape of X_train(Train): ', X_train.shape)

    input_dim = (X[0].shape[0], X[0].shape[1], 1)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_dim, activation='relu', name='Conv1'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu', name='Conv2'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding="same", activation='relu', name='Conv3'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding="same", activation='relu', name='Conv4'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(drop_out))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name='FullConnect1'))
    model.add(Dropout(drop_out))
    # model.add(Dense(512, activation='relu', name='FullConnect2'))
    model.add(Dense(512, activation='relu', name='FullConnect2'))
    model.add(Dropout(drop_out))
    # model.add(Dense(256, activation='relu', name='FullConnect3'))
    # model.add(Dropout(drop_out))
    model.add(Dense(256, activation='relu', name='FullConnect3'))
    # model.add(Dense(1024, activation='relu', name='FullConnect1'))
    # model.add(Dense(100, activation='relu', name='FullConnect2'))
    # # model.add(Dense(256, activation='relu', name='FullConnect3'))
    model.add(Dense(10, activation='softmax', name='SoftmaxLayer'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    filepath = "CAM_weights_best_" + name + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, restore_best_weights=False,
                                 mode='auto')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, Y, epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks_list, verbose=1)
    acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    acc_loss_graph(acc, loss, name)


def model_test(all_test_path, envi, output_name):
    model_path = glob.glob('./*.hdf5')
    print(model_path)
    model = load_model(model_path[0])
    X_test = np.load('./CAM_test_' + envi + '_mfcc_128.npy', allow_pickle=True)
    X_test_new = np.vstack(X_test)
    X_test_new = X_test_new.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    print('\n', 'The shape of X_test(Train): ', X_test_new.shape, '\n')
    predictions = model.predict(X_test_new)

    preds = np.argmax(predictions, axis=1)
    preds = [label_dict_inv[x] for x in preds]

    path = glob.glob(all_test_path + '/*.wav')
    result = pd.DataFrame({'name': path, 'Predicted_label': preds})

    result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
    result.to_csv(output_name)


def result_vote(output_name):
    result_tenfold = pd.read_csv('0_submit.csv')
    for _ in range(1, 10):
        result_tenfold = pd.merge(result_tenfold, pd.read_csv(f'{_}_submit.csv'), on='name')

    label = result_tenfold.columns.drop('name')
    all_label = pd.DataFrame(result_tenfold, columns=label)

    label_merge = []
    for _ in all_label.values:
        vote = Counter(_).most_common(1)
        label_merge.append(vote[0][0])

    result_merge = pd.DataFrame({'name': result_tenfold['name'], 'label': label_merge})
    result_merge.to_csv(output_name)


if __name__ == '__main__':
    participant = 'Participant'
    envi1 = 'Quiet'
    envi2 = 'Rain'
    envi3 = 'Crowd'
    call_gpu()
    set_seeds(688)
    plt.rcParams['figure.figsize'] = (20, 15)

    label_dict = {'BePatrolling': 0, 'ComeBack': 1, 'FollowMe': 2, 'HoldOn': 3, 'MoveOn': 4,
                  'TurnLeft': 5, 'TurnRight': 6, 'TheDrone': 7, 'TheNurse': 8, 'TheSpider': 9}
    label_dict_inv = {v: k for k, v in label_dict.items()}

    train_path = './Train_' + participant + '_Data_10 Types'
    test_path1 = './Test_' + envi1 + '_P_Data_10 Types'
    result_name1 = 'New_CAM_' + envi1 + '_' + participant + '_submit_10 Types.csv'
    test_path2 = './Test_' + envi2 + '_P_Data_10 Types'
    result_name2 = 'New_CAM_' + envi2 + '_' + participant + '_submit_10 Types.csv'
    test_path3 = './Test_' + envi3 + '_P_Data_10 Types'
    result_name3 = 'New_CAM_' + envi3 + '_' + participant + '_submit_10 Types.csv'

    generate_train(train_path, feature_y=128)
    generate_test(test_path1, envi=envi1, feature_y=128)
    generate_test(test_path2, envi=envi2, feature_y=128)
    generate_test(test_path3, envi=envi3, feature_y=128)
    model_train(epochs=300, batch_size=20, drop_out=0.2, name=participant)
    model_test(test_path1, envi1, result_name1)
    model_test(test_path2, envi2, result_name2)
    model_test(test_path3, envi3, result_name3)


