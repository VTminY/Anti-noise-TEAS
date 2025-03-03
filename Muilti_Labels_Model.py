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
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.python.keras.layers import *
from tqdm import tqdm
import librosa
import librosa.display


# 调用GPU进行训练
def call_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    gpu_options = tensorflow.compat.v1.GPUOptions(allow_growth=True)
    sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(
        gpu_options=gpu_options))
    # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


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
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):  # 遍历数据集的所有文件
            label_all = fn.split('\\')[-2]
            label_signal = label_all.split('_')[0]
            label_gender = label_all.split('_')[1]
            label.extend([[label_dict[label_signal], label_dict[label_gender]]])
            # X, sample_rate = librosa.load(fn, res_type='kaiser_fast') # 采样率为22050
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
        X, sample_rate = librosa.load(fn, sr=8000)
        mels = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        log_mel_spectrogram = librosa.power_to_db(mels)
        feature.extend([log_mel_spectrogram])
    return feature


def generate_train(parent_dir, feature_y=64):
    sub_dirs = np.array(['DW_1', 'DW_2', 'DW_3', 'GB_1', 'GB_2', 'GB_3',
                         'HM_1', 'HM_2', 'HM_3', 'HW_1', 'HW_2', 'HW_3',
                         'ISOK_1', 'ISOK_2', 'ISOK_3', 'TY_1', 'TY_2', 'TY_3'])
    temp = extract_features(parent_dir, sub_dirs, feature_y=feature_y, max_file=1000)
    temp = np.array(temp, dtype=object)
    data = temp.transpose()
    np.save('CAM_train_mfcc_128', data)
    # print(data.shape)


def generate_test(path, feature_y=64):
    X_test = extract_features_test(path, feature_y=feature_y)
    temp = np.array(X_test)
    np.save('./CAM_test_mfcc_128', temp)


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
    # print(train_data)
    X = train_data[:, 0]
    Y = np.array(train_data[:, 1])
    Y = MultiLabelBinarizer().fit_transform(Y)
    print(Y.shape)
    print(Y)
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
    model.add(Dense(4096, activation='relu', name='FullConnect1'))
    model.add(Dropout(drop_out))
    # model.add(Dense(512, activation='relu', name='FullConnect2'))
    model.add(Dense(1024, activation='relu', name='FullConnect2'))
    model.add(Dropout(drop_out))
    # model.add(Dense(256, activation='relu', name='FullConnect3'))
    # model.add(Dropout(drop_out))
    model.add(Dense(128, activation='relu', name='FullConnect3'))
    # model.add(Dense(256, activation='relu', name='FullConnect3'))
    # model.add(Dense(6, activation='softmax', name='SoftmaxLayer'))
    model.add(Dense(9, activation='sigmoid', name='SigmoidLayer'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    # model.compile(optimizer='Adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    model.summary()

    filepath = "CAM_weights_best_" + name + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='binary_accuracy', verbose=0, restore_best_weights=False,
                                 mode='auto')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, Y, epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks_list, verbose=1)  # validation_split = 0.2

    acc = history.history['binary_accuracy']
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    acc_loss_graph(acc, loss, name)


def model_test(all_test_path, output_name):
    model_path = glob.glob('./*.hdf5')
    print(model_path)
    model = load_model(model_path[0])
    model_name = model_path[0].split('\\')[-1]
    model_img_name = 'Structure_' + model_name.split('.')[0] + '.png'
    keras.utils.plot_model(model, to_file=model_img_name, show_shapes=True)

    X_test = np.load('./CAM_test_mfcc_128.npy', allow_pickle=True)
    X_test_new = np.vstack(X_test)
    X_test_new = X_test_new.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    print('\n', 'The shape of X_test(Train): ', X_test_new.shape, '\n')
    predictions = model.predict(X_test_new)

    p = pd.DataFrame(predictions)
    p.to_csv('Predictions_Score.csv')

    pred_signal = predictions[:, :6]
    pred_gender = predictions[:, 6:9]
    preds_signal = np.argmax(pred_signal, axis=1)
    preds_gender = np.argmax(pred_gender, axis=1)
    preds_gender = [i+6 for i in preds_gender]
    result_signal = [label_dict_inv[x] for x in preds_signal]
    result_gender = [label_dict_inv[x] for x in preds_gender]
    path = glob.glob(all_test_path + '/*.wav')

    name = [0]*len(path)
    for index, path_str in enumerate(path):
        name[index] = path[index].split('\\')[-1].split('.')[0]
    result = pd.DataFrame({'name': name, 'Predicted_Signal': result_signal, 'Predicted_Gender': result_gender})
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
    participant = 'Participants'
    call_gpu()
    set_seeds(616)
    plt.rcParams['figure.figsize'] = (20, 15)
    label_dict = {'DW': 0, 'GB': 1, 'HM': 2, 'HW': 3, 'ISOK': 4, 'TY': 5, 'HMY': 6, 'STC': 7, 'YCJ': 8}
    label_dict_inv = {v: k for k, v in label_dict.items()}
    train_path = './Train_CAM_' + participant + '_Data'
    test_path = './Test_CAM_' + participant + '_Data'
    result_name = 'CAM_' + participant + '_submit.csv'
    model_train(epochs=200, batch_size=10, drop_out=0.25, name=participant)
    model_test(test_path, result_name)


