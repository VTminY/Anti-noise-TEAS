import glob
from tqdm import tqdm
import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
# Display
import matplotlib
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import librosa.display

plt.figure(dpi=600)
matplotlib.rc("font", family='SimHei')
matplotlib.rcParams['axes.unicode_minus'] = False


def get_test_wav_features(wav_path, feature_y=128, file_format='wav'):
    if file_format == 'wav':
        n_fft = int((3.2 * 8000 / feature_y) * 4)
        hop_length = int(3.2 * 8000 / feature_y)
        X, sample_rate = librosa.load(wav_path, sr=8000)
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        log_mel = librosa.power_to_db(mel)
    if file_format == 'npy':
        X_test = np.load('./CAM_test_mfcc_128.npy', allow_pickle=True)
        X_test_new = np.vstack(X_test)
        feature = X_test_new.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    return log_mel


def make_gradcam_heatmap(wav_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(wav_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    # grads.shape(1, 10, 10, 2048)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    # pooled_grads 是一个一维向量,shape=(2048,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# resize Grad-CAM
def display_grad_cam(heat_map, size1, size2):
    # Rescale heatmap to a range 0-255
    heat_map = np.uint8(255 * heat_map)

    # Use jet colormap to colorize heatmap

    jet = matplotlib.colormaps.get_cmap("hot")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heat_map]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((size1, size2))
    # jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    return jet_heatmap


label_dict = {'DW': 0, 'GB': 1, 'HM': 2, 'HW': 3, 'ISOK': 4, 'TY': 5}
label_dict_inv = {v: k for k, v in label_dict.items()}

# Make model
model = load_model('CAM_weights_best_YCJ.hdf5')
model.summary()

# Remove last layer's softmax
model.layers[-1].activation = None

last_conv_layer_name = 'Conv2'

# The local path to our target image
folder_path = './Test_CAM_YCJ_Data'

os.mkdir('./Grad-CAM-Result')
output_path = './Grad-CAM-Result'
Data_mean_heatmap = pd.DataFrame()
# print(folder_path)


df1 = pd.DataFrame({'One': [folder_path.split('/')[-1]]})
df1.to_excel(output_path + '\\Data_heatmap.xlsx', sheet_name=folder_path.split('/')[-1], index=False)
df1.to_excel(output_path + '\\Data_mel.xlsx', sheet_name=folder_path.split('/')[-1], index=False)


for fn in tqdm(glob.glob(os.path.join(folder_path, "*.wav"))[:]):
    label_name = fn.split('\\')[-1]
    label_name = label_name.split('.')[0]
    # Prepare wav_array
    log_mel = get_test_wav_features(fn)
    dim1 = log_mel.shape[0]
    dim2 = log_mel.shape[1]
    wav_array = log_mel.reshape(1, dim1, dim2, 1)
    # Print what the top predicted class is
    predictions = model.predict(wav_array)
    preds = np.argmax(predictions, axis=1)
    preds = [label_dict_inv[x] for x in preds]
    print("The signal predicted: ", preds)
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(wav_array, model, last_conv_layer_name)
    # save_and_display_gradcam(img_path, heatmap)
    # Save Grad-CAM
    Data_heatmap = pd.DataFrame(heatmap)
    Data_mel = pd.DataFrame(log_mel)
    with pd.ExcelWriter(output_path + '/Data_heatmap.xlsx', mode='a') as writer:
        Data_heatmap.to_excel(writer, sheet_name=label_name, index=False)
    # Data_heatmap.to_csv('Data_heatmap.csv')
    with pd.ExcelWriter(output_path + '/Data_mel.xlsx', mode='a') as writer:
        Data_mel.to_excel(writer, sheet_name=label_name, index=False)
    # Data_mel.to_csv('Data_mels.csv')

    Data_mean_heatmap_new = pd.DataFrame(np.mean(heatmap, axis=1), columns=[label_name])
    Data_mean_heatmap_all = pd.concat([Data_mean_heatmap, Data_mean_heatmap_new], axis=1)
    Data_mean_heatmap = Data_mean_heatmap_all
    # print(Data_mean_heatmap_new)
    # Data_mean_heatmap.to_csv('Data_mean_heatmap.csv')

    # grad_cam = display_grad_cam(heatmap, dim1, dim2)
    # Display Grad-CAM
    figure2, (axs1, axs2) = plt.subplots(1, 2)
    fig1 = axs1.matshow(heatmap, cmap='hot')
    figure2.colorbar(fig1)
    fig2 = axs2.matshow(log_mel)
    figure2.colorbar(fig2)
    plt.savefig(output_path + "/" + label_name + '.jpg', dpi=600)
    matplotlib.pyplot.close()
    # figure1 = plt.figure(2)
    # librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    # plt.show()

Data_mean_heatmap_all_mean = pd.DataFrame(np.mean(Data_mean_heatmap_all, axis=1), columns=['ALL_mean'])
with pd.ExcelWriter(output_path + '/Data_heatmap_mean.xlsx') as writer:
    Data_mean_heatmap_all.to_excel(writer, sheet_name='Data_mean_heatmap_all', index=False)
    Data_mean_heatmap_all_mean.to_excel(writer, sheet_name='Data_mean_heatmap_all_mean', index=False)


