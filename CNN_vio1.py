from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import wave
import math
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
import keras.optimizers

from python_speech_features import mfcc
import scipy.io.wavfile as wav
from pydub import AudioSegment
from scipy.signal import find_peaks
import os
import datetime

"""
load training data
"""

train_filenameList = []
train_filenumber = 0
train_lable_name_list = []
train_seg_path = []

f = open("whole train.txt")
for line in f:
    line = line.split("   ")
    name = line[0]
    annotation_name = line[1][:-1]

    train_filenameList.append(name)
    train_lable_name_list.append(annotation_name)

    train_filenumber += 1

print("training files:",train_filenameList)
print("training file number:",train_filenumber)

"""
load testing data
"""

test_filenameList = []
test_filenumber = 0
test_lable_name_list = []
test_seg_path = []


f = open("test.txt")
for line in f:
    line = line.split("   ")
    name = line[0]
    annotation_name = line[1][:-1]

    test_filenameList.append(name)
    test_lable_name_list.append(annotation_name)

    test_filenumber += 1

print("test files:",test_filenameList)
print("test file number:",test_filenumber)

"""
load validation data
"""
valid_filenameList = []
valid_filenumber = 0
valid_lable_name_list = []
valid_seg_path = []

f = open("validation.txt")
for line in f:
    line = line.split("   ")
    name = line[0]
    annotation_name = line[1][:-1]

    valid_filenameList.append(name)
    valid_lable_name_list.append(annotation_name)

    valid_filenumber += 1

print("validation files:",valid_filenameList)
print("validation file number:",valid_filenumber)


def label_process(path):
    onset_single_1 = []
    onset_single = []

    f = open(path)
    line = f.readline()
    while line:
        onset_single_1.append(line.split(",")[0])
        line = f.readline()
    # print(whole_train_onset)
    f.close()

    onset_single_1 = onset_single_1[1:-1]     # extract the time column in the annotation file.
    onset_start_time = onset_single_1[0]      # the first and last the first onset time are used for wave segment
    onset_finish_time = onset_single_1[-1]

    for item in onset_single_1:
        item_inter = float(item) - float(onset_start_time) + 0.1
        onset_single.append(item_inter)

    return onset_single, onset_start_time, onset_finish_time


'''
print signal sample rate
'''
def SignalInfo(path):
    global nframes, waveData, framerate

    f = wave.open(path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print("frame rate =", framerate)


'''
draw specgram
'''
def specgram():
    NFFT = framesize
    spectrum, freqs, ts, fig = plt.specgram(waveData[0], NFFT=NFFT, Fs=framerate, window=np.hanning(M=framesize),
                                            noverlap=overlapSize, mode='default', scale_by_freq=True, sides='default',
                                            scale='dB', xextent=None)
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.title('Spectrogram')
    plt.show()

def train_wav_seg(path, onset_start_time, onset_finish_time):
    seg_path = "audio_seg/" + path[-29:]
    sound = AudioSegment.from_wav(path)
    sound_seg = sound[(float(onset_start_time) * 1000)-100:(float(onset_finish_time) * 1000)+100]
    sound_seg.export(seg_path, format="wav")
    train_seg_path.append(seg_path)


def test_wav_seg(path, onset_start_time, onset_finish_time):
    seg_path = "audio_seg/" + path[-29:]
    sound = AudioSegment.from_wav(path)
    sound_seg = sound[(float(onset_start_time) * 1000)-100:(float(onset_finish_time) * 1000)+100]
    sound_seg.export(seg_path, format="wav")
    test_seg_path.append(seg_path)


def valid_wav_seg(path, onset_start_time, onset_finish_time):
    seg_path = "audio_seg" + path[-29:]
    sound = AudioSegment.from_wav(path)
    sound_seg = sound[(float(onset_start_time) * 1000)-100:(float(onset_finish_time) * 1000)+100]
    sound_seg.export(seg_path, format="wav")
    valid_seg_path.append(seg_path)


def MFCC_gen_tradition(framesize, overlapSize, audio_path, mfcc_length):
    (rate, sig) = wav.read(audio_path)
    move_time = math.ceil(nframes/overlapSize)

    window = np.hamming(framesize)

    start = 0
    frame = np.zeros((move_time,framesize))
    for i in range(0,move_time):
        inter_frame = sig[start:start + framesize]
        if len(inter_frame) < framesize:
            inter_frame = list(inter_frame)+[0]*(framesize-len(inter_frame))
        frame[i] = inter_frame
        frame[i] = frame[i] * window

        start = start + overlapSize

    mag_frames = np.absolute(np.fft.rfft(frame,framesize))
    pow_frames = ((1.0 / framesize) * (mag_frames ** 2))

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (framerate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, mfcc_length + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = (hz_points / (framerate / 2)) * (framesize / 2)

    fbank = np.zeros((mfcc_length, int(np.floor(framesize / 2 + 1))))
    for m in range(1, mfcc_length + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    mfcc_feat = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(mfcc_length + 1)]

    data_single = []

    for item in mfcc_feat:
        item_new = []
        for element in item:
            item_new.append((element - np.mean(mfcc_feat))/(np.var(mfcc_feat)))
        data_single.append(item_new)

    return data_single


'''
calculate MFCC with library
'''
def MFCC_gen(framesize, overlapSize, audio_path, mfcc_length):
    data_single = []
    (rate, sig) = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate, winlen=framesize/framerate, winstep=overlapSize/framerate, nfft=1024, nfilt=mfcc_length, numcep=mfcc_length)

    for item in mfcc_feat:
        item_new = []
        for element in item:
            item_new.append((element - np.mean(mfcc_feat))/(np.var(mfcc_feat)))
        data_single.append(item_new)

    return data_single


'''
calculate Delta MFCC
'''
def delta_mfcc(data_single, cita, mfcc_length):
    del_mfcc = []

    for i in range(0, cita):
        data_single = [[0]*mfcc_length] + data_single
        data_single.append([0]*mfcc_length)

    for i in range(cita, len(data_single)-cita):
        numerator = [0] * mfcc_length
        denominator = 0
        new = []
        for j in range(1, cita+1):
            new_list = []
            for item in range(0, mfcc_length):
                new_list.append(j * (data_single[i+j][item] - data_single[i-j][item]))
            for a in range(0, len(new_list)):
                numerator[a] = numerator[a] + new_list[a]
        for k in range(1, cita + 1):
            denominator = denominator + k**2
        for item in numerator:
            new.append(item/(2*denominator))
        del_mfcc.append(new)

    return del_mfcc

'''
match MFCCs with lables
'''
def load_dataset(framesize, overlapSize, data_set, onset_set, CNNinput_length, mfcc_length):
    input_data_single = []

    for i in range(0, int(CNNinput_length/2)):
        data_set = [[0]*mfcc_length] + data_set
        data_set.append([0]*mfcc_length)

    frame_onset_time = []
    onset_index = []

    for i in range(int(CNNinput_length/2), len(data_set)-int(CNNinput_length/2)):
        input_data_single.extend(data_set[i-int(CNNinput_length/2):
                                          i+(int(CNNinput_length/2)+1)])
        onset_time = (((i - int(CNNinput_length/2)) * overlapSize) + 0.5 * framesize) / framerate
        frame_onset_time.append(onset_time)

    for item in onset_set:
        difference = []
        for i in range(0, len(frame_onset_time)):
            difference.append(abs(item - frame_onset_time[i]))
        onset_index.append(difference.index(min(difference)))

    output_data_single = [[1,0]] * (int(len(input_data_single)/CNNinput_length))
    for item in onset_index:
        output_data_single[item] = [0,1]

    return input_data_single, output_data_single

'''
combine all the training files' data together
'''
def train_combine(train_input_single, train_output_single, fuzzy):
    train_data_set_list.extend(train_input_single)
    train_onset_set_list.extend(train_output_single)

    for i in range(0, len(train_onset_set_list)):
        if train_onset_set_list == [0,1]:
            train_onset_set_list[i-fuzzy:i+fuzzy+1] = [0,1]

    return train_data_set_list, train_onset_set_list


'''
combine all the testing files' data together
'''
def test_combine(test_input_single, test_output_single, fuzzy):
    test_data_set_list.extend(test_input_single)
    test_onset_set_list.extend(test_output_single)

    for i in range(0, len(test_onset_set_list)):
        if test_onset_set_list == [0,1]:
            test_onset_set_list[i-fuzzy:i+fuzzy+1] = [0,1]

    return test_data_set_list, test_onset_set_list


'''
combine all the validation files' data together
'''
def valid_combine(valid_input_single, valid_output_single, fuzzy):
    valid_data_set_list.extend(valid_input_single)
    valid_onset_set_list.extend(valid_output_single)

    for i in range(0, len(valid_onset_set_list)):
        if valid_onset_set_list == [0,1]:
            valid_onset_set_list[i-fuzzy:i+fuzzy+1] = [0,1]

    return valid_data_set_list, valid_onset_set_list


def F_measure(predict_onset, test_answer):
    predict_onset = np.array(predict_onset)
    tolerance = 4
    frame_onset_number = 0

    peaks, properties = find_peaks(predict_onset, height=0.05, distance=1)
    peaks = list(peaks)
    N_tp_test = len(peaks)
    delete_item = [0]*N_tp_test
    # peak_remove = peaks.copy()

    N_tp = 0
    for i in range(0, len(test_answer)):
        if list(test_answer[i]) == [0, 1]:
            frame_onset_number += 1
            a = 0
            for j in range(0, len(peaks)):
                # print(i)
                if peaks[j] >= (i - tolerance) and peaks[j] <= (i + tolerance):
                    N_tp += 1
                    a += 1
                    delete_item[j] = 1
            if a > 1:
                N_tp = N_tp - (a - 1)
            for k in range(1, len(delete_item)):
                if delete_item[k-1] == 0 and delete_item[k] == 1:
                    # print(k)
                    delete_item[k] = 2
                    peaks.remove(peaks[k])

    if N_tp != 0:
        precision = N_tp / N_tp_test
        recall = N_tp / frame_onset_number
        F = 2 * precision * recall / (precision + recall)
    else:
        F = 0

    return N_tp_test, frame_onset_number, N_tp, F


train_data_set_list = []
test_data_set_list = []
valid_data_set_list = []

train_onset_set_list = []
test_onset_set_list = []
valid_onset_set_list = []

train_channel2 = []
test_channel2 = []
valid_channel2 = []

'''
set hyperparameters
'''
input_length = 13
mfcc_length = 70
fuzzy = 0
framesize = 512
overlapSize = 512

"""
load training data
"""
for number in range(0, train_filenumber):
    train_onset_single, onset_start_time, onset_finish_time = label_process(train_lable_name_list[number])
    train_wav_seg(train_filenameList[number], onset_start_time, onset_finish_time)
    SignalInfo(train_filenameList[number])
    trainMFCC_single = MFCC_gen(framesize, overlapSize, train_seg_path[number], mfcc_length)
    train_input_single, train_output_single = load_dataset(framesize, overlapSize, trainMFCC_single, train_onset_single, input_length, mfcc_length)
    train_input_list, train_output_list = train_combine(train_input_single, train_output_single, fuzzy)

    del_mfcc = delta_mfcc(trainMFCC_single, 2, mfcc_length)
    del_train_input_single, del_train_output_single = load_dataset(framesize, overlapSize, del_mfcc, train_onset_single,
                                                               input_length, mfcc_length)
    train_channel2.extend(del_train_input_single)

"""
reshape into 4 dimension data
"""
train_input = np.array(train_input_list)
train_input = np.reshape(train_input, (int(len(train_input_list) / input_length), input_length, mfcc_length))
train_input = train_input.T

del_train_input = np.array(train_channel2)
del_train_input = np.reshape(del_train_input, (int(len(train_channel2) / input_length), input_length, mfcc_length))
del_train_input = del_train_input.T

train_channel_2 = np.stack((train_input, del_train_input), axis=0)
train_channel_2 = train_channel_2.T

train_output = np.array(train_output_list)

print("train input shape:", train_channel_2.shape)
print("train output shape:", train_output.shape)

"""
load testing data
"""
for number in range(0, test_filenumber):
    test_onset_single, onset_start_time, onset_finish_time = label_process(test_lable_name_list[number])
    test_wav_seg(test_filenameList[number], onset_start_time, onset_finish_time)
    SignalInfo(test_filenameList[number])
    testMFCC_single = MFCC_gen(framesize, overlapSize, test_seg_path[number], mfcc_length)
    test_input_single, test_output_single = load_dataset(framesize, overlapSize, testMFCC_single, test_onset_single, input_length, mfcc_length)
    test_input_list, test_output_list = test_combine(test_input_single, test_output_single, fuzzy)

    del_mfcc = delta_mfcc(testMFCC_single, 2, mfcc_length)
    del_test_input_single, del_test_output_single = load_dataset(framesize, overlapSize, del_mfcc, test_onset_single,
                                                               input_length, mfcc_length)
    test_channel2.extend(del_test_input_single)

"""
reshape into 4 dimension data
"""
test_input = np.array(test_input_list)
test_input = np.reshape(test_input, (int(len(test_input_list) / input_length), input_length, mfcc_length))
test_input = test_input.T

del_test_input = np.array(test_channel2)
del_test_input = np.reshape(del_test_input, (int(len(test_channel2) / input_length), input_length, mfcc_length))
del_test_input = del_test_input.T

test_channel_2 = np.stack((test_input, del_test_input), axis=0)
test_channel_2 = test_channel_2.T

test_output = np.array(test_output_list)

print("test input shape:", test_channel_2.shape)
print("test output shape:", test_output.shape)

"""
load validation data
"""
for number in range(0, valid_filenumber):
    valid_onset_single, onset_start_time, onset_finish_time = label_process(valid_lable_name_list[number])
    valid_wav_seg(valid_filenameList[number], onset_start_time, onset_finish_time)
    SignalInfo(valid_filenameList[number])
    validMFCC_single = MFCC_gen(framesize, overlapSize, valid_seg_path[number], mfcc_length)
    valid_input_single, valid_output_single = load_dataset(framesize, overlapSize, validMFCC_single, valid_onset_single, input_length, mfcc_length)
    valid_input_list, valid_output_list = valid_combine(valid_input_single, valid_output_single, fuzzy)

    del_mfcc = delta_mfcc(validMFCC_single, 2, mfcc_length)
    del_valid_input_single, del_valid_output_single = load_dataset(framesize, overlapSize, del_mfcc, valid_onset_single,
                                                               input_length, mfcc_length)
    valid_channel2.extend(del_valid_input_single)

"""
reshape into 4 dimension data
"""

valid_input = np.array(valid_input_list)
valid_input = np.reshape(valid_input, (int(len(valid_input_list) / input_length), input_length, mfcc_length))
valid_input = valid_input.T

del_valid_input = np.array(valid_channel2)
del_valid_input = np.reshape(del_valid_input, (int(len(valid_channel2) / input_length), input_length, mfcc_length))
del_valid_input = del_valid_input.T

valid_channel_2 = np.stack((valid_input, del_valid_input), axis=0)
valid_channel_2 = valid_channel_2.T

valid_output = np.array(valid_output_list)

print("valid input shape:", valid_channel_2.shape)
print("valid output shape:", valid_output.shape)

"""
CNN process
"""
batch_size = 128
num_classes = 2
epochs = 150

img_rows, img_cols = input_length, mfcc_length
input_shape = (input_length, mfcc_length, 2)

x_train = train_channel_2
y_train = train_output

x_test = test_channel_2
y_test = test_output

x_valid = valid_channel_2
y_valid = valid_output

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 data_format='channels_last'))
model.add(Conv2D(64, (3, 3), activation='relu',
                 input_shape=input_shape,
                 data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.save('my_model.h5')


"""
start validation
"""
validation_F = [0]

for step in range(5, epochs, 5):
    print("current validation epoch:", step)

    model = load_model('my_model.h5')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              shuffle=True,
              verbose=2)
    # del model

    y_valid_F = model.predict(x_valid)
    y_valid_F_1 = [i[1] for i in y_valid_F]
    detect_onset, reference_onset, detect_correct, F_measure_result = F_measure(y_valid_F_1, valid_output)
    print("current F-measure: ", F_measure_result)
    validation_F.append(F_measure_result)

    validation_stop = step - 5
    if validation_F[len(validation_F) - 1] < validation_F[len(validation_F) - 2]:  # F-score starts decreasing
        model.save('validation_check.h5')
        print("##### start checking validation #####")
        check_F = []
        model_check = load_model('validation_check.h5')
        model_check.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])

        model_check.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=5,
                        shuffle=True,
                        verbose=2)

        y_valid_F = model_check.predict(x_valid)
        y_valid_F_1 = [i[1] for i in y_valid_F]
        detect_onset, reference_onset, detect_correct, F_measure_result = F_measure(y_valid_F_1, valid_output)
        print("current F-measure: ", F_measure_result)
        check_F.append(F_measure_result)
        if check_F[0] > validation_F[len(validation_F) - 1]:  # F-score re-increase
            print("##### validation failed #####")
            model.save('my_model.h5')
            continue
        else:
            model_check = load_model('validation_check.h5')
            model_check.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=keras.optimizers.Adadelta(),
                                metrics=['accuracy'])

            model_check.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=10,
                            shuffle=True,
                            verbose=2)
            score = model_check.evaluate(x_valid, y_valid, verbose=0)

            y_valid_F = model_check.predict(x_valid)
            y_valid_F_1 = [i[1] for i in y_valid_F]
            detect_onset, reference_onset, detect_correct, F_measure_result = F_measure(y_valid_F_1, valid_output)
            print("current F-measure: ", F_measure_result)
            check_F.append(F_measure_result)
            if check_F[1] < validation_F[len(validation_F) - 1]:  # F-score continually decrease
                print("##### validation true #####")
                model.save('my_model.h5')
                break
            else:
                print("##### validation failed #####")
                continue
    else:
        model.save('my_model.h5')

"""
result review
"""
model = load_model('my_model.h5')

y_test_F = model.predict(x_test)
y_test_F_0 = [i[0] for i in y_test_F]
y_test_F_1 = [i[1] for i in y_test_F]

test_output_0 = [i[0] for i in test_output]
test_output_1 = [i[1] for i in test_output]

# x = np.arange(0, len(y_test_F_1), 1)
# plt.plot(x, y_test_F_1, color="r", linewidth=1, label="detect onset")
# plt.plot(x, test_output_1, color="b", linewidth=1, label="reference onset")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Violin 1")
# plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.3))
# plt.show()

# x = np.arange(0, len(validation_F)*5, 5)
# plt.plot(x, validation_F, color="r", linewidth=1, label="detect onset")
# plt.xlabel("epoch")
# plt.ylabel("F-measure")
# plt.title("violin1 step=10")
# plt.show()

"""
F measure
"""
detect_onset, reference_onset, detect_correct, F_measure_result = F_measure(y_test_F_1, test_output)
print("F-measure = ", F_measure_result)


"""
store data
"""
curr_time = datetime.datetime.now()
curr_time = curr_time.strftime("%Y-%m-%d %H:%M:%S")

content = [curr_time, '\nF-measure = ', str(F_measure_result), '\nvalidation stop epoch = ', str(validation_stop), '\n', '\n']
f = open('result.txt', 'a')
f.writelines(content)
f.close()


