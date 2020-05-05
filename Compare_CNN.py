import CNN_compare_violin1
import numpy as np
import matplotlib.pyplot as plt
import datetime

curr_time = datetime.datetime.now()
curr_time = curr_time.strftime("%Y-%m-%d %H:%M:%S")

"""
compare framesize
"""
def compare_frame_size():
    input_length = 13
    mfcc_length = 13
    fuzzy = 0
    framesize = 1024
    overlapSize = 512
    FFT_size = 1024
    batch = 64
    dropout = 0.1
    kernel = 8

    # frame size = 1024
    print("frame size = ", framesize)
    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # frame size = 512
    framesize = 512
    overlapSize = 512
    FFT_size = 1024

    print("frame size = ", framesize)
    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)

    F_result_2 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nframe size = ', '1024',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n',
               '\nframe size = ', '512',
               '\ntest F-measure = ', str(F_result_2),
               '\n', '\n', '\n']
    f = open('frame size result.txt', 'a')
    f.writelines(content)
    f.close()

"""
compare initial model parameter
"""
def compare_initial():
    input_length = 13
    mfcc_length = 13
    fuzzy = 0
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 64
    dropout = 0.1
    kernel = 8

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    # round 1
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # round 2
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_2 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # round 3
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_3 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nround 1',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n',
               '\nround 2',
               '\ntest F-measure = ', str(F_result_2),
               '\n', '\n',
               '\nround 3',
               '\ntest F-measure = ', str(F_result_3),
               '\n', '\n', '\n']
    f = open('initial compare.txt', 'a')
    f.writelines(content)
    f.close()



def compare_FB():
    input_length = 13
    mfcc_length = 13
    fuzzy = 0
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 64
    dropout = 0.1
    kernel = 8

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    # filterbank number = 13
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # filterbank number = 25
    mfcc_length = 26

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_2 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # filterbank number = 32
    mfcc_length = 32

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_3 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # filterbank number = 38
    mfcc_length = 38

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_4 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # filterbank number = 44
    mfcc_length = 44

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_5 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # filterbank number = 50
    mfcc_length = 50

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_6 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # filterbank number = 60
    mfcc_length = 60

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_7 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # filterbank number = 70
    mfcc_length = 70

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_8 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nfilter bank = 13',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n'
               '\nfilter bank = 26',
               '\ntest F-measure = ', str(F_result_2),
               '\n', '\n',
               '\nfilter bank = 32',
               '\ntest F-measure = ', str(F_result_3),
               '\n', '\n',
               '\nfilter bank = 38',
               '\ntest F-measure = ', str(F_result_4),
               '\n', '\n',
               '\nfilter bank = 44',
               '\ntest F-measure = ', str(F_result_5),
               '\n', '\n',
               '\nfilter bank = 50',
               '\ntest F-measure = ', str(F_result_6),
               '\n', '\n',
               '\nfilter bank = 60',
               '\ntest F-measure = ', str(F_result_7),
               '\n', '\n',
               '\nfilter bank = 70',
               '\ntest F-measure = ', str(F_result_8),
               '\n', '\n', '\n']
    f = open('filter bank.txt', 'a')
    f.writelines(content)
    f.close()

def compare_batch():
    input_length = 13
    mfcc_length = 13
    fuzzy = 0
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 64
    dropout = 0.1

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    # batch size = 64
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # batch size = 128
    batch = 128
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_2 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # batch size = 256
    batch = 256
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_3 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nbatch size = 64',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n'
               '\nbatch size = 128',
               '\ntest F-measure = ', str(F_result_2),
               '\n', '\n',
               '\nbatch size = 256',
               '\ntest F-measure = ', str(F_result_3),
               '\n', '\n', '\n']
    f = open('batch size compare', 'a')
    f.writelines(content)
    f.close()


def compare_dropout():
    input_length = 13
    mfcc_length = 13
    fuzzy = 0
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 128

    dropout = 0.1

    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    dropout = 0.2
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_2 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    dropout = 0.3
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_3 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    dropout = 0.4
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_4 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    dropout = 0.5
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout)
    F_result_5 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\ndropout = 0.1',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n'
               '\ndropout = 0.2',
               '\ntest F-measure = ', str(F_result_2),
               '\n', '\n',
               '\ndropout = 0.3',
               '\ntest F-measure = ', str(F_result_3),
               '\n', '\n',
               '\ndropout = 0.4',
               '\ntest F-measure = ', str(F_result_4),
               '\n', '\n',
               '\ndropout = 0.5',
               '\ntest F-measure = ', str(F_result_5),
               '\n', '\n', '\n']
    f = open('dropout compare.txt', 'a')
    f.writelines(content)
    f.close()

def kernel_number():
    input_length = 13
    mfcc_length = 13
    fuzzy = 0
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 128
    dropout = 0.5


    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    kernel = 8
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    kernel = 16
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_2 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    kernel = 32
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_3 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    kernel = 64
    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_4 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nkernel = 8, 16',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n'
               '\nkernel = 16, 16',
               '\ntest F-measure = ', str(F_result_2),
               '\n', '\n',
               '\nkernel = 32, 16',
               '\ntest F-measure = ', str(F_result_3),
               '\n', '\n',
               '\nkernel = 64, 16',
               '\ntest F-measure = ', str(F_result_4),
               '\n', '\n', '\n']
    f = open('kernel compare.txt', 'a')
    f.writelines(content)
    f.close()

def fuzzy_compare():
    input_length = 13
    mfcc_length = 13
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 128
    dropout = 0.5
    kernel = 8

    fuzzy = 0
    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    fuzzy = 10
    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_2 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    fuzzy = 20
    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_3 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nfuzzy = 0',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n'
               '\nfuzzy = 10',
               '\ntest F-measure = ', str(F_result_2),
               '\n', '\n',
               '\nfuzzy = 20',
               '\ntest F-measure = ', str(F_result_3),
               '\n', '\n', '\n']
    f = open('fuzzy compare.txt', 'a')
    f.writelines(content)
    f.close()


def delta_MFCC():
    input_length = 13
    mfcc_length = 13
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 128
    dropout = 0.5
    kernel = 8
    fuzzy = 0
    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nwith delta MFCC',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n', '\n']
    f = open('delta MFCC.txt', 'a')
    f.writelines(content)
    f.close()

def voice_type():
    input_length = 13
    mfcc_length = 13
    framesize = 512
    overlapSize = 512
    FFT_size = 1024
    batch = 128
    dropout = 0.5
    kernel = 8
    fuzzy = 0
    train_input, train_output, test_input, test_output, valid_input, valid_output = CNN_compare_violin1.data_prepare(input_length, mfcc_length, fuzzy, framesize, overlapSize, FFT_size)

    CNN_compare_violin1.initial_CNN_process(input_length, mfcc_length, train_input, train_output, test_input, test_output,  valid_input, valid_output, 0, batch, dropout, kernel)
    F_result_1 = CNN_compare_violin1.CNN_process(train_input, train_output, test_input, test_output, valid_input, valid_output, batch)

    # store result
    content = [curr_time,
               '\nmulti voice type',
               '\ntest F-measure = ', str(F_result_1),
               '\n', '\n', '\n']
    f = open('voice type.txt', 'a')
    f.writelines(content)
    f.close()


compare_frame_size()
compare_initial()
compare_FB()
compare_batch()
compare_dropout()
kernel_number()
fuzzy_compare()
delta_MFCC()
voice_type()