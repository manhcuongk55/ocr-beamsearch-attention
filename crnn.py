import os
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Bidirectional, Dropout, MaxPooling2D
from keras.layers import Reshape, Lambda, BatchNormalization
from keras import applications
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from loader import TextImageGenerator, MAX_LEN, CHAR_DICT, SIZE, VizCallback, ctc_lambda_func, attention_rnn
import numpy as np
import tensorflow as tf
from keras import backend as K
import argparse

def maxpooling(base_model):
    model = Sequential(name='vgg16')
    for layer in base_model.layers[:-1]:
        if 'pool' in layer.name:
            pooling_layer = MaxPooling2D(pool_size=(2, 2), name=layer.name)
            model.add(pooling_layer)
        else:
            model.add(layer)
    return model

def get_model(input_shape, training, finetune):
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    base_model = maxpooling(base_model)
    inner = base_model(inputs)

    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    inner = Dropout(0.25)(inner) 
    inner = attention_rnn(inner)

    lstm1 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25))(inner)
    lstm2 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2', dropout=0.25, recurrent_dropout=0.25))(lstm1)
    
    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense2')(lstm2)
    
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    
    y_func = K.function([inputs], [y_pred])
    
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)

def train_kfold(idx, kfold, datapath, labelpath,  epochs, batch_size, lr, finetune, name):
    sess = tf.Session()
    K.set_session(sess)

    model, y_func = get_model((*SIZE, 3), training=True, finetune=finetune)
    ada = Adam(lr=lr)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    ## load data
    train_idx, valid_idx = kfold[idx]
    train_generator = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 16, train_idx, True, MAX_LEN)
    train_generator.build_data()
    valid_generator  = TextImageGenerator(datapath, labelpath, *SIZE, batch_size, 16, valid_idx, False, MAX_LEN)
    valid_generator.build_data()

    ## callbacks
    weight_path = 'model/{}_{}.h5'.format(name, idx)
    ckp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    vis = VizCallback(sess, y_func, valid_generator, len(valid_idx))
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

    if finetune:
        print('load pretrain model')
        model.load_weights(weight_path)

    model.fit_generator(generator=train_generator.next_batch(),
                    steps_per_epoch=int(len(train_idx) / batch_size),
                    epochs=epochs,
                    callbacks=[ckp, vis, earlystop],
                    validation_data=valid_generator.next_batch(),
                    validation_steps=int(len(valid_idx) / batch_size))
    
def train(datapath, labelpath, epochs, batch_size, lr, finetune=False, name='model'):
    nsplits = 5

    nfiles = np.arange(len(os.listdir(datapath)))

    kfold = list(KFold(nsplits, random_state=2018).split(nfiles))
    for idx in range(nsplits):
        train_kfold(idx, kfold, datapath, labelpath, epochs, batch_size, lr, finetune, name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='data/train/0916_Data Samples 2/', type=str)
    parser.add_argument("--label", default='data/train/0916_Data Samples 2/labels.json', type=str)

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--name', default='best', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    train(args.train, args.label, args.epochs, args.batch_size, args.lr, args.finetune, args.name)
    
