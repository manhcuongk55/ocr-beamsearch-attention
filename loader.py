import json
import cv2
import os, random
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.layers import multiply, Dense, Permute, Lambda, RepeatVector
import itertools
import editdistance
from lib.random_eraser import get_random_eraser
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint
from PIL import Image
import json
random.seed(2018)

letters = " !\"#&\\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
MAX_LEN = 70
WIDTH, HEIGHT = 1280, 64
SIZE = WIDTH, HEIGHT
CHAR_DICT = len(letters) + 1

chars = letters
wordChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
corpus = ' \n '.join(json.load(open('labels.json')).values())
word_beam_search_module = tf.load_op_library('lib/TFWordBeamSearch.so')
mat=tf.placeholder(tf.float32)
beamsearch_decoder = word_beam_search_module.word_beam_search(mat, 25, 'Words', 0.1, corpus, chars, wordChars)

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))


def beamsearch(sess, y_pred):
    y_pred = y_pred.transpose((1, 0, 2))
    results = sess.run(beamsearch_decoder, {mat:y_pred[2:]})
    blank=len(chars)
    results_text = []
    for res in results:
        s=''
        for label in res:
            if label==blank:
                break
            s+=chars[label] # map label to char
        results_text.append(s)
    return results_text

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def attention_rnn(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    timestep = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Dense(timestep, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

class VizCallback(keras.callbacks.Callback):
    def __init__(self, sess, y_func, text_img_gen, text_size, num_display_words=3):
        self.y_func = y_func
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        self.text_size = text_size
        self.sess = sess

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen.next_batch())[0]
            num_proc = min(word_batch['the_inputs'].shape[0], num_left)
            # predict
            inputs = word_batch['the_inputs'][0:num_proc]
            pred = self.y_func([inputs])[0]
            decoded_res = beamsearch(self.sess, pred)#decode_batch(pred)
            # label
            labels = word_batch['the_labels'][:num_proc].astype(np.int32)
            labels = [labels_to_text(label) for label in labels]
            
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], labels[j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(labels[j])

            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: '
              '%.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        batch = next(self.text_img_gen.next_batch())[0]
        inputs = batch['the_inputs'][:self.num_display_words]
        labels = batch['the_labels'][:self.num_display_words].astype(np.int32)
        labels = [labels_to_text(label) for label in labels]
         
        pred = self.y_func([inputs])[0]
        pred_beamsearch_texts = beamsearch(self.sess, pred)
        #pred_texts = decode_batch(pred)
        for i in range(min(self.num_display_words, len(inputs))):
            print("label: {} - predict: {}".format(labels[i], pred_beamsearch_texts[i]))

        self.show_edit_distance(self.text_size)

class TextImageGenerator:
    def __init__(self, img_dirpath, labels_path, img_w, img_h,
                 batch_size, downsample_factor, idxs, training=True, max_text_len=9, n_eraser=5):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.idxs = idxs
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.labels= json.load(open(labels_path)) if labels_path != None else None
        self.img_dir = sorted(os.listdir(self.img_dirpath))     # images list
        random.shuffle(self.img_dir)

        if self.idxs is not None:
            self.img_dir = [self.img_dir[idx] for idx in self.idxs]

        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.ones((self.n, self.img_h, self.img_w, 3), dtype=np.float16)
        self.training = training
        self.n_eraser = n_eraser
        self.random_eraser = get_random_eraser(s_l=0.004, s_h=0.005, r_1=0.01, r_2=1/0.01, v_l=-128, v_h=128)
        self.texts = []
        image_datagen_args = {
		'shear_range': 0.1,
		'zoom_range': 0.01,
		'width_shift_range': 0.001,
		'height_shift_range': 0.1,
		'rotation_range': 1,
		'horizontal_flip': False,
		'vertical_flip': False
	}
        self.image_datagen = ImageDataGenerator(**image_datagen_args)

    def build_data(self):
        print(self.n, " Image Loading start... ", self.img_dirpath)
        for i, img_file in enumerate(self.img_dir):
            img = image.load_img(self.img_dirpath + img_file, target_size=SIZE[::-1], interpolation='bicubic')
            img = image.img_to_array(img)
            img = preprocess_input(img)
            self.imgs[i] = img
            if self.labels != None: 
                self.texts.append(self.labels[img_file][:MAX_LEN])
            else:
                #valid mode
                self.texts.append('')
        print("Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]].astype(np.float32), self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.zeros([self.batch_size, self.img_w, self.img_h, 3], dtype=np.float32)     # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_text_len], dtype=np.float32)             # (bs, 9)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()

                if self.training:
                    params = self.image_datagen.get_random_transform(img.shape)
                    img = self.image_datagen.apply_transform(img, params)
                    if randint(0, 1) == 1:
                        for _ in range(self.n_eraser):
                            img = self.random_eraser(img)
                        img = elastic_transform(img, 10, 2, 0.1)

                img = img.transpose((1, 0, 2))
                # random eraser if training
                X_data[i] = img
                Y_data[i,:len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_inputs': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }

            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)
