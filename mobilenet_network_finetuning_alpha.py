import os
import tensorflow as tf
from keras_mobilenet import MobileNet
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import datetime
import argparse
from keras.models import Model
from keras import regularizers, optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from math import ceil
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from keras import backend as K
from config import config

parser = argparse.ArgumentParser(description='Yet Another Darknet To Keras Converter.')
parser.add_argument(
    '--shallow_network',
    help='Bool when network is shallowed True/',
    default=False)
args = parser.parse_args()
shallow = args.shallow_network

train_data_dir = config['train_data_dir']
validation_data_dir = config['validation_data_dir']
if shallow == True:
    best_model_path_finetuning = config['best_model_path_finetuning_shallow']
    best_model_path_pretraining = config['best_model_path_pretraining_shallow']
    model_save_pretraining_shallow = config['model_weight_save_pretraining_shallow']
    model_save = config['model_save_shallow']
    model_weight_save = config['model_weight_save_shallow']
    protobuf_file = config['protobuf_file_finetuning_shallow']
else:
    best_model_path_finetuning = config['best_model_path_finetuning']
    best_model_path_pretraining = config['best_model_path_pretraining']
    model_save_pretraining_shallow = config['model_weight_save_pretraining']
    model_save = config['model_save_finetuning']
    model_weight_save = config['model_weight_save_finetuning']
    protobuf_file = config['protobuf_file_finetuning']
input_img = config['input_img']
target_size = config['target_size']
batch_size = config['batch_size']
nb_train_samples = config['nb_train_samples']
nb_validation_samples = config['nb_validation_samples']
nb_test_samples = config['nb_test_samples']
epochs = config['epochs_finetuning']
classes = config['classes']
feartrues = config['featrues']
output_node_names = config['output_node_names']
lr = config['lr']
model_weight_save_alpha = config['model_weight_save_pretraining_alpha']
# features_train_path = 'features/bottleneck_features_train.npy'
# features_validation_path = 'features/bottleneck_features_validation.npy'

# def save_bottlebeck_features():
#     datagen = ImageDataGenerator(rescale=1. / 255)
#
#     # build the Mobile network
#     model = MobileNet(input_shape=input_img, classes=401, fine_tuning=True, include_top=False)
#     model.load_weights(best_model_path_pretraining)
#
#     generator_train = datagen.flow_from_directory(
#         train_data_dir,
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode=None,
#         shuffle=False)
#     bottleneck_features_train = model.predict_generator(generator_train, nb_train_samples // batch_size)
#     np.save(open(features_train_path, 'w'),bottleneck_features_train)
#
#     generator_validation = datagen.flow_from_directory(
#         validation_data_dir,
#         target_size=(224, 224),
#         batch_size=batch_size,
#         class_mode=None,
#         shuffle=False)
#     bottleneck_features_validation = model.predict_generator(generator_validation, nb_validation_samples // batch_size)
#     np.save(open(features_validation_path, 'w'),bottleneck_features_validation)
#
# def train_top_model():
#     train_data = np.load(open(features_train_path))
#     train_labels = np.array(
#         [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
#
#     validation_data = np.load(open(features_validation_path))
#     validation_labels = np.array(
#         [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
#
#     model = Sequential()
#     model.add(Dense(128, activation='sigmoid', activity_regularizer=regularizers.l1(0.01), name='featrues'))
#     model.add(Dense(classes, activation='softmax'))
#
#     model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#
#     model.fit(train_data, train_labels,
#               epochs=epochs,
#               batch_size=batch_size,
#               validation_data=(validation_data, validation_labels))
#     model.save_weights(top_model_weights_path)
#     model.save(top_model_path)

def finetuning():

    # build the Mobile network
    model_pretraining = MobileNet(input_shape=input_img, classes=classes, shallow=shallow, alpha= 0.5)
    model_pretraining.load_weights(model_weight_save_alpha)
    for layer in model_pretraining.layers[:83]:
        layer.trainable = False
    x = model_pretraining.layers[82].output
    x = Dense(512, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01), name='featrues')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(model_pretraining.input, x)
    model.compile(optimizer=optimizers.SGD(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    if os.path.exists('best_model/mobilenet_finetuning_shallow_alpha.h5'):
        print('====================>load model_mobilenet_weights_best_finetuning.h5')
        model.load_weights('best_model/mobilenet_finetuning_shallow_alpha.h5')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
    )
    tb_cb = TensorBoard(log_dir='logs/mobilenet', histogram_freq=1)
    checkpoint = ModelCheckpoint(filepath='best_model/mobilenet_finetuning_shallow_alpha.h5', save_weights_only=True, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    cbks = [tb_cb, checkpoint, earlystop]
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=ceil(nb_train_samples/batch_size),
        validation_data=validation_generator,
        validation_steps=ceil(nb_validation_samples/batch_size),
        class_weight='auto',
        callbacks=cbks)
    model.save_weights('mobilenet_model/mobilenet_weight_finetuning_shallow_alpha.h5')
    model.save('mobilenet_model/mobilenet_finetuning_shallow_alpha.h5')

    sess = K.get_session()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, output_node_names=[output_node_names])
    with gfile.FastGFile('mobilenet_model/mobilenet_finetuning_shallow_alpha.pb', 'wb') as f:
      f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print(str(t1))
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    #tf.add_to_collection('graph_config', config)
    # save_bottlebeck_features()
    # train_top_model()
    finetuning()
    t2 = datetime.datetime.now()
    print(str(t2-t1))