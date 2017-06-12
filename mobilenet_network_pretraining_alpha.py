import os
import tensorflow as tf
from keras_mobilenet import MobileNet
from keras import optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import datetime
import argparse
from math import ceil
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from config import config

parser = argparse.ArgumentParser(description='Yet Another Darknet To Keras Converter.')

parser.add_argument(
    '--shallow_network',
    help='Bool when network is shallowed True/',
    default=True)

parser.add_argument(
    '--alhpa',
    help='parameter of the network to change the width of model',
    default=0.25)

args = parser.parse_args()
shallow = args.shallow_network
alhpa = args.alhpa
train_data_dir = config['train_data_dir']
validation_data_dir = config['validation_data_dir']

input_img = config['input_img']
target_size = config['target_size']
batch_size = config['batch_size']
nb_train_samples = config['nb_train_samples']
nb_validation_samples = config['nb_validation_samples']
epochs = config['epochs_pretraining']
classes = config['classes']
output_node_names = config['output_node_names']
lr = config['lr_pretraining']
if alhpa == 1:
    if shallow == True:
        best_model_path_pretraining = config['best_model_path_pretraining_shallow']
        model_save = config['model_save_pretraining_shallow']
        model_weight_save = config['model_weight_save_pretraining_shallow']
        protobuf_file = config['protobuf_file_pretraining_shallow']
    else:
        best_model_path_pretraining = config['best_model_path_pretraining']
        model_save = config['model_save_pretraining']
        model_weight_save = config['model_weight_save_pretraining']
        protobuf_file = config['protobuf_file_pretraining']
elif alhpa == 0.25:
    if shallow == True:
        best_model_path_pretraining = config['best_model_path_pretraining_shallow_alpha256']
        model_save = config['model_pretraining_shallow_alpha256']
        model_weight_save = config['weight_pretraining_shallow_alpha256']
        protobuf_file = config['protobuf_pretraining_shallow_alpha256']
    else:
        best_model_path_pretraining = config['best_model_path_pretraining_alpha256']
        model_save = config['model_save_pretraining_alpha256']
        model_weight_save = config['model_weight_save_pretraining_alpha256']
        protobuf_file = config['protobuf_file_pretraining_alpha256']
def pretraining():
    # Training
    model = MobileNet(input_shape=input_img, classes=classes, shallow=shallow, alpha=alhpa)
    model.compile(optimizer=optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    if os.path.isfile(best_model_path_pretraining):
        print("load pretraining train model")
        model.load_weights(best_model_path_pretraining)

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
    checkpoint = ModelCheckpoint(filepath=best_model_path_pretraining,save_weights_only=True, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=20)
    cbks = [tb_cb, checkpoint]
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=ceil(nb_train_samples/batch_size),
        validation_data=validation_generator,
        validation_steps=ceil(nb_validation_samples/batch_size),
        class_weight='auto',
        callbacks=cbks)

    model.save(model_save)
    model.save_weights(model_weight_save)

    sess = K.get_session()
    output_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names=[output_node_names])
    with gfile.FastGFile(protobuf_file, 'wb') as f:
          f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    print(str(t1))
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    pretraining()
    t2 = datetime.datetime.now()
    print(str(t2-t1))