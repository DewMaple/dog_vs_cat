import datetime
import os
import sys
from argparse import ArgumentParser

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

sys.path.append('/Users/jcui/MachineLearning/dog_vs_cat/')

from models.resetnet import build_train_model

LOGS_DIR = os.path.join(os.path.dirname(__file__), '../logs')
NAME_PREFIX = 'dog_vs_cat'


def create_checkpoint():
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "{}/{}_{}".format(LOGS_DIR, NAME_PREFIX, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    f_name = '{epoch:02d}_{val_acc:.4f}'
    model_file_path = '{}/{}_{}_{}.h5'.format(log_dir, NAME_PREFIX, f_name, filename)

    checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=1, period=1)
    return [checkpoint, tensor_board], model_file_path


def main(train_dataset, val_dataset, epochs, batch_size, lr=0.001, image_size=224, weights_file=None):
    train_gen = ImageDataGenerator(
        brightness_range=(0.6, 1),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1. / 255).flow_from_directory(train_dataset, target_size=(image_size, image_size))

    val_gen = ImageDataGenerator(rescale=1. / 255). \
        flow_from_directory(val_dataset, target_size=(image_size, image_size))

    model = build_train_model(train_gen.num_classes, image_size)
    if weights_file is not None:
        print('Loading weights file: {}'.format(weights_file))
        model.load_weights(weights_file, by_name=True, skip_mismatch=True)

    # model.compile(optimizer=Adam(lr=lr, decay=lr / 100), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=lr, decay=lr / 100),
                  loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred],
                  loss_weights=[1, 0.5],
                  metrics={'fc_prediction': 'accuracy'})

    model.summary()

    callbacks, model_file_path = create_checkpoint()
    steps_per_epoch = train_gen.samples / batch_size
    val_steps = val_gen.samples / batch_size

    def gen_wrapper(generator):

        while True:
            x, y = next(generator)
            random_y = np.random.rand(x.shape[0], 1)

            x_ = [x, np.argmax(y, axis=1)]
            y_ = [y, random_y]
            yield x_, y_

    model.fit_generator(gen_wrapper(train_gen),
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks,
                        validation_data=gen_wrapper(val_gen),
                        validation_steps=val_steps,
                        verbose=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('val_dataset', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    main(args.train_dataset, args.val_dataset, args.epochs, args.batch_size, args.lr, args.image_size, args.weights)
