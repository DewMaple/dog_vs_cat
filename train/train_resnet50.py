import datetime
import os
import sys
from argparse import ArgumentParser

import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator

from models.resetnet import build_resnet50_model

LOGS_DIR = os.path.join(os.path.dirname(__file__), '../logs')
NAME_PREFIX = 'dog_vs_cat_resnet50'


class OnEpochEnd(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))

        print('Optimizer: {}, Epoch is {}, origin lr: {}, current lr is {}'.format(
            self.model.optimizer.__class__.__name__, epoch, K.eval(lr), K.eval(lr_with_decay)))


def create_checkpoint():
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "{}/{}_{}".format(LOGS_DIR, NAME_PREFIX, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    f_name = '{epoch:02d}_{val_acc:.4f}'
    model_file_path = '{}/{}_{}_{}.h5'.format(log_dir, NAME_PREFIX, f_name, filename)

    checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=1, period=1)
    return [checkpoint, tensor_board, OnEpochEnd()], model_file_path


def main(train_dataset, val_dataset, epochs, batch_size, lr=0.001, image_size=224, weights_file=None):
    train_generator = ImageDataGenerator(rescale=1 / 255.,
                                         rotation_range=10,
                                         zoom_range=0.2,
                                         horizontal_flip=True)

    train_gen = train_generator.flow_from_directory(train_dataset,
                                                    target_size=(image_size, image_size),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

    val_generator = ImageDataGenerator(rescale=1 / 255.,
                                       rotation_range=10,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_gen = val_generator.flow_from_directory(val_dataset,
                                                target_size=(image_size, image_size),
                                                class_mode='binary')

    model, base_model = build_resnet50_model(train_gen.num_classes, image_size)
    if weights_file is not None:
        print('Loading weights file: {}'.format(weights_file))
        model.load_weights(weights_file, by_name=True, skip_mismatch=True)

    callbacks, model_file_path = create_checkpoint()

    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=SGD(lr, momentum=0.9),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.summary()
    model.fit_generator(train_gen,
                        initial_epoch=0,
                        epochs=epochs,
                        steps_per_epoch=train_gen.samples / batch_size,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=val_gen.samples / batch_size,
                        verbose=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('val_dataset', type=str)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    main(args.train_dataset, args.val_dataset, args.epochs, args.batch_size, args.lr, args.image_size, args.weights)
