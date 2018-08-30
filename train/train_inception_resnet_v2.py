import datetime
import os
import sys
from argparse import ArgumentParser

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import SGD, RMSprop
from keras_applications.inception_resnet_v2 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator

from models.resetnet import inception_resnet_v2

LAYERS_TO_FREEZE = 172

LOGS_DIR = os.path.join(os.path.dirname(__file__), '../logs')
NAME_PREFIX = 'dog_vs_cat_incpetion_resnet_v2'


def transfer_learning(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=RMSprop(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def fine_tuning(model, base_model):
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


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
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_gen.flow_from_directory(
        train_dataset,
        target_size=(image_size, image_size),
        batch_size=batch_size)

    val_generator = val_gen.flow_from_directory(
        val_dataset,
        target_size=(image_size, image_size),
        batch_size=batch_size)

    model, base_model = inception_resnet_v2(train_generator.num_classes, image_size)
    if weights_file is not None:
        model.load_weights(weights_file, by_name=True, skip_mismatch=True)

    callbacks, model_file_path = create_checkpoint()

    model = transfer_learning(model, base_model)

    model.fit_generator(
        train_generator,
        epochs=30,
        steps_per_epoch=train_generator.samples / batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples / batch_size,
        callbacks=callbacks,
        verbose=1)

    model = fine_tuning(model, base_model)
    model.fit_generator(
        train_generator,
        initial_epoch=31,
        epochs=200,
        steps_per_epoch=train_generator.samples / batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples / batch_size,
        callbacks=callbacks,
        verbose=1)

    model.save('dog_vs_cat_inception_resnet_v2_{}.h5'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('val_dataset', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=150)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    main(args.train_dataset, args.val_dataset,
         epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
         image_size=args.image_size, weights_file=args.weights)
