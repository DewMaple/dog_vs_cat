import keras.backend as K
from keras import layers, Model, Input
from keras.applications import ResNet50, InceptionResNetV2
from keras.engine import Layer
from keras.layers import Embedding
from keras.regularizers import l2


class CenterLoss(Layer):
    def call(self, inputs, **kwargs):
        return K.sum(K.square(inputs[0] - inputs[1][:, 0]), 1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1


def inception_resnet_v2(num_classes, input_size=150):
    base_model = InceptionResNetV2(include_top=False,
                                   weights='imagenet',
                                   input_shape=(input_size, input_size, 3),
                                   pooling='max')
    x = base_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1])(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    features = layers.PReLU(shared_axes=[1])(x)

    prediction = layers.Dense(num_classes, activation='softmax', name='fc_prediction')(features)
    model = Model(inputs=base_model.input, outputs=prediction)
    model.summary()
    return model, base_model


def build_train_model(num_classes, input_size):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3), pooling=None)
    #
    # for layer in base_model.layers:
    #     layer.trainable = False
    #
    x = base_model.output
    x = layers.MaxPooling2D(input_shape=base_model.layers[-1].output_shape[1:])(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024, kernel_regularizer=l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, kernel_regularizer=l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    features = layers.Activation(activation='relu')(x)

    prediction = layers.Dense(num_classes, activation='softmax', name='fc_prediction')(features)

    input_target = Input(shape=(1,))  # single value ground truth labels as inputs
    centers = Embedding(num_classes, 256)(input_target)
    l2_loss = CenterLoss()([features, centers])

    model_center_loss = Model(inputs=[base_model.input, input_target], outputs=[prediction, l2_loss])

    return model_center_loss, base_model


def build_resnet50_model(num_classes, input_size):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3), pooling=None)

    x = base_model.output
    x = layers.MaxPooling2D(input_shape=base_model.layers[-1].output_shape[1:])(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1])(x)
    x = layers.Dropout(0.75)(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    features = layers.PReLU(shared_axes=[1])(x)

    prediction = layers.Dense(num_classes, activation='softmax', name='fc_prediction')(features)
    model = Model(inputs=base_model.input, outputs=prediction)
    return model, base_model


if __name__ == '__main__':
    inception_resnet_v2(2)
