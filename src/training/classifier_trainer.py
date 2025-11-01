import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense, DepthwiseConv2D, SeparableConv2D,
    LayerNormalization, MultiHeadAttention, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths and parameters
DATA_DIR = 'assets/dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_SAVE = 'models/cnn/detection_model.h5'

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10

# Data
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
valid_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical', shuffle=False)


# Input -> multi-branch conv stem -> several residual/separable conv blocks -> a lightweight attention block -> global pooling -> classifier head
def build_model(input_shape=(224, 224, 3), num_classes=None):
    if num_classes is None:
        num_classes = train_gen.num_classes

    x_in = Input(shape=input_shape)

    # Separable convolution to extract spatial features efficiently
    a = Conv2D(96, 3, padding='same')(x_in)
    a = BatchNormalization()(a)
    a = Activation('swish')(a)

    # Depthwise + Pointwise convolution for efficient channel-wise feature extraction
    b = DepthwiseConv2D(3, padding='same')(x) # Applies a 3x3 filter to each input channel separately.
    b = Conv2D(96, 1, padding='same')(b) # Pointwise 1x1 convolution to combine channel information.
    b = BatchNormalization()(b)
    b = Activation('swish')(b)

    # Merge branches with residual-ish connection
    m = Add()([a, b])

    # Separable/residual blocks (repeated)
    def residual_sep_block(inp, filters, repeats=2):
        y = inp
        for i in range(repeats):
            # extract richer features with much less params vs normal Conv2D
            s = SeparableConv2D(filters, 3, padding='same')(y)
            s = BatchNormalization()(s)
            s = Activation('swish')(s)
            y = Add()([y, s])
        return y

    # learn more abstract features
    m = residual_sep_block(m, 128, repeats=2)

    # allow model to expand feature capacity deeper
    m = residual_sep_block(m, 192, repeats=2)

    # Small transformer-like attention: reshape to sequence -> MHA -> add
    seq = LayerNormalization()(m) # stabilize attention scores

    try:
        # capture long-range dependencies
        att = MultiHeadAttention(num_heads=4, key_dim=32)(m, m)
        m = Add()([m, att])
    except Exception:
        # if TF build doesn't support MHA
        m = SeparableConv2D(256, 3, padding='same', activation='swish')(m)

    # reduce spatial dims and avoid huge parameter explosion
    m = GlobalAveragePooling2D()(m)

    # Classifier head
    out = Dense(num_classes, activation='softmax')(m)

    model = Model(inputs=x_in, outputs=out)
    return model


def main():
    model = build_model()
    model.compile(optimizer=Adam(1e-3), metrics=['accuracy'])

    # Train
    model.fit(train_gen, epochs=EPOCHS, validation_data=valid_gen)

    # Fine-tune
    model.compile(optimizer=Adam(1e-4), metrics=['accuracy'])
    model.fit(train_gen, epochs=5, validation_data=valid_gen)

    # Evaluate and save
    loss, acc = model.evaluate(test_gen)
    print(f'Test accuracy: {acc:.4f}')

    os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
    model.save(MODEL_SAVE)
    print(f'Model saved to {MODEL_SAVE}')


if __name__ == '__main__':
    main()
