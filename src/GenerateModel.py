import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths
data_dir = 'assets/dataset'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
model_save_path = 'models/cnn/EfficientNetB0.h5'

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
valid_gen = valid_datagen.flow_from_directory(
    valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_gen,
    epochs=epochs,
    validation_data=valid_gen
)

# Optionally fine-tune
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_gen,
    epochs=5,
    validation_data=valid_gen
)

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f'Test accuracy: {acc:.4f}')

# Save
model.save(model_save_path)
print(f'Model saved to {model_save_path}')
