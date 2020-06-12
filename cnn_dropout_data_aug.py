from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

###### Constants ######
train_dir = '/tmp2/b07902053/mango-data/training'
validation_dir = '/tmp2/b07902053/mango-data/testing'
result_acc_path = './results/cnn3-acc.png'
result_loss_path = './results/cnn3-loss.png'

###### Set memory growth of GPUs (optional) ######
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

#solve_cudnn_error()

###### Model Configuration ######
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, kernel_initializer='random_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(3, kernel_initializer='random_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('softmax'))

model.summary()

model.compile(optimizer=Adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

###### Image Generator with Aumentation ######
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')
print(validation_generator.class_indices)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

####### Model Training #######
checkpoint = ModelCheckpoint("best_cnn3.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

# use model.fit_generator() for old keras version
history = model.fit(
    train_generator,
    steps_per_epoch=175, # 5600 / 32
    epochs=100,
    validation_data=validation_generator,
    validation_steps=25, # 800 / 32
    callbacks=[checkpoint])

# print(history.history)

####### Plot Results #######
import matplotlib.pyplot as plt

acc = history.history['accuracy']
# acc = history.history['acc'] # for old Keras version
val_acc = history.history['val_accuracy']
# val_acc = history.history['val_acc'] # for old Keras version
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig, ax = plt.subplots(1)
ax.plot(epochs, acc, 'bo', label='Training acc')
ax.plot(epochs, val_acc, 'b', label='Validation acc')
ax.set_title('Training and validation accuracy')
ax.legend()
fig.savefig(result_acc_path)

fig, ax = plt.subplots(1)
ax.plot(epochs, loss, 'bo', label='Training loss')
ax.plot(epochs, val_loss, 'b', label='Validation loss')
ax.set_title('Training and validation loss')
ax.legend()
fig.savefig(result_loss_path)
plt.close()

####### Clean Up Session #######
K.clear_session()
