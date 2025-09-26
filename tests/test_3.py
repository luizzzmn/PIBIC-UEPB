import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers, models
import kagglehub
import os
import pathlib
from sklearn.model_selection import train_test_split

# CARREGAMENTO E PREPARAÇÃO DOS DADOS 

print("Carregando e preparando o dataset...")
path = kagglehub.dataset_download("rakibuleceruet/drowsiness-prediction-dataset")
data_dir = pathlib.Path(os.path.join(path, '0 FaceImages'))

active_paths = list(data_dir.glob('Active Subjects/*.jpg'))
active_labels = [0] * len(active_paths) # 0 para 'alerta'
fatigue_paths = list(data_dir.glob('Fatigue Subjects/*.jpg'))
fatigue_labels = [1] * len(fatigue_paths) # 1 para 'fadiga'

all_image_paths = [str(p) for p in (active_paths + fatigue_paths)]
all_labels = active_labels + fatigue_labels

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label

train_dataset = train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE).cache().shuffle(len(train_paths)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
validation_dataset = val_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

# CONSTRUINDO O MODELO COM TRANSFER LEARNING (MOBILENETV2)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

# CARREGA O MODELO BASE MOBILENETV2 PRÉ-TREINADO
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # CONGELANDO OS PESOS DO MODELO

# CAMADAS DE CLASSIFICAÇÃO 
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_tl = tf.keras.Model(inputs, outputs) 

print("Modelo de Transfer Learning pronto")

model_tl.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando o treinamento do modelo de Transfer Learning...")
history_tl = model_tl.fit(
    train_dataset,
    epochs=15,
    validation_data=validation_dataset
)

print("\n--- Avaliação Final do Modelo de Transfer Learning ---")
val_loss_tl, val_acc_tl = model_tl.evaluate(validation_dataset, verbose=2)
print(f"Acurácia final no conjunto de validação: {val_acc_tl * 100:.2f}%")

model_tl.save('modelo_transfer_learning.keras')
print("\n✅ Modelo de Transfer Learning salvo em 'modelo_transfer_learning.h5'")


model_tl.save('modelo_transfer_learning.keras')
print("\nModelo de Transfer Learning salvo em 'modelo_transfer_learning.keras'")


loaded_model = tf.keras.models.load_model('modelo_transfer_learning.keras')

#EXTRATOR DE CARACTERÍSTICAS

feature_extractor = tf.keras.Model(
    inputs=loaded_model.input,
    outputs=loaded_model.get_layer('global_average_pooling2d').output
    )


import pickle
with open('history_transfer_learning.pkl', 'wb') as file:
    pickle.dump(history_tl.history, file)
    
print("Modelo e histórico de treinamento salvos com sucesso!")

feature_extractor.summary()