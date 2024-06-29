import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam, RMSprop

# Function to load and preprocess images
def load_images_from_folder(folder, label=False, img_size=(48, 48)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=img_size, color_mode='grayscale')
        img_array = img_to_array(img)
        img_array /= 255.0  # Normalizing the image
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

# Define a function to create a CNN model
def create_cnn_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(2, activation='softmax')
    ])
    
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn_grid_search(X_train, y_train, param_grid, cv=3):
    best_score = 0
    best_params = {}
    
    for optimizer in param_grid['optimizer']:
        for dropout_rate in param_grid['dropout_rate']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    print(f"Training with optimizer={optimizer}, dropout_rate={dropout_rate}, batch_size={batch_size}, epochs={epochs}")
                    
                    model = create_cnn_model(optimizer=optimizer, dropout_rate=dropout_rate)
                    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
                    
                    val_accuracy = max(history.history['val_accuracy'])
                    if val_accuracy > best_score:
                        best_score = val_accuracy
                        best_params = {
                            'optimizer': optimizer,
                            'dropout_rate': dropout_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }
    
    return best_score, best_params