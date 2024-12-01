# Fruit Quality Detection System

This repository contains the implementation of a fruit quality detection system using Convolutional Neural Networks (CNN) and Arduino integration. The project classifies fruits into 24 categories and identifies their quality as 'fresh' or 'rotten.' The system integrates text-to-speech (TTS) functionality and communicates the results to an Arduino.

## Features
- Classifies images of fruits into 24 categories.
- Detects and announces fruit quality (fresh/rotten) using text-to-speech.
- Sends classification results to an Arduino via serial communication.
- Outputs the message:
  - `<Quality> food detected` for valid fruit detections.
  - `No food detected` for non-fruit images.

---

## Project Structure
- **Model Training**: A deep CNN architecture implemented with TensorFlow and Keras for fruit quality classification.
- **Arduino Integration**: Sends classification results to Arduino for further processing or actuation.
- **Text-to-Speech**: Provides audio feedback based on the classification.

---

## Dataset and Preprocessing
The dataset consists of 24 classes of fruits categorized by quality (e.g., `fresh_Apple`, `rotten_Apple`). It is organized into folders, and the images are processed with the following parameters:
```python
training_set = tf.keras.utils.image_dataset_from_directory(
    'Train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)
```

---

## Model Architecture
The CNN model uses a sequential architecture with the following layers:
- **Convolutional Layers**: Extract features using 2D convolutions with ReLU activation.
- **Pooling Layers**: Downsample feature maps with MaxPooling.
- **Dropout**: Regularization to reduce overfitting.
- **Dense Layers**: Fully connected layers for classification.

The output layer uses a **softmax** activation function to classify images into 24 categories.

### Model Summary
```python
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Additional convolutional and pooling layers
cnn.add(tf.keras.layers.Dropout(0.25))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=1500, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.4)) # Regularization
cnn.add(tf.keras.layers.Dense(units=24, activation='softmax'))
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Integration with Arduino
Using Python's `serial` library, the system communicates classification results to an Arduino. Based on the inference:
- If a fruit is detected, it sends:
  - `good` for fresh fruits
  - `bad` for rotten fruits
- If no fruit is detected, it sends no signal.

### Example Communication Flow
1. **Detection**: Fresh apple is detected.
2. **Laptop to Arduino Message**: Sends `good` via serial.
3. **Arduino Ouput**: built-in led blinks for 15 times with 200 milli-seconds delay

---

## Running the Project
### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Arduino IDE and hardware
- `pyserial` library for serial communication

### Steps
1. Clone the repository.
   ```bash
   git clone https://github.com/SiPAyi/quality_fruits_seperator.git
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Train the CNN model or use the pre-trained weights provided.
4. Connect the Arduino and run the Python script in Testing_final_model.ipynb for integration and inference.

---

## Results
- The model achieves accurate classification of fruit quality.
- Real-time feedback via text-to-speech and Arduino enhances usability.
- Supports diverse fruit types with high reliability.

---

## Contributors
- **Sai Kottapeta(Me)** (Integrating model, camera feed, text-to-speech functionality, Arduino code)
- **Ganesh** (Data collection, Data pre-processing, Model training)
- **ChatGPT** (Code and Documentation Assistance)

---

## Demo
Watch the system in action!  
[![Demo Video](https://img.shields.io/badge/Demo-Video-blue)](https://github.com/SiPAyi/quality_fruits_seperator/blob/main/VID_20241201_130236.mp4)

---

## Acknowledgments
- TensorFlow and Keras for the deep learning framework.
- Arduino for hardware integration.
- University project team and advisors for their guidance.

