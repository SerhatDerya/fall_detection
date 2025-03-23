import os.path

import cv2
from tensorflow import keras
import pandas as pd
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2


class fall_detection_model():
    def __init__(self):
        self.model = keras.Sequential([
            # First Convolution Layer
            #keras.layers.Conv2D(filters=16, kernel_size=(3,3), input_shape=(128,128,3), activation='relu', kernel_regularizer=l2(0.0066)),
            keras.layers.Conv2D(filters=16, kernel_size=(3,3), input_shape=(128,128,3), activation='relu'),
            keras.layers.MaxPooling2D(),

            # Second Convolution Layer
            #keras.layers.Conv2D(32, (3,3), padding='same' , activation='relu', kernel_regularizer=l2(0.0066)),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),

            # Third Convolution Layer
            #keras.layers.Conv2D(64, (3,3) , activation='relu', kernel_regularizer=l2(0.0066)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            #keras.layers.MaxPooling2D(),

            # Flatten layer to transition from Convolutional to Dense Layers
            keras.layers.Flatten(),

            # Dense (Fully Connected) Layer
            #keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.0066)),
            keras.layers.Dense(128, activation='relu'),

            # Output Layer
            keras.layers.Dense(3, activation='softmax')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.0034)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, images, labels, num_epochs=8):
        self.df = pd.DataFrame()
        self.df["Images"] = images
        self.df["Images"] /= 255
        self.df["Labels"] = labels

        X_train = np.array(self.df["Images"].tolist())
        y_train = np.array(self.df["Labels"])
        print(X_train.shape)
        print(y_train.shape)
        history = self.model.fit(X_train, y_train, validation_split=0.25 , epochs=num_epochs)
        return history

    def predict(self, img, bounding_boxes):
        complete_images = []
        predicts = []
        nb = 0
        if bounding_boxes is not None:
            for box in bounding_boxes:
                image_height, image_width, _ = img.shape
                xmin, ymin, width, height = box[:]
                xmin = int(xmin * image_width)
                ymin = int(ymin * image_height)
                width = int(width * image_width)
                height = int(height * image_height)
                complete_images.append(img[ymin - height // 2:ymin + height // 2, xmin - width // 2:xmin + width // 2])
        else:
            complete_images.append(img)
            nb += 1

        for cropped_img in complete_images:
            plt.imshow(cropped_img)
            plt.axis('off')
            plt.show()
            cropped_img_resized = cv2.resize(cropped_img, (128, 128))
            plt.imshow(cropped_img_resized)
            plt.show()
            cropped_img_resized = cropped_img_resized / 255.0
            cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)
            prediction = self.model.predict(cropped_img_resized)
            k = np.argmax(prediction)
            predicts.append(k)
            if (k == 0):
                print("Fall detected")
            elif (k == 1):
                print("No fall detected. Person is walking or standing")
            else:
                print("No fall detected. Person is sitting")

        return predicts, nb

    def evaluate(self, images, labels):
        self.df = pd.DataFrame()
        self.df["Images"] = images
        self.df["Images"] /= 255
        self.df["Labels"] = labels

        X_test = np.array(self.df["Images"].tolist())
        y_test = np.array(self.df["Labels"])
        return self.model.evaluate(X_test, y_test)

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


class yolo_model():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes = [0]

    def print_and_save_bounds(self, results, image_path):
        img = Image.open(image_path)
        image_width, image_height = img.size
        img_filename = os.path.splitext(os.path.basename(image_path))[0]

        filename = f"results/{img_filename}.txt"
        with open(filename, "w") as f:
            boxes = []
            for i, detection in enumerate(results.pandas().xyxy[0].values):
                x_min, y_min, x_max, y_max, confidence, class_id, _ = detection.tolist()
                if(confidence>0.25):
                    label = results.names[int(class_id)]
                    class_label = class_id
                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    norm_width = (x_max - x_min) / image_width
                    norm_height = (y_max - y_min) / image_height

                    print(f"Labels: {x_center} {y_center} {norm_width} {norm_height} {confidence}")
                    print(f"Object {i+1}: {label} (Confidence: {confidence:.2f})")
                    f.write(f"{label} {x_center} {y_center} {norm_width} {norm_height} {confidence}\n")
                    temp = [x_center, y_center, norm_width, norm_height]
                    boxes.append(temp)
            return boxes

    def draw_bounding_box(self, img_array, object_number, x, y, x_plus_w, y_plus_h):
        color = (0,255,0) #BGR
        return cv2.rectangle(img_array, (x,y), (x_plus_w,y_plus_h), color, 2)

    def detect_objects(self, image_path):
        img = Image.open(image_path)

        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = self.model(img, size=640)
        #results.show()
        boxes = self.print_and_save_bounds(results, image_path)

        num_bounding_boxes = len(results.xyxy[0])
        print(f"Number of Objects Detected: {num_bounding_boxes}")

        # Draw bounding boxes
        for i in range(num_bounding_boxes):
            x, y, x_plus_w, y_plus_h, _, _ = results.xyxy[0][i]
            imgBox = self.draw_bounding_box(img_array, i+1, round(x.item()), round(y.item()), round(x_plus_w.item()), round(y_plus_h.item()))
            return boxes
            # return boxes, imgBox

