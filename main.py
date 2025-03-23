import cv2
import matplotlib.pyplot as plt
import data_handler
import my_models
import os
import numpy as np
from PIL import Image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = my_models.fall_detection_model()
    model.load("fall_detection_model_14.weights.h5")
    yolo_model = my_models.yolo_model()
    print(model.summary())

    '''
    img_path = 'fall_dataset/fall001.jpg'
    img = plt.imread(img_path)
    bounding_boxes, imgBoxes = yolo_model.detect_objects(img_path)
    model.predict(img, bounding_boxes)
    '''



    '''
    img = Image.fromarray(imgBoxes, 'RGB')
    img.save('my.png')
    img.show()
    '''


    mode = 'val'
    label_path = 'fall_dataset/labels/' + mode
    label_files = os.listdir(label_path)
    label_files.sort()

    TP = 0 # Fall - Detected
    FP = 0 # Not Fall - Detected
    TN = 0 # Not Fall - Not Detected
    FN = 0 # Fall - Not Detected

    image_path = 'fall_dataset/images/' + mode
    image_files = os.listdir(image_path)
    image_files.sort()

    nb = 0

    for i,f in enumerate(image_files):
        #f = f.replace(' ', '_')
        img = plt.imread('fall_dataset/images/' + mode + '/' + f)
        bounding_boxes = yolo_model.detect_objects('fall_dataset/images/' + mode + '/' + f)
        predicts, no_box = model.predict(img, bounding_boxes)
        nb += no_box

        with open('fall_dataset/labels/' + mode + '/' + label_files[i], 'r') as file:
            labels = file.readlines()
            for j in range(len(labels)):
                labels[j] = int(labels[j].split()[0])

        if  (0 in labels) and (0 in predicts):
            TP += 1
        elif (0 in labels) and (0 not in predicts):
            FN += 1
        elif (0 not in labels) and (0 in predicts):
            FP += 1
        elif (0 not in labels) and (0 not in predicts):
            TN += 1


        print("TP: ", TP)
        print("FP: ", FP)
        print("TN: ", TN)
        print("FN: ", FN)
        print("NB: ",nb)

    print("Accuracy: ", (TP + TN) / (TP + FP + TN + FN))
    print("Precision: ", TP / (TP + FP))
    print("Recall: ", TP / (TP + FN))
    print("F1 Score: ", 2 * TP / (2 * TP + FP + FN))
    print("No box: ", nb)


