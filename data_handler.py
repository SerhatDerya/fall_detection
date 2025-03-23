import os
import matplotlib.pyplot as plt
import cv2

class Fall_dataset():

    def __init__(self, mode):
        self.mode = mode

        input_path = 'fall_dataset/images/' + self.mode
        self.img_files = os.listdir(input_path)
        self.img_files.sort()

        label_path = 'fall_dataset/labels/' + self.mode
        self.label_files = os.listdir(label_path)
        self.label_files.sort()

        print(f"Number of {self.mode} images: ", len(self.img_files))
        print(f"Number of {self.mode} labels: ", len(self.label_files))
        # the first column of the label file is the label [0:fallen, 1:standing, 2:sitting]

        # read the images (with bounding boxes) and labels
        self.images = []
        self.labels = []

        print("Reading the images and extracting bounding boxes...")

        for i in range(len(self.img_files)):
            img = plt.imread(input_path + '/' + self.img_files[i])
            with open(label_path + '/' + self.label_files[i], 'r') as f:
                r = f.readlines()
            bounding_boxes = []
            for j in r:
                j = j.split()
                bounding_boxes.append([int(j[0]), float(j[1]), float(j[2]), float(j[3]), float(j[4])])
            for box in bounding_boxes:
                image_height, image_width, _ = img.shape
                label, xmin, ymin, width, height = box[0], int(box[1]*image_width), int(box[2]*image_height), int(box[3]*image_width), int(box[4]*image_height)
                self.labels.append(label)
                self.images.append(img[ymin-height//2:ymin+height//2, xmin-width//2:xmin+width//2])

        print(f"Number of {self.mode} images: ", len(self.images))
        print(f"Number of {self.mode} labels: ", len(self.labels))

        print("Resizing extracted images...")

        # resize the images to a preferred size
        pref_size = (128,128)
        for i in range(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], pref_size)

        print("Images and labels are ready!")

    def getAll(self):
        return self.images, self.labels

    def __getitem__(self, index):
        img = self.images[index]
        plt.imshow(img)
        plt.show()
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)



'''
# get train and test image and label files
train_img_files = os.listdir('fall_dataset/images/train')
train_img_files.sort()
train_path = 'fall_dataset/images/train/'

train_label_files = os.listdir('fall_dataset/labels/train')
train_label_files.sort()
label_path = 'fall_dataset/labels/train/'

print("Number of training images: ", len(train_img_files))
print("Number of training labels: ", len(train_label_files))
# the first column of the label file is the label [0:fallen, 1:standing, 2:sitting]

# read the images (with bounding boxes) and labels

complete_images = []
complete_labels = []

print("Reading the images and extracting bounding boxes...")

for i in range(len(train_img_files)):
    img = plt.imread('fall_dataset/images/train/' + train_img_files[i])
    with open(label_path + train_label_files[i], 'r') as f:
        r = f.readlines()
    bounding_boxes = []
    for j in r:
        j = j.split()
        bounding_boxes.append([int(j[0]), float(j[1]), float(j[2]), float(j[3]), float(j[4])])
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        label, xmin, ymin, width, height = box[0], int(box[1]*image_width), int(box[2]*image_height), int(box[3]*image_width), int(box[4]*image_height)
        complete_labels.append(label)
        complete_images.append(img[ymin-height//2:ymin+height//2, xmin-width//2:xmin+width//2])

print("Number of complete images: ", len(complete_images))
print("Number of complete labels: ", len(complete_labels))

print("Resizing extracted images...")
# resize the images to a preferred size
pref_size = (128,128)
for i in range(len(complete_images)):
    complete_images[i] = cv2.resize(complete_images[i], pref_size)

print("Images and labels are ready!")


img = complete_images[390]
plt.imshow(img)
plt.show()
'''