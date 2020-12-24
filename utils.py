from PIL import ImageFont, ImageDraw, Image
import pickle
import numpy as np
import cv2 as cv
import os
import random


def create_training_data(Categories, Directory):
    training_data = []
    for category in Categories:
        path = os.path.join(Directory, category)
        category_num = Categories.index(category)
        for img in os.listdir(path):
            img_array = cv.imread(os.path.join(path, img), 0)
            training_data.append([img_array, category_num])

    random.shuffle(training_data)

    return training_data


def store_processed_data(X_train, X_test, y_train, y_test):
    pickle_out = open("Resources/_TrainingData/X_train.pickle", "wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("Resources/_TrainingData/X_test.pickle", "wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("Resources/_TrainingData/y_train.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open("Resources/_TrainingData/y_test.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()


def retrieve_processed_data():
    pickle_in = open("Resources/_TrainingData/X_train.pickle", "rb")
    X_train = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("Resources/_TrainingData/X_test.pickle", "rb")
    X_test = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("Resources/_TrainingData/y_train.pickle", "rb")
    y_train = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("Resources/_TrainingData/y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)
    pickle_in.close()

    return X_train, X_test, y_train, y_test


def put_text(img, text, x_pos, y_pos, height, text_color):
    font_path = "ArialCE.ttf"
    font = ImageFont.truetype(font_path, 20)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x_pos, y_pos + height + 5), text, font=font, fill=text_color)
    img = np.array(img_pil)

    return img
