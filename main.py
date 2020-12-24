from utils import put_text
from PIL import ImageFont, ImageDraw, Image
from tensorflow import keras
import cv2 as cv
import numpy as np
import random

## Parameters
step = 2  # Detect shape every nth frame
padding = 8  # Padding added to the matrix so that shape doesn't touch the edges
area_min = 1000  # The minimum area required for a shape to be recognized
rect_color = (0, 220, 0)  # Color of the bounding rectangles
rect_thickness = 2  # Thickness of rectangles
text_color = (0, 220, 0)  # Color of text
font_size = 0.6  # Size of text font
text_thickness = 2  # Thickness of text

model = keras.models.load_model("ShapeDetectionModel")


def getContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    images_ = []

    xList = []
    yList = []
    wList = []
    hList = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > area_min:
            perimeter = cv.arcLength(cnt, True)

            approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
            x_pos, y_pos, width, height = cv.boundingRect(approx)

            xList.append(x_pos)
            yList.append(y_pos)
            wList.append(width)
            hList.append(height)

            cv.rectangle(imgContours, (x_pos, y_pos), (x_pos + width, y_pos + height), rect_color, rect_thickness)

            if y_pos > 20 and y_pos + height < 480 and x_pos > 20 and x_pos + width < 480:
                imgSegmented = IMG[y_pos - 20:y_pos + height + 20, x_pos - 20:x_pos + width + 20]
            else:
                imgSegmented = IMG[y_pos:y_pos + height, x_pos:x_pos + width]

            imgSegmented = cv.resize(imgSegmented, (s_matrix_size, s_matrix_size))

            images_.append(imgSegmented)

    return images_, xList, yList, wList, hList


def convert_to_binary(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    areas = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        areas.append(area)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area == max(areas):
            cv.drawContours(imgBinary, cnt, -1, (255, 255, 255), 2)


cap = cv.VideoCapture('Resources/camera_images/video.mp4')

counter = 0
last_prediction = ''
last_predictions = ['', '', '']
s_matrix_size = 56 - (2 * padding)

while True:
    success, IMG = cap.read()

    if success:
        IMG = cv.resize(IMG, (500, 500))

        imgBinary = np.zeros((s_matrix_size, s_matrix_size))
        imgContours = IMG.copy()

        shapesGray = cv.cvtColor(IMG, cv.COLOR_BGR2GRAY)
        shapesBlur = cv.GaussianBlur(shapesGray, (7, 7), 1)

        median = np.median(shapesGray)
        shapesCanny = cv.Canny(shapesBlur, int(max(0, (1.0 - 0.33) * median)), int(min(255, (1.0 + 0.33) * median)))

        images, x_list, y_list, w_list, h_list = getContours(shapesCanny)

        if len(images) > 0 and counter % step == 0:

            random_number = random.randint(0, len(images) - 1)

            x = x_list[random_number]
            y = y_list[random_number]
            w = w_list[random_number]
            h = h_list[random_number]

            shapesGray = cv.cvtColor(images[random_number], cv.COLOR_BGR2GRAY)
            shapesBlur = cv.GaussianBlur(shapesGray, (7, 7), 1)

            median = np.median(shapesGray)
            shapesCanny = cv.Canny(shapesBlur, int(max(0, (1.0 - 0.33) * median)),
                                   int(min(255, (1.0 + 0.33) * median)))

            convert_to_binary(shapesCanny)

            shapesCanny = cv.resize(shapesCanny, (500, 500))

            imgBinary = imgBinary / 255
            imgBinary = np.pad(imgBinary, padding)

            class_names = ["Circle", "Square", "Triangle"]
            prediction = class_names[np.argmax(model.predict(imgBinary.reshape(-1, 56, 56, 1)))]

            imgContours = put_text(imgContours, prediction, x, y, h, text_color)

            last_predictions[random_number] = prediction

            for i in range(0, len(images)):
                if i != random_number:
                    x = x_list[i]
                    y = y_list[i]
                    w = w_list[i]
                    h = h_list[i]

                    shapesGray = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)
                    shapesBlur = cv.GaussianBlur(shapesGray, (7, 7), 1)

                    median = np.median(shapesGray)
                    shapesCanny = cv.Canny(shapesBlur, int(max(0, (1.0 - 0.33) * median)),
                                           int(min(255, (1.0 + 0.33) * median)))

                    convert_to_binary(shapesCanny)

                    shapesCanny = cv.resize(shapesCanny, (500, 500))

                    imgContours = put_text(imgContours, last_predictions[i], x, y, h, text_color)

        elif len(images) > 0:
            for i in range(0, len(images)):
                x = x_list[i]
                y = y_list[i]
                w = w_list[i]
                h = h_list[i]

                shapesGray = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)
                shapesBlur = cv.GaussianBlur(shapesGray, (7, 7), 1)

                median = np.median(shapesGray)
                shapesCanny = cv.Canny(shapesBlur, int(max(0, (1.0 - 0.33) * median)),
                                       int(min(255, (1.0 + 0.33) * median)))

                convert_to_binary(shapesCanny)

                shapesCanny = cv.resize(shapesCanny, (500, 500))

                imgContours = put_text(imgContours, last_predictions[i], x, y, h, text_color)

        cv.imshow("Output", imgContours)
        counter += 1

    if cv.waitKey(5) & 0xFF == ord('q'):
        break
