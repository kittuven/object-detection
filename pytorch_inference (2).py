import math
import os
from ultralytics import YOLO
import cv2
import pandas as pd
import csv
import datetime
import time

def line_length(x1, y1, x2, y2):
    length = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return length


def area_rectangle(x1, y1, x2, y2):
    l1 = line_length(x1, y1, x2, y1)
    l2 = line_length(x2, y2, x2, y1)
    area = l1 * l2
    return area, l1, l2


def cvt_time(number):
    time_min = 0
    time_sec = 0
    if number >= 60:
        time_min = number // 60
        time_sec = number - (time_min * 60)
    else:
        time_min = 0
        time_sec = number
    return time_min, time_sec

frame = cv2.imread('img.jpg')
H, W, _ = frame.shape

model_path = 'best.pt'

# Load a model
model = YOLO(model_path)  # load a custom model
# print(model.eval())

threshold = 0.25
classes = { 0: 'bus', 1: 'car', 2: 'motor', 3: 'person', 4: 'truck'}
id = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
width, height = 0, 0
results = model(frame)[0]
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    area, width, height = area_rectangle(x1, y1, x2, y2)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    class_text = classes[class_id]
    id[class_id] = id[class_id] + 1
    text = class_text + str(id[class_id])
    cv2.putText(frame, text, (int(x1), int(y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA, False)



img = cv2.resize(frame, (1000, 800))
cv2.imshow("test", img)
cv2.waitKey(0)



