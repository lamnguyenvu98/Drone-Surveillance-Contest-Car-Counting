import cv2
import numpy as np
from utils.centroidTracker import CentroidTracker
from utils.trackableobject import TrackableObject
from random import randint
from collections import deque
import dlib, time
from utils.yolo import *
import argparse

classes = [c.strip() for c in open('models/car.names').readlines()]
conf_threshold = 0.6  # lay confidence > 0.5
nmsThreshold = 0.4  # > 0.5 se ap dung Non-max Surpression
shape = 416
colors = []
colors.append([(randint(0, 255), randint(0, 255), randint(0, 255)) for i in range(1000)])
#detected_classes = ['car', 'bus', 'truck', 'train']
ct = CentroidTracker(maxDisappeared=20, maxDistance=50)
pts = [deque(maxlen=10) for _ in range(1000)]
counter = 0
center = None
trackers = []
empty = []
trackableObjects = {}
totalFrames = 0
totalCar = 0
(W, H) = (None, None)
net = yolo_net("models/yolov4_training_last.weights", "models/yolov4_testing.cfg")
out = cv2.VideoWriter('result/car_count2.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (1920, 1080))
vid = cv2.VideoCapture("car.mp4")
while True:
    _, img = vid.read()
    h_img, w_img = img.shape[:2]
    ratio = 1500 / w_img
    img = cv2.resize(img, (1500, int(h_img * ratio)))

    title, img = img[:121, :], img[120:h_img+1, :]
    h_img, w_img = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    status = "Waiting"
    rects = []
    if totalFrames % 15 == 0:
        status = "Detecting"
        trackers = []
        outputs = yolo_output(net, img, shape)
        bbox, classIds, confs = yolo_predict(outputs, conf_threshold, h_img, w_img)
        indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nmsThreshold)
        for i in indices:
            i = i[0]
            #if classes[classIds[i]] not in detected_classes: continue
            box = bbox[i]
            color = colors[0][i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, x + w, y + h)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    else:
        for idx, tracker in enumerate(trackers):
            status = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            cv2.rectangle(img, (startX, startY), (endX, endY), colors[0][idx], 2)
            rects.append((startX, startY, endX, endY))
    
    obj = ct.update(rects)
    for (objectID, centroid) in obj.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            oy = [c[1] for c in to.centroids]
            directionY = centroid[1] - np.mean(oy)
            to.centroids.append(centroid)
            if not to.counted:
                if directionY < 0 and centroid[1] < (781 - 120):
                    totalCar += 1
                    empty.append(totalCar)
                    to.counted = True

        trackableObjects[objectID] = to

        text = "ID {}".format(objectID)
        colorID = colors[0][objectID]
        cv2.circle(img, (centroid[0], centroid[1]), 4, colorID, -1)
        cv2.putText(img, text, (centroid[0] - 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        center = (centroid[0], centroid[1])
        pts[objectID].append(center)
        for i in range(1, len(pts[objectID])):
            if pts[objectID][i - 1] is None or pts[objectID][i] is None:
                continue
            thickness = int(np.sqrt(10 / float(i + 1)) * 2.5)
            cv2.line(img, pts[objectID][i - 1], pts[objectID][i], colorID, thickness)

    cv2.putText(title, "Vu Lam Nguyen", (688, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
    cv2.putText(title, f"{totalCar}", (1336, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
    cv2.line(img, (0, 781 - 120), (w_img, 781 - 120), (0, 0, 0), 2)
    
    result = np.vstack((title, img))
    cv2.imshow('Result', result)
    result = cv2.resize(result, (1920, 1080))
    out.write(result)
    key = cv2.waitKey(1)
    if key == ord('q'): break
    totalFrames += 1

vid.release()
out.release()
cv2.destroyAllWindows()
