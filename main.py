import cv2
import numpy as np
from utils.centroidTracker import CentroidTracker
from utils.trackableobject import TrackableObject
from random import randint
from collections import deque
import dlib
from utils.yolo import *
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--shape", default=416, type=int, help="input shape of detection model (should be divisible by 32)")
ap.add_argument("-o", "--output", type=str, help="path to write result")
ap.add_argument("-m", "--disappeared", default=20, type=int, help="Provide number of frame to delete ID of object if they don't comeback")
ap.add_argument("-d", "--distance", default=50, type=int, help="threshold number if distance below this value, assign new ID to object")
ap.add_argument("-f", "--frame", default=20, type=int, help="number of frame to skip")
ap.add_argument("-c", "--confidence", default=0.6, type=float, help="confidence threshold of detection model")
ap.add_argument("-nms", "--nonmax", default=0.4, type=float, help="non-max surpression threshold of detection model")
ap.add_argument("-l", "--class", default="models/car.names", type=str, help="path to class label file")
ap.add_argument("-w", "--weight", default="models/yolov4_training_last.weights", type=str, help="path to weight file")
ap.add_argument("-g", "--cfg", default="models/yolov4_testing.cfg", type=str, help="path to config file")
ap.add_argument("-v", "--video", default="car.mp4", type=str, help="path of processing video")
args = vars(ap.parse_args())

classes = [c.strip() for c in open(args["class"]).readlines()]
conf_threshold = args["confidence"]
nmsThreshold =args["nonmax"] 
shape = args["shape"]
if shape % 32 != 0:
    print("[ERROR] Shape value should be divisible by 32")
    sys.exit(1)
colors = []
colors.append([(randint(0, 255), randint(0, 255), randint(0, 255)) for i in range(1000)])
ct = CentroidTracker(maxDisappeared=20, maxDistance=50)
pts = [deque(maxlen=10) for _ in range(1000)]
counter = 0
center = None
trackers = []
empty = []
trackableObjects = {}
totalFrames = 0
totalCar = 0
net = yolo_net(args["weight"], args["cfg"])
out = None
if args["output"] is not None and out is None:
    out = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*'XVID'), 25, (1920, 1080), True)
vid = cv2.VideoCapture(args["video"])
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
    if totalFrames % args["frame"] == 0:
        status = "Detecting"
        trackers = []
        outputs = yolo_output(net, img, shape)
        bbox, classIds, confs = yolo_predict(outputs, conf_threshold, h_img, w_img)
        indices = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nmsThreshold)
        for i in indices:
            i = i[0]
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
    for (objectID, bbox) in obj.items():
        to = trackableObjects.get(objectID, None)
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        centroid = (int((x2 - x1) / 2), int((y2 - y1) / 2))

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
        bbox_thick = int(0.6 * (h_img + w_img) / 100)
        t_size = cv2.getTextSize(text, 0, 0.5, thickness=bbox_thick // 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), colorID, 3)
        cv2.rectangle(img, (x1,y2), (x1 + t_size[0]+7, y2 - t_size[1] - 5), colorID, -1) #filled
        cv2.putText(img, text, (x1+2, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(title, "Vu Lam Nguyen", (688, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
    cv2.putText(title, f"{totalCar}", (1336, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
    cv2.line(img, (0, 781 - 120), (w_img, 781 - 120), (0, 0, 0), 2)
    
    result = np.vstack((title, img))
    cv2.imshow('Result', result)
    result = cv2.resize(result, (1920, 1080))
    if out is not None:
        out.write(result)
    key = cv2.waitKey(1)
    if key == 27: break
    totalFrames += 1

vid.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
