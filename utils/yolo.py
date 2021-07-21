import numpy as np
import cv2
import argparse



def yolo_net(path_weights, path_cfg, use_GPU=False):
    net = cv2.dnn.readNet(path_weights, path_cfg)
    if use_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def yolo_output(net, img, shape= 256):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (shape, shape), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    return outputs

def yolo_predict(outputs, conf_threshold, H, W):
    bbox, classIds, confs = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > conf_threshold:
                w, h = int(detection[2] * W), int(detection[3] * H)
                x, y = int(detection[0] * W - w / 2), int(detection[1] * H - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classid)
                confs.append(float(confidence))
    return bbox, classIds, confs
