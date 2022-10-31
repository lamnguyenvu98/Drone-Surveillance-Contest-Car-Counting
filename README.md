# Drone-Surveillance-Contest: Car Counting

## The project has the following pipelines:

1. Detection

- In order to get the position of each car, YOLOv4 was used to detect this type of object in each frame. The pre-trained weight of YOLOv4 didn't work well with the input video provided by CVZone because this video was recored from the top view, cars in this video looked a lot different than the ones from COCO dataset. Hence, training YOLOv4 on custom dataset was necessary. Extracing each frame from input video and annotating them was one way to build a custom datasset quickly.

2. Tracking

- Centroid Tracking was used to give each car an unique ID. The algorithm assumed that same object would be moved the minimum distance compared to other objects, which means the two pairs of centroids (centrer coordinate of bounding box) having minimum distance in subsequent frames are considered to be the same object. The distance was computed using euclidean distance formula.

- The biggest downside to Centroid Tracking algorithm is that a separate object detector has to be run on each and every input frame — in most situations, this behavior is undesirable as object detectors, like YOLO can be computationally expensive to run. Hence, the program will be slow. Correlation Tracker was used and played an important role to speed up this program. It can keep tracking of the ojects as it moves in subsequent frames without having to perform object detection and it run a lot faster. Correlation Tracker will return boundng box of the tracking objects then feed these bounding boxes to Centroid Tracker.

- Correlation Tracker are not able to detect new objects comming in so program only uses this algorithm until it has reached N-th frame, then re-run object detection. After that, the entire process repeats. Hence, YOLOv4 will run once after every N frames.

3. Counting

- A list was created to store ID of counted objects. If ID of an object is already in this list, the program won't count it again.

*Link to download weights*: https://drive.google.com/file/d/18EYLqz8ZHlxu3DUkqqqmBGXAHDN2uc7h/view?usp=sharing

*Place it in **models** folder*
