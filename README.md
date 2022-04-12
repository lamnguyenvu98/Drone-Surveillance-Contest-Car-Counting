# Drone-Surveillance-Contest: Car Counting

- In this contest, in order to count number of cars, I used Yolov4 to detect cars then send bounding box coordinates to Centroid tracker algorithm (euclidean distance based) and Correlation Tracker (algorithm in Dlib library) to track cars and give them specific ID number. The whole idea of object tracking was implemented by Adrian Roseborg, thanks to his amazingly helpful blog: https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

- In order to train Yolov4, I first created my own dataset by writing small script, which took each frame of the video and wrote them to dataset folder. After that, I wrote a notebook and train Yolov4 on my dataset in Google Colab.

- The Centroid Tracker algorithm mostly depends on running object detection on every frame, which cost a lot of computational resources. Correlation Tracker was used to keep ID of each car and their bounding box coordinate while object detection wasn't running, so overall the program run slightly faster. In my program, only every 20 frames will I run my Yolov4 for object detection. During that every 20-frame period, I use correlation tracker to track objects.

*Link to download weights*: https://drive.google.com/file/d/18EYLqz8ZHlxu3DUkqqqmBGXAHDN2uc7h/view?usp=sharing
*Place it in **models** folder*
