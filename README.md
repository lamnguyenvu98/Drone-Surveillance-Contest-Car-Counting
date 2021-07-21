# Drone-Surveillance-Contest: Car Counting

In order to count number of cars in the footage, I need to go through 2 parts in a pipeline: Detect cars then track them and give each one of them specific ID

In detection part:

- I trained YOLOv4 model to detect cars. I wrote a small script to write each frame in the footage to dataset folder.
- I created 2 subfolders in dataset folder named as train and valid.
- I took 10% of images to valid folder and the rest was placed in train folder.
- I used labelImg to label all images in both train folder and valid folder.
- I wrote simple notebook file to train yolov4 on google colab and train my model.
- The yolov4 returned class_id, bounding box coordinate (center_x, center_y, width, height) and confidence score.

*Link to download weights*: https://drive.google.com/file/d/18EYLqz8ZHlxu3DUkqqqmBGXAHDN2uc7h/view?usp=sharing
*Place it in models folder*

In tracking part:
- I used CentroidTracker (euclidean distance based) to track object and give them ID number. 
- In order to count car, I draw a horizontal line in the footage to visualize if cars are above this line then count them. 
- I create a dictionary (key is ID of object and value is a boolean datatype) - to specify if object ID above this line then set it as counted (True) so I don't have to count them again.
- To archieve faster processing, I used correlation tracker in dlib to track object and run yolo detection once after every 30 frames (because object detection models always cost a lot of computational resources so I skipped the detection part for 20 frames then run yolo detection again).

The idea of tracking part was from Adrian Roseborg and his amazing blog: https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

I want to thank Mr Murtaza and also Adrian Rosborg for their amazingly helpful tutorials. They help me so much taking my first step in Computer Vision's field.
