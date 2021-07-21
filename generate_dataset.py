import cv2
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_dataset = current_dir + "/datasets/"

vid = cv2.VideoCapture('car.mp4')

while True:
    _, frame = vid.read()
    count_files = len(os.listdir(path_to_dataset))
    file_name = f"{path_to_dataset}dataset_{count_files}.jpg"
    cv2.imwrite(file_name, frame)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

vid.realse()
cv2.destroyAllWindows()