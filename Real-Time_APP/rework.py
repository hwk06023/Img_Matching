import cv2
import threading
import sys
import os
import json
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

class GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.running = False  # Flag indicating if the camera is on
        self.isCapture = False  # Flag indicating if capture is active

        # Lists for captured images, keypoints, and descriptors computed with SIFT
        self.point_img_lst = list()
        self.kp_lst = list()
        self.des_lst = list()

        self.detector = cv2.SIFT_create()  # SIFT detector for computing keypoints and descriptors
        self.resize_imsize = (512, 512)  # Image resize size for computation reduction when using SIFT

        self.folder_path = r"./Real-Time_APP/captured_data"  # Save folder path for captured images and descriptors
        # If the folder does not exist, create it
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"Folder '{self.folder_path}' created.")

        self.capture_id = 0  # ID of saved jpg and json files

        # UI initialization of PyQt5
        self.initUI()

    def initUI(self):
        vbox = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(self)
        self.btn_start = QtWidgets.QPushButton("Camera On", self)
        self.btn_stop = QtWidgets.QPushButton("Camera Off", self)
        self.btn_capture = QtWidgets.QPushButton("Capture", self)

        # Widgets that will be displayed
        vbox.addWidget(self.label)  # For displaying the camera feed
        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_stop)
        vbox.addWidget(self.btn_capture)
        self.setLayout(vbox)

        # Link buttons with functions
        self.btn_start.clicked.connect(self.startClick)
        self.btn_stop.clicked.connect(self.stopClick)
        self.btn_capture.clicked.connect(self.captureClick)

    def run(self):
        cap = cv2.VideoCapture(0)  # 0 means the default camera
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.label.resize(int(width), int(height))  # Resize label to match the camera image size

        while self.running:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)

                # When capture is clicked, it will execute
                if self.isCapture:
                    self.capture(img, self.capture_id)
                    print("Screen is captured\n")

                    self.capture_id += 1  # Increase the ID
                    self.isCapture = False  # Turn off the flag

                self.label.setPixmap(pixmap)
            # This means reading the image was not successful
            else:
                QtWidgets.QMessageBox.about(self, "Error", "Cannot read frame.")
                print("Cannot read frame.")
                break

        cap.release()
        print("Thread end.")

    def captureClick(self):
        self.isCapture = True

    def capture(self, image, image_name):
        image_name = str(image_name).zfill(8)  # Change the integer ID to an 8-character string with 0 padding

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(image, self.resize_imsize)  # Resize image for computation reduction

        kp, des = self.detector.detectAndCompute(resized_image, None)

        # Append images and keypoints and descriptors to lists when capturing
        self.point_img_lst.append(resized_image)
        self.kp_lst.append(kp)
        self.des_lst.append(des)

        # Save image
        image_path = os.path.join(self.folder_path, image_name + ".jpg")
        cv2.imwrite(image_path, image)

        # Save keypoints and descriptors as JSON
        features_path = os.path.join(self.folder_path, image_name + ".json")
        features_data = {
            "keypoints": [
                {"pt": (kp_i.pt[0], kp_i.pt[1]), "size": kp_i.size, "angle": kp_i.angle, "response": kp_i.response,
                 "octave": kp_i.octave, "class_id": kp_i.class_id}
                for kp_i in kp
            ],
            "descriptors": des.tolist()
        }

        with open(features_path, 'w') as json_file:
            json.dump(features_data, json_file, indent=4)

    def stopClick(self):
        self.running = False

        # Initialize capture lists
        self.point_img_lst = list()
        self.kp_lst = list()
        self.des_lst = list()

        print("Stopped...")

    def startClick(self):
        self.running = True
        th = threading.Thread(target=self.run)
        th.start()
        print("Started...")

    def onExit(self):
        print("Exit")
        self.stopClick()

app = QtWidgets.QApplication([])
ex = GUI()
app.aboutToQuit.connect(ex.onExit)
ex.show()
sys.exit(app.exec_())