import cv2
import numpy as np

import eng_to_ipa as ipa

import threading
import sys
import os
import json
from glob import glob

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

class GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.running = False # flag if camera is on
        self.isCapture = False # flag if capture is on
        self.isLoad = False
        
        # lists for captured images, keypoints, descriptions that computed with SIFT
        #self.point_img_lst = list()
        self.kp_lst = list()
        self.des_lst = list()
        
        self.image_name_lst = list()
        self.image_name_ipa_lst = list()
        
        self.detector = cv2.SIFT_create() # SIFT detector for compute keypoints and descriptions
        self.resize_imsize = (512, 512) # image resize size for computation reduction when compute SIFT
        
        self.folder_path = r"./captured_data" # save folder path for captured images and descriptions
        # if folder does not exist, create folder
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"Folder '{self.folder_path}' created.")
            
        self.capture_id = 0 # id of saved jpg and json
        
        # UI initialization of PyQt5
        self.initUI()
    
    def initUI(self):
        vbox = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(self)
        self.btn_start = QtWidgets.QPushButton("Camera On", self)
        self.btn_stop = QtWidgets.QPushButton("Camera Off", self)
        self.btn_capture = QtWidgets.QPushButton("Capture", self)
        self.btn_load = QtWidgets.QPushButton("Load", self)
        
        self.capture_line_edit = QtWidgets.QLineEdit(self)
        self.load_line_edit = QtWidgets.QLineEdit(self)
        
        # widgets that will be displayed
        vbox.addWidget(self.label) # for camera display
        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_stop)
        vbox.addWidget(self.capture_line_edit)
        vbox.addWidget(self.btn_capture)
        vbox.addWidget(self.load_line_edit)
        vbox.addWidget(self.btn_load)
        self.setLayout(vbox)
        
        # link button with functions
        self.btn_start.clicked.connect(self.startClick)
        self.btn_stop.clicked.connect(self.stopClick)
        self.btn_capture.clicked.connect(self.captureClick)
        self.btn_load.clicked.connect(self.loadClick)
    
    def run(self):
        cap = cv2.VideoCapture(0) # 0 means camera
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.label.resize(int(width), int(height)) # label to camera image size

        while self.running:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                h,w,c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                
                # when capture is clicked, it will execute
                if self.isCapture:
                    image_name = self.capture_line_edit.text()
                    for image_name in self.image_name_lst:
                        print("The name that you enter is already exist. Try another name.")
                        image_name = self.capture_line_edit.text()
                    self.capture_line_edit.setText("")
                        
                    self.image_name_lst.append(image_name)
                    self.image_name_ipa_lst.append(ipa.convert(image_name))
                    
                    #self.capture(img, self.capture_id)
                    self.capture(img, image_name)
                    print("Screen is captured\n")
               
                    #self.capture_id += 1 # incread id
                    self.isCapture = False # turn off the flag
                
                if self.isLoad:
                    flag = 1
                    
                    features_paths = glob(f"{self.folder_path}/*.json")
                    features_fnames = [f"{os.path.basename(features_path)}"[:-5] for features_path in features_paths]
                    print(features_fnames)
                    if self.load_line_edit.text() != "$ALL":
                        input_features_fnames = self.load_line_edit.text().split(' ')
                        print(input_features_fnames)
                        
                        if not all(element in features_fnames for element in input_features_fnames):
                            flag = 0
                        else:
                            features_paths = [f"{self.folder_path}/{features_fname}.json" for features_fname in input_features_fnames]
                    
                    if flag:
                        self.load(features_paths)
                        print(f"{[os.path.basename(features_path) for features_path in features_paths]} features are loaded")
                    else:
                        print(f"Some files in your input do not exist. No features are loaded")
                    
                    
                    self.isLoad = False
                    
                self.label.setPixmap(pixmap)
            # this means reading image is not successful
            else:
                QtWidgets.QMessageBox.about(self, "Error", "Cannot read frame.")
                print("cannot read frame.")
                break

        cap.release()
        print("Thread end.")
    
    def captureClick(self):
        self.isCapture = True
        
    def loadClick(self):
        self.isLoad = True
    
    def stopClick(self):
        self.running = False
        
        # initialize capture list
        #self.point_img_lst = list()
        self.kp_lst = list()
        self.des_lst = list()
        
        self.image_name_lst = list()
        self.image_name_ipa_lst = list()
        
        print("stoped..")
    
    def startClick(self):
        self.running = True
        th = threading.Thread(target=self.run)
        th.start()
        print("started..")
    
    def capture(self, image, image_name):
        #image_name = str(image_name).zfill(8) # change int id to 8 string with 0 padding
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(image, self.resize_imsize) # resize image for computation reduction

        kp, des = self.detector.detectAndCompute(resized_image, None)
        
        # append images and kp and des to list when capture
        #self.point_img_lst.append(resized_image)
        self.kp_lst.append(kp)
        self.des_lst.append(des)
        
        # save image
        image_path = os.path.join(self.folder_path, image_name + ".jpg")
        cv2.imwrite(image_path, image)
        
        # save kp, des as json
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
    
    def load_keypoints_and_descriptors(self, features_path):
        with open(features_path, 'r') as json_file:
            features_data = json.load(json_file)

        keypoints = [cv2.KeyPoint(x=pt["pt"][0], y=pt["pt"][1], size=pt["size"], angle=pt["angle"],
                                  response=pt["response"], octave=pt["octave"], class_id=pt["class_id"])
                     for pt in features_data["keypoints"]]

        descriptors = np.array(features_data["descriptors"], dtype=np.float32)

        return keypoints, descriptors
    
    def load(self, features_paths):
        for features_path in features_paths:
            #image_path = f"{self.folder_path}/{os.path.basename(features_path).split('.')[0]+'.jpg'}"
            #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #resized_image = cv2.resize(image, self.resize_imsize) # resize image for computation reduction
            
            keypoints, descriptors = self.load_keypoints_and_descriptors(features_path)
            
            #self.point_img_lst.append(resized_image)
            self.kp_lst.append(keypoints)
            self.des_lst.append(descriptors)
    
    def onExit(self):
        print("exit")
        self.stopClick()

app = QtWidgets.QApplication([])
ex = GUI()
app.aboutToQuit.connect(ex.onExit)
ex.show()
sys.exit(app.exec_())