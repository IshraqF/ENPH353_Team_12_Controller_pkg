#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import String
import tensorflow as tf
import numpy as np
from rosgraph_msgs.msg import Clock
import matplotlib.pyplot as plt
from keras.models import load_model
import time
import csv

class plate_detection():
    def __init__(self):
        """
        Initialize the LineFollower class
        """
        self.bridge = CvBridge()
        # self.timer = String()
        self.plate = String()
        self.current_time = Clock()
        self.file = open('/home/fizzer/ros_ws/src/2022_competition/enph353/enph353_gazebo/scripts/plates.csv', 'r', newline='')
        self.reader = csv.reader(self.file)
        self.rows = []

        # Loop over each row in the CSV file and append it to the list
        for row in self.reader:
            self.rows.append(row)
        # self.done = 0
        self.loc = 1
        self.inital_time = 0
        self.done = 0
        self.state = 0
        self.timer_counter = 0
        self.current_time_counter = 0
        self.count = 0
        self.conv_model = load_model('/home/fizzer/ros_ws/src/controller_pkg/node/License_PLate_Model.h5', compile = False)
        # self.inital_time = 0
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.callback)
        self.pub_plate = rospy.Publisher('/license_plate', String, queue_size=1)
        self.sub_clock = rospy.Subscriber('/clock', Clock, self.clock_callback)
        self.count_time_after = 0
        self.time_counter = 0

    def clock_callback(self, msg):
        self.current_time = msg.clock
    
    def callback(self, data):
        """
        Callback function for image data
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)
        
        # print(self.conv_model.summary())
        self.plate.data = "Team12,multi21,"
        if (self.done == 0):
            self.plate.data = "Team12,multi21,0,FA12"
            print("start timer sent")
            self.pub_plate.publish(self.plate)
            self.done = 1
            self.inital_time = self.current_time.secs
        
        if (self.current_time.secs > self.inital_time+100 and self.done == 1): #change this back to the desired time
            self.plate.data = "Team12,multi21,-1,FA12"
            self.pub_plate.publish(self.plate)
            self.done = 2
            
        print("callback")
        cv2.imshow('callback_stream', cv_image)
        cv2.waitKey(1)
        
        self.detect_car(cv_image)
    
    def save_image(self, warped):
        plate = "P" + str(self.loc)
        filename = ""
        if self.count < 7:
            filename = "plate_{}{}.png".format(plate, str(self.rows[self.count]))
            self.count += 1
        else:
            while(True):
                print("AHHHHHHHH")
        if filename:
            cv2.imwrite(filename, warped)
    
    def publish_plate(self, warped):
        if(self.current_time.secs > self.current_time_counter+1):
            print("publish at")
            print(self.current_time.secs)
            self.pub_plate.publish(self.plate)
            self.save_image(warped)
            self.loc += 1
            #time.sleep(3)
            self.plate.data = "Team12,multi21,"
            self.current_time_counter = self.current_time.secs
            #self.callback

    def display_plate(self, img):
        print("display plate")
        cv2.imshow('display_plate_image',img)
        cv2.waitKey(1)
        
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #print(img.shape)
        # Preprocess the image
        img = cv2.resize(img, (100, 100))
        # img = img / 255.0
        img_aug = np.expand_dims(img, axis=0)


        # Predict the character
        y_predict = self.conv_model.predict(img_aug)[0]
        predictedChar = np.argmax(y_predict)
        print(predictedChar)
        if predictedChar < 26:
            predictedChar = chr(predictedChar + 65)  # A-Z
        else:
            predictedChar = str(predictedChar - 26)  # 0-9
        
        return str(predictedChar)

        # Display the image and prediction
        #plt.imshow(img)
        #caption = f"Predicted Character: {predictedChar}"
        #plt.text(0.5, 0.5, caption, color='orange', fontsize=16,
                #horizontalalignment='left', verticalalignment='bottom')
        # cv2_imshow(img)
        # print(predictedChar)
    
    def find_corner_points(self, points):
        min_x_idx = np.argmin(points[:, 0])
        min_x_points = points[points[:, 0] == points[min_x_idx, 0]]
        min_y_idx = np.argmin(min_x_points[:, 1])
        smallest_x_y = min_x_points[min_y_idx]
        
        max_x_idx = np.argmax(points[:, 0])
        max_x_points = points[points[:, 0] == points[max_x_idx, 0]]
        min_y_idx = np.argmin(max_x_points[:, 1])
        largest_x_smallest_y = max_x_points[min_y_idx]

        max_x_idx = np.argmax(points[:, 0])
        max_x_points = points[points[:, 0] == points[max_x_idx, 0]]
        max_y_idx = np.argmax(max_x_points[:, 1])
        largest_x_y = max_x_points[max_y_idx]

        min_x_idx = np.argmin(points[:, 0])
        min_x_points = points[points[:, 0] == points[min_x_idx, 0]]
        max_y_idx = np.argmax(min_x_points[:, 1])
        smallest_x_largest_y = min_x_points[max_y_idx]

        return smallest_x_y, largest_x_smallest_y, largest_x_y, smallest_x_largest_y
    
    def find_plate_corners(self, img):
        # Find the contours again so the later functions are satisfied
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour with a polygon
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Ensure that the polygon has 4 vertices
        assert len(approx) == 4, "The contour isn't a rectangle!"
        corner_points = np.squeeze(approx)

        return np.array(corner_points.tolist())

    
    def sort_contours_list(self, contours_list):
        centroid = np.mean(contours_list, axis=0)

        # Sort the points by their angle from the centroid
        sorted_points = sorted(contours_list, key=lambda point: np.arctan2(point[1]-centroid[1], point[0]-centroid[0]))

        return sorted_points
    
    def cutout_letters(self, img, warped):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_bound = np.array([86, 107, 0])
        upper_bound = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        gray = 255 - gray
        masked_img = cv2.bitwise_and(gray,gray,mask = mask)
        #masked_img = cv2.bitwise_not(masked_img)
        # cv2_imshow(masked_img)
        #small blur just to reduce small noise
        #could increase this to 5 if we wanted
        masked_img = cv2.GaussianBlur(masked_img, (3,3), 0)
        mask = masked_img

        #now i need to overlay the mask on the gray image and make sure its grayscaled
        #i can grayscale the image again before 
        letters = [mask[340:390, 30:105], mask[340:390, 95:167], mask[340:390, 233:305], mask[340:390, 295:367]]
        count = 0

        for imagez in letters:
            char = self.display_plate(imagez)
            self.plate.data += char
            cv2.imshow("Plate number", imagez)
            cv2.waitKey(1)
            count += 1

        if self.loc < 9 and count == 4:
            self.publish_plate(warped)
        else:
            self.loc = 7
    
    def cutout_plate_number(self, img, warped):
        img = img[10:300, 10:375]

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([0, 0, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        filtered_img = cv2.bitwise_and(img, img, mask = mask)

        gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        for i, contour in enumerate(contours):
            # Obtain bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Crop image using NumPy array slicing
            character_image = thresh[y-10:y+h+10, x-10:x+w+10]

            if i == 1:
                # char = self.display_plate(character_image)
                # self.plate.data += char
                #self.loc += 1
                #----------------------
                self.plate.data += str(self.loc)
                self.plate.data += ","

            # cv2.imshow("Plate Location", character_image)
            # cv2.waitKey(1)
            # now = datetime.datetime.now()
            # filename = f"warped_image_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}.png"
            # cv2.imwrite(filename, images)


            #SHOULD PROBABLY TURN THIS TO A MORE BLACK AND WHITE IMAGE!
            # print(images.shape)
            # cv2.imshow("Plate Location", images)
            # cv2.waitKey(1)
    
    def detect_plate(self, image):
        #Find the mask of the image given an hsv threshold range
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([0, 0, 100]) #originally 92
        upper_bound = np.array([0, 0, 250]) #originally 205

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        filtered_image = cv2.bitwise_and(image, image, mask = mask)

        #Find the contours of the image
        gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]

        if len(contours) > 2:
            #Draw the contours on a black image of the same size
            sortedContours = sorted(contours, key = cv2.contourArea, reverse = True)
            black_img = np.zeros_like(image)
            black_img = cv2.GaussianBlur(black_img,(5,5),0)

            cv2.drawContours(black_img, [sortedContours[0]], -1, (255, 255, 255), 2)

            #Use the binary contour image to find the corners
            contours_list = self.find_plate_corners(black_img)

            #Sort the corners so they are always in the same order
            contours_list = self.sort_contours_list(contours_list)

            #Assign the corners to points and use PerspectiveTransform to warp the plates
            pts1 = np.float32([contours_list[0], contours_list[1], contours_list[2], contours_list[3]])
            height, width = 400, 400
            pts2 = np.array([(0, 0), (width-1, 0), (width-1, height-80), (1, height-80)], dtype="float32")
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(image, M, (width, height))



        # cv2.imshow("Warped", warped)
        # print("^^warped")

            self.cutout_plate_number(warped, warped)
            self.cutout_letters(warped, warped)
    
    def detect_car(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([0, 125, 62])
        upper_bound = np.array([0, 255, 222])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        filtered_img = cv2.bitwise_and(image, image, mask = mask)

        gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]

        if len(contours) > 2:
            sortedContours = sorted(contours, key = cv2.contourArea, reverse = True)

            if cv2.contourArea(sortedContours[0]) > 35000 and cv2.contourArea(sortedContours[0]) < 50000:
                points = sortedContours[0][:, 0, :]

                smallest_x_y, largest_x_smallest_y, largest_x_y, smallest_x_largest_y = self.find_corner_points(points)

                corner_points = np.array([smallest_x_y, largest_x_smallest_y, largest_x_y, smallest_x_largest_y])

                img_height, img_width, img_channels = image.shape

                # - is left and + is right
                center_x, center_y = img_width/2, img_height/2
                
                if (smallest_x_y[0] - center_x) > 0:
                    # h = largest_x_y[1] - largest_x_smallest_y[1]
                    # w = img_width - largest_x_smallest_y[0]
                    x, y, w, h = cv2.boundingRect(sortedContours[0])
                    wt = img_width - (x+w)
                    # crop = image[largest_x_smallest_y[1]:largest_x_smallest_y[1]+h, largest_x_smallest_y[0]:largest_x_smallest_y[0]+w]
                    crop = image[y:y+h, x+w:x+w+wt]
                else:
                    # h = smallest_x_largest_y[1] - smallest_x_y[1]
                    # w = smallest_x_y[0]
                    x, y, w, h = cv2.boundingRect(sortedContours[0])
                    # crop = image[smallest_x_y[1]:smallest_x_y[1]+h, 0:w]
                    crop = image[y:y+h, 0:x+50]
                
                self.detect_plate(crop)


# Main program execution.
if __name__ == "__main__":
    rospy.init_node('plate_detection')
    plate = plate_detection()
    rospy.spin()