#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
import numpy as np



class LineFollower:
    def __init__(self):
        self.bridge = CvBridge()
        self.move = Twist()
        self.timer = String()
        self.done = 0
        self.inital_time = 0
        self.current_time = Clock()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.pub_move = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.pub_timer = rospy.Publisher('/license_plate', String, queue_size=1)
        self.sub_clock = rospy.Subscriber('/clock', Clock, self.clock_callback)

        self.state = 0
        self.time_counter = 0
        self.red_crosswalk_counter1 = 0
        self.detect_pedestrians_counter = 0
        self.red_crosswalk_counter2 = 0
        self.crosswalk_counter = 0
        self.start_crosswalk_time = 0
        self.initial_time_counter = 0
    
    def clock_callback(self, msg):
        self.current_time = msg.clock

    def callback(self, data):
        self.publish_move()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)
        try:
            cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #self.state = 99
        print(self.state)
        #cv2.imshow('stream', cv_image2)
        #cv2.waitKey(1)

        if (self.state == 0):
            self.time_counter = 0
            self.red_crosswalk_counter1 = 0
            self.detect_pedestrians_counter = 0
            self.red_crosswalk_counter2 = 0

            self.follow_line(cv_image)
            self.check_crosswalk(cv_image)
            self.publish_move()
        elif (self.state == 1):
            self.check_crosswalk(cv_image)
            self.publish_move()
        elif (self.state == 2): #grassy hill section
            self.follow_hill_line(cv_image2)
            self.publish_move()
        elif (self.state == 3):
            self.follow_inner_loop(cv_image)
            self.publish_move()

        elif (self.state == 99): #For testing only
            #self.check_crosswalk(cv_image)
            self.follow_inner_loop(cv_image)
            #self.crosswalk_green_follow(cv_image)
            #self.follow_line(cv_image)
            #self.publish_move()
        else:
            self.move.linear.x = 0
            self.move.angular.z = 0
            self.publish_move()

    def check_crosswalk(self, cv_image):
        red_crosswalk = cv_image[:,:,1]
        blurred_red_crosswalk = cv2.GaussianBlur(red_crosswalk, (5, 5), cv2.BORDER_DEFAULT)
        ret, binary_red_crosswalk = cv2.threshold(blurred_red_crosswalk, 50, 255, cv2.THRESH_BINARY)

        red_line = 255 - binary_red_crosswalk[600:700, 540:740]
        red_line_bottom = 255 - binary_red_crosswalk[680:700, 540:740]

        if (np.mean(red_line_bottom) > 100):
            self.red_crosswalk_counter2 = 1

        #If a crosswalk is detected
        if(self.red_crosswalk_counter2 == 1):
            print("detected crosswalk")
            self.state = 1
            if(self.red_crosswalk_counter1 == 0):
                self.move.linear.x = 0
                self.move.angular.z = 0
                self.pub_move.publish(self.move)
                self.red_crosswalk_counter1 = 1
                self.crosswalk_counter = 1 + self.crosswalk_counter
            #detect movement through middle of the screen
            if(self.detect_pedestrians(cv_image) != 0):
                self.detect_pedestrians_counter = 1
            
            if(self.detect_pedestrians_counter == 1):
                if (self.time_counter == 0):
                    self.start_crosswalk_time = self.current_time.secs
                    self.time_counter = 1

                if(self.crosswalk_green_follow(cv_image) == 1):
                    self.state = 0
                    if(self.crosswalk_counter == 2):
                            self.state = 2

    def crosswalk_green_follow(self, cv_image):
        print("green follow")

        hsv_filter_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        hsv_lo = np.array([0,0,0])
        hsv_hi = np.array([0,255,255])
        mask = cv2.inRange(hsv_filter_image, hsv_lo, hsv_hi)
        image = mask[600:, 600:]

        rows,cols = image.shape
        pts1 = np.float32([[618,115],[435,5],[10,100]]) #points on the original image
        pts2 = np.float32([[618,115],[618,5],[10,100]]) #points on the desired image (previous points get mapped/stretched to this) (need at least 3)

        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(image,M,(cols,rows))
        image = dst[:,300:678]

        kernel = np.ones((5,5),np.uint8)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

        moments = cv2.moments(gradient)
        red_crosswalk = cv_image[:,:,1]
        blurred_red_crosswalk = cv2.GaussianBlur(red_crosswalk, (5, 5), cv2.BORDER_DEFAULT)
        ret, binary_red_crosswalk = cv2.threshold(blurred_red_crosswalk, 50, 255, cv2.THRESH_BINARY)
        red_line_bottom = 255 - binary_red_crosswalk[697:700, 540:740]

        # find the center of the binary_image
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"]) + 700
        else:
            # set values to zero
            center_x, center_y = 0, 0
        error = center_x - 310
        self.move.linear.x = 0.3
        self.move.angular.z = -0.02 * error - 0.002

        if(np.mean(red_line_bottom) > 1 and self.current_time.secs > self.start_crosswalk_time+0.8):
            print("return 1")
            return 1
        else:
            return 0

    def detect_pedestrians(self, cv_image):

        pedestrian_cutout = cv_image[400:450, 350:950]
        pedestrian_cutout_hsv = cv2.cvtColor(pedestrian_cutout, cv2.COLOR_RGB2HSV)

        hsv_lo = np.array([0,0,0])
        hsv_hi = np.array([90,255,255])
        mask = cv2.inRange(pedestrian_cutout_hsv, hsv_lo, hsv_hi)

        mask = 255 - mask
        centroid_mask = cv2.moments(mask)
        if(centroid_mask["m00"] != 0):
            cX = int(centroid_mask["m10"] / centroid_mask["m00"])
        else:
            cX=1000000 #blow it up

        if(abs(cX - 300) < 50):
            return 1
        else:
            return 0

    def follow_line(self, cv_image):
        grayscaled_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grayscaled_image, (5, 5), cv2.BORDER_DEFAULT)
        ret, binary_image = cv2.threshold(blurred_image, 85, 255, cv2.THRESH_BINARY)

        # invert the binary image to convert the track from black to white
        binary_image = 255-binary_image

        # crop the image to focus on the bottom part only
        image = binary_image[500:]

        red_image = cv_image[:,:,1]
        blurred_red_image = cv2.GaussianBlur(red_image, (5, 5), cv2.BORDER_DEFAULT)
        ret, binary_red_image = cv2.threshold(blurred_red_image, 50, 255, cv2.THRESH_BINARY)
        binary_red_image = binary_red_image[500:]

        image = cv2.bitwise_and(binary_red_image,image)
        # find the center of the track
        moments = cv2.moments(image)

        # find the center of the binary_image
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"]) + 700
        else:
            # set values to zero
            center_x, center_y = 0, 0

        # find the error and update movement accordingly
        error = center_x - (cv_image.shape[1] / 2)

        if(self.current_time.secs < self.initial_time_counter+5):
            error = error - 50
        else:
            error = error + 30

        self.move.linear.x = 0.3
        self.move.angular.z = -0.02 * error - 0.002


    def follow_hill_line(self, cv_image):
        hsv_filter_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        hsv_lo = np.array([0,0,0])
        hsv_hi = np.array([100,55,255])
        mask = cv2.inRange(hsv_filter_image, hsv_lo, hsv_hi)
        image = mask[600:, 600:]
        red_image = cv_image[:,:,1]

        blurred_red_image = cv2.GaussianBlur(red_image, (5, 5), cv2.BORDER_DEFAULT)
        ret, binary_red_image = cv2.threshold(blurred_red_image, 50, 255, cv2.THRESH_BINARY)
        binary_red_image = binary_red_image[600:, 600:]
        image = cv2.bitwise_and(binary_red_image,image)

        rows,cols = image.shape
        pts1 = np.float32([[598,215],[350,20],[10,200]]) #points on the original image
        pts2 = np.float32([[550,215],[500,20],[10,200]]) #points on the desired image (previous points get mapped/stretched to this) (need at least 3)

        M = cv2.getAffineTransform(pts1,pts2)

        image = cv2.warpAffine(image,M,(cols,rows))

        # find the center of the track
        moments = cv2.moments(image)

        # find the center of the binary_image
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"]) + 700
        else:
            # set values to zero
            center_x, center_y = 500, 0

        # find the error and update movement accordingly
        image = cv2.circle(image,(center_x, 50), 30, (255,0,0), -1)
        
        error = center_x - 500
        self.move.linear.x = 0.25
        self.move.angular.z =  -0.02 * error

        if(self.current_time.secs <= self.start_crosswalk_time+3):
            print("go straight extra little bit")
            self.move.linear.x = 0.15
            self.move.angular.z = 0

        if(self.check_end_grass(hsv_filter_image) == 1):
            self.start_crosswalk_time = self.current_time.secs
            self.state = 3
    
    def check_end_grass(self, hsv_filter_image):
        hsv_filter_image = hsv_filter_image[600:, 500:700]

        hsv_lo = np.array([0,0,0])
        hsv_hi = np.array([100,55,255])
        mask = cv2.inRange(hsv_filter_image, hsv_lo, hsv_hi)

        if(np.mean(mask) > 100 and self.current_time.secs >= self.start_crosswalk_time+15):
            self.crosswalk_counter = 0
            return 1
        else:
            return 0

    
    def follow_inner_loop(self, cv_image):

        road_isolated = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        hsv_lo = np.array([0,0,0])
        hsv_hi = np.array([0,0,100])
        binary_image = cv2.inRange(road_isolated, hsv_lo, hsv_hi)

        # crop the image to focus on the bottom part only
        image = binary_image[500:]
        moments = cv2.moments(image)

        # find the center of the binary_image
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"]) + 700
        else:
            # set values to zero
            center_x, center_y = 0, 0


        # find the error and update movement accordingly
        error = center_x - (cv_image.shape[1] / 2)
        if(self.current_time.secs <= self.start_crosswalk_time + 8):
            error = error - 40
            print("time")
        else:
            error = error + 50
        self.move.linear.x = 0.35
        self.move.angular.z = -0.018 * error

        road_isolated2 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        hsv_lo = np.array([0,0,0])
        hsv_hi = np.array([0,0,19])
        hsv_filtered_image = cv2.inRange(road_isolated2, hsv_lo, hsv_hi)

        if (np.mean(hsv_filtered_image) > 1.3 or np.mean(hsv_filtered_image[:, 600:]) > 1.45):
            print("car!")
            self.move.linear.x = 0
            self.move.angular.z = 0



    def publish_move(self):
        self.pub_move.publish(self.move)
        if (self.done == 0):
            self.timer.data = "Team12,multi21,0,FA12"
            self.pub_timer.publish(self.timer)
            self.move.linear.x = 0
            self.move.angular.z = 0
            print("ayo")
            self.done = 1
            self.inital_time = self.current_time.secs
            self.initial_time_counter = self.inital_time
        self.pub_move.publish(self.move)
        
        if self.current_time.secs > self.inital_time+60 and self.done == 1: #change this back to the desired time
            self.timer.data = "Team12,multi21,-1,FA12"
            self.pub_timer.publish(self.timer)
            self.done = 2

        if (self.done == 2):
            self.move.linear.x = 0
            self.move.angular.z = 0
            self.pub_move.publish(self.move)

    def run(self):
        """
        Continuously publish the Twist message to 
        move the robot as long as the ROS node is 
        not shut down
        """
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('line_follower')
    line_follower = LineFollower()
    rospy.spin()