#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import String

class detect_plate():
    def __init__(self):
        """
        Initialize the LineFollower class
        """
        self.bridge = CvBridge()
        # self.timer = String()
        # self.done = 0
        self.state = 0
        # self.inital_time = 0
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.callback)
        # self.pub_timer = rospy.Publisher('/license_plate', String, queue_size=1)
    
    def callback(self, data):
        """
        Callback function for image data
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)

        self.detect_plate(cv_image)
    
    def detect_plate(self, image):
        # cv2.imshow("Camera", image)
        # cv2.waitKey(1)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([0, 125, 62])
        upper_bound = np.array([0, 255, 222])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        filtered_img = cv2.bitwise_and(image, image, mask=mask)

        img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]

        if len(contours) >= 2:
            sortedContours = sorted(contours, key=cv2.contourArea, reverse=True)
        #   for i in sortedContours:
        #     print(cv2.contourArea(i))

        # print("\n")

        
            if cv2.contourArea(sortedContours[0]) > 30000:
                largest_contour = sortedContours[0]

                x, y, w, h = cv2.boundingRect(largest_contour)

                cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 3)

            # cv2.imshow("Camera", image)
            # cv2.waitKey(1)

                if self.state ==1 or self.state == 7 or self.state == 8:
                    crop = image[y:y+h, x:x+w+200]
                    cv2.imshow("Cropped", crop)
                    cv2.waitKey(1)
                    self.state += 1
                elif self.state < 8:
                    crop = image[y-5:y+h, 0:x+50]
                    cv2.imshow("Cropped", crop)
                    cv2.waitKey(1)
                    self.state += 1

            # cv2_imshow(filtered_img)
        print(self.state)
        cv2.imshow("Camera", image)
        cv2.waitKey(1)


# Main program execution.
if __name__ == "__main__":
    rospy.init_node('detect_plate')
    plate = detect_plate()
    rospy.spin()