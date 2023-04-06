#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
from std_msgs.msg import String

# Minimum number of matches required to perform homography
MIN_MATCH_COUNT = 10

# @class My_App
#  @brief Main class for the object tracking application using homography
class plate_detect():

    def __init__(self):
        self.images = {"P1.png": None, "P2.png": None, "P3.png": None, "P4.png": None,
                       "P5.png": None, "P6.png": None, "P7.png": None, "P8.png": None}
        
        for img in self.images.keys():
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            kp, desc = self.template_kp_desc(image)
            self.images[img] = (kp, desc)
        
        self.plate = String()
        self.bridge = CvBridge()
        self.image_sub = self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.callback)
        self.pub_plate = rospy.Publisher('/license_plate', String, queue_size=1)
    
    def template_kp_desc(self, template):
        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(template, None)
        return kp, desc

    def callback(self, data):
        """
        Callback function for image data
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)

        self.identify_plate(cv_image)
        # self.publish_license()

    def identify_plate(self, cv_image):
        # Create a SIFT object
        sift = cv2.SIFT_create()

        # Convert the current frame to grayscale and compute SIFT keypoints and descriptors for grayscale image and template image
        grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        kp_grayscale, desc_grayscale = sift.detectAndCompute(grayscale, None)

        # Create a FLANN matcher
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        for img, (kp, desc) in self.images.items():

            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

            # Match the descriptors using FLANN
            matches = flann.knnMatch(desc, desc_grayscale, k=2)

            # Keep only good matches based on Lowe's ratio test
            good_points = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_points.append(m)

            # If enough good matches are found, perform homography
            MIN_MATCH_COUNT = 10
            if len(good_points) > MIN_MATCH_COUNT:
                query_pts = np.float32(
                    [kp[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32(
                    [kp_grayscale[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(
                    query_pts, train_pts, cv2.RANSAC, 5.0)
                # matchesMask = mask.ravel().tolist()

                # Draw the object bounding box on the current frame
                h, w = image.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]
                                ).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                # frame = cv2.polylines(
                #     cv_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                
                # Get the x, y coordinates of the top-left corner and bottom-right corner of the bounding box
                x1, y1 = np.int32(dst[0][0])
                x2, y2 = np.int32(dst[2][0])
                
                if x2-x1>=98 and (y2+30)-y1>=72:
                    # Crop the frame to the bounding box
                    cropped_frame = cv_image[y1:y2+50, x1:x2]

                    # # Draw the matches on the current frame
                    # draw_matches = dict(matchColor=(0, 255, 0),  # draw matches in green color
                    #                     singlePointColor=None,
                    #                     matchesMask=matchesMask,  # draw only inliers
                    #                     flags=2)
                    # frame = cv2.drawMatches(
                    #     image, kp_image, frame, kp_grayscale, good_points, None, **draw_matches)
                    
                    # display in the Qt window
                    cv2.imshow("Cropped Image", cropped_frame)
                    cv2.waitKey(1)
                                        
# Main program execution.
if __name__ == "__main__":
    rospy.init_node('identify_plate')
    plate = plate_detect()
    rospy.spin()
    