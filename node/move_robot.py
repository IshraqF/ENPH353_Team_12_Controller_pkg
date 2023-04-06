#! /usr/bin/env python3

# import rospy
# from geometry_msgs.msg import Twist

# rospy.init_node('topic_publisher')
# pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
# rate = rospy.Rate(2)
# move = Twist()
# move.linear.x = 0.5
# move.angular.z = 0.5

# while not rospy.is_shutdown():
#     pub.publish(move)
#     rate.sleep()

import rospy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock

class LineFollower:
    def __init__(self):
        """
        Initialize the LineFollower class
        """
        self.bridge = CvBridge()
        self.move = Twist()
        self.timer = String()
        self.done = 0
        self.inital_time = 0
        self.current_time = Clock()
        self.image_sub = rospy.Subscriber(
            '/R1/pi_camera/image_raw', Image, self.callback)
        self.pub_move = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.pub_timer = rospy.Publisher('/license_plate', String, queue_size=1)
        self.sub_clock = rospy.Subscriber('/clock', Clock, self.clock_callback)
    
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

        self.follow_line(cv_image)
        self.publish_move()

    def follow_line(self, cv_image):
        """
        Algorithm to follow a line
        """

        # cv2.imshow("image", cv_image)
        # cv2.waitKey(1)

        grayscaled_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(
            grayscaled_image, (5, 5), cv2.BORDER_DEFAULT)
        ret, binary_image = cv2.threshold(
            blurred_image, 100, 255, cv2.THRESH_BINARY)

        # invert the binary image to convert the track from black to white
        binary_image = 255-binary_image

        # crop the image to focus on the bottom part only
        image = binary_image[700:]

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
        self.move.linear.x = 0.1
        self.move.angular.z = -0.01 * error

    def publish_move(self):
        """
        Publish the Twist message to move the robot
        """
        self.pub_move.publish(self.move)
        if (self.done == 0):
            # publish '0' once when the program starts
            self.timer.data = 'Team12,multi21,0,AB12'
            self.pub_timer.publish(self.timer)
            self.done = 1
            self.inital_time = self.current_time.secs
        
        if self.current_time.secs > self.inital_time+60 and self.done == 1:
            self.timer.data = 'Team12,multi21,-1,AB12'
            self.pub_timer.publish(self.timer)
            self.done = 2

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