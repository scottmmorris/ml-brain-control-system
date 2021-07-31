#!/usr/bin/env python

import requests
import rospy
from geometry_msgs.msg import Twist

BASE = "http://192.168.20.6:56789/"

def executeAction(action, msg):
    if (action == 0):
        msg.angular.z = 0
        msg.linear.x = 0
    elif (action == 1):
        msg.angular.z = 1
        msg.linear.x = 0
    elif (action == 2):
        msg.angular.z = -1
        msg.linear.x = 0
    elif (action == 3):
        msg.angular.z = 0
        msg.linear.x = -1
    elif (action == 4):
        msg.linear.x = 0.05
        msg.angular.z = 0

    return msg



rospy.init_node("GoForward")
cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
move_cmd = Twist()
rate = rospy.Rate(10)
move_cmd.linear.x = 0
move_cmd.angular.z = 0
print("To stop press CTRL + C")

while not rospy.is_shutdown():
    response = requests.get(BASE + "1")
    print("Publsihing")
    action = response.json()
    print(action)
    move_cmd = executeAction(action, move_cmd)
    cmd_vel.publish(move_cmd)
    rate.sleep()

rospy.sleep(10)
print("Shutting down go_forward package")