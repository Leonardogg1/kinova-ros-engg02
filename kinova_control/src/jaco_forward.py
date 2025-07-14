#!/usr/bin/env python3
import roslib; roslib.load_manifest('kinova_demo')
import rospy
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import trajectory_msgs.msg
from std_srvs.srv import Empty
import time
import sys
import actionlib
import kinova_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg


import math

class JacoSimulator:
    def __init__(self):
        rospy.init_node('jaco_simulacao')
        
        time.sleep(3)  
        # Unpause the physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        resp = unpause_gazebo()
        
        self.prefix = 'j2n6s300'
        self.nbJoints = 6	
        self.nbfingers = 3	
        
        # Subscriber da pose
        self.ee_pose = None
    def jaco_subscriber(self):
        # wait to get current position
        self.topic_address = '/' + self.prefix + '/joint_states'
        self.pose_subscriber = rospy.Subscriber(self.topic_address, sensor_msgs.msg.JointState, self.pose_callback, queue_size=1)
        try:
          rospy.wait_for_message(self.topic_address, sensor_msgs.msg.JointState, timeout=5.0)
          rospy.loginfo("Successfully connected to topic: %s", self.topic_address)
        except rospy.ROSException:
          rospy.logerr("Timeout waiting for topic: %s", self.topic_address)
          rospy.logwarn("Verify: 1) Robot is connected 2) Correct topic name 3) Messages are being published")
    	
    def pose_callback(self, msg):
        """Guarda a ultima pose do efetuador"""
        self.ee_pose = msg            
    
    def get_inputs(self):
        self.pos_juntas = input('Informe 6 posicoes para as juntas (de espaco entre as posicoes):')
        self.pos_dedos = input('Informe 3 posicoes para os dedos entre 1 (fecahdo) e 0 (aberto) (de espaco entre as posicoes):')    
        self.list_juntas = list(map(float, self.pos_juntas.split()))
        self.list_dedos = list(map(float, self.pos_dedos.split()))
        
        return [self.list_juntas, self.list_dedos]
        
    def move_jaco(self):
        juntas, dedos = self.get_inputs()
        self.jaco_subscriber()
        self.moveJoint(juntas)
        self.moveFingers(dedos)
        self.print_pose()
    
    def moveJoint(self, jointcmds):
        topic_name = '/' + self.prefix + '/effort_joint_trajectory_controller/command'
        pub = rospy.Publisher(topic_name, JointTrajectory, queue_size=1)
        jointCmd = JointTrajectory()  
        point = JointTrajectoryPoint()
        jointCmd.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.0);  
        point.time_from_start = rospy.Duration.from_sec(5.0)
        for i in range(0, self.nbJoints):
          jointCmd.joint_names.append(self.prefix +'_joint_'+str(i+1))
          point.positions.append(jointcmds[i])
          point.velocities.append(0)
          point.accelerations.append(0)
          point.effort.append(0) 
        jointCmd.points.append(point)
        rate = rospy.Rate(100)
        count = 0
        while (count < 50):
          pub.publish(jointCmd)
          count = count + 1
          rate.sleep()     

    def moveFingers(self, jointcmds):
        topic_name = '/' + self.prefix + '/effort_finger_trajectory_controller/command'
        pub = rospy.Publisher(topic_name, JointTrajectory, queue_size=1)  
        jointCmd = JointTrajectory()  
        point = JointTrajectoryPoint()
        jointCmd.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.0);  
        point.time_from_start = rospy.Duration.from_sec(5.0)
        for i in range(0, self.nbfingers):
          jointCmd.joint_names.append(self.prefix +'_joint_finger_'+str(i+1))
          point.positions.append(jointcmds[i])
          point.velocities.append(0)
          point.accelerations.append(0)
          point.effort.append(0) 
        jointCmd.points.append(point)
        rate = rospy.Rate(100)
        count = 0
        while (count < 500):
          pub.publish(jointCmd)
          count = count + 1
          rate.sleep()
     
    def print_pose(self):
         if self.ee_pose:
           rospy.loginfo(
            "\n=== Joint State ===\n"
            "Joint Names: %s\n"
            "Positions (rad): %s\n"
            #"Velocities (rad/s): %s\n"
            #"Efforts (Nm): %s\n"
            "==================" % (
                self.ee_pose.name,
                ["%.4f" % p for p in self.ee_pose.position]))#,
                #["%.4f" % v for p in self.ee_pose.velocity],
                #["%.4f" % e for p in self.ee_pose.effort]
            #)
           #)
           rospy.logdebug(
            "Positions (deg): %s" % 
            ["%.2f" % math.degrees(p) for p in self.ee_pose.position]
           )
         else:
           rospy.logwarn("Sem Pose")     


if __name__ == "__main__":	
    jaco = JacoSimulator()
    try:
      res = jaco.move_jaco()
      rospy.spin()
      #while not rospy.is_shutdown():
        #res = jaco.move_jaco()
    except rospy.ROSInterruptException:
    	print("program interrupted before completion")
    #rospy.sleep(1)
