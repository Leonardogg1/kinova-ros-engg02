#!/usr/bin/env python3
import rospy
import sys
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
import math
from sensor_msgs.msg import Imu
from std_msgs.msg import UInt8

class JacoController:
    def __init__(self, robot_type="j2n6s300"):
        # Inicializando o moveit
        self.contg = 0
        self.conti = 0
        self.cont_gest = 500 #modificar limite
        self.cont_imu = 1000 #modificar limite
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('jaco_controller')
        
        
        # Setup do moveit
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()
        rospy.sleep(1)
        self.arm = moveit_commander.MoveGroupCommander("arm", wait_for_servers=30.0)
        self.gripper = moveit_commander.MoveGroupCommander("gripper", wait_for_servers=30.0)
        
        # Configuracao
        self.arm.set_end_effector_link(f"{robot_type}_end_effector")
        self.arm.set_planning_time(10.0)
        self.arm.set_goal_position_tolerance(0.01)  # 1cm
        self.arm.set_goal_orientation_tolerance(0.1)  # ~6 graus
        
        rospy.loginfo("Controlador do Jaco Iniciado")

    def QuaternionNorm(self, Qx, Qy, Qz, Qw):
        qx_temp,qy_temp,qz_temp,qw_temp = Qx, Qy, Qz, Qw
        qnorm = math.sqrt(qx_temp*qx_temp + qy_temp*qy_temp +    qz_temp*qz_temp + qw_temp*qw_temp)
        qx = qx_temp/qnorm
        qy = qy_temp/qnorm
        qz = qz_temp/qnorm
        qw = qw_temp/qnorm
        Q_normed = [qx, qy, qz, qw]
        return Q_normed

    def Quaternion2Euler(self, Qx, Qy, Qz, Qw):
        Q_normed = self.QuaternionNorm(Qx, Qy, Qz, Qw)
        qx_ = Q_normed[0]
        qy_ = Q_normed[1]
        qz_ = Q_normed[2]
        qw_ = Q_normed[3]

        tx = math.atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_))
        ty = math.asin(2 * qw_ * qy_ + 2 * qx_ * qz_)
        tz = math.atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_))
        EulerXYZ = [tx,ty,tz]
        return EulerXYZ
    
    def move_para_pose(self, quat):
        """Move o efeutador para pose cartesiana especificada"""
        self.conti+=1
        if self.conti >= self.cont_imu:
            target_pose = Pose()
            q0=quat.orientation.w
            q1=quat.orientation.x
            q2=quat.orientation.y
            q3=quat.orientation.z
        
            target_pose.orientation.w = q0
            target_pose.orientation.x = q1
            target_pose.orientation.y = q2
            target_pose.orientation.z = q3
        
            EulerXYZ = self.Quaternion2Euler(q1, q2, q3, q0)
        
            target_pose.position.x = EulerXYZ[0]
            target_pose.position.y = EulerXYZ[1]
            target_pose.position.z = EulerXYZ[2]
        
            self.arm.set_pose_target(target_pose)
        
            rospy.loginfo(f"Movendo para a posicao: [{x:.3f}, {y:.3f}, {z:.3f}]")
            rospy.loginfo(f"Com orientacao: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
        
            # Planeja e executa
            plan = self.arm.plan()
            if not plan[0]:
                rospy.logerr("Planejamento Falhou!")
                return False
            
            result = self.arm.execute(plan[1], wait=True)
            if not result:
                rospy.logerr("Execucao Falhou!")
                return False
            self.conti = 0            
            return True
        return True
            

    def acao_gripper(self, posicao):
        """Controle do gripper (0.0=aberto, 1.0=fechado)"""
        self.contg += 1
        if self.contg >= self.cont_gest:
            param = posicao.data # 1 FIST / 4 FINGERS_SPREAD
            self.gripper.set_named_target("Open" if param == 4 else "Close")
            self.gripper.go(wait=True)
            self.contg = 0
            return True
        return True
            

if __name__ == '__main__':   
    try:
        # Inicializando
        jaco = JacoController()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Programa Interrompido!!!")
        pass
