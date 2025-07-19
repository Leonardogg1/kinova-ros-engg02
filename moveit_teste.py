#!/usr/bin/env python3
import rospy
import sys
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from math import pi
from tf.transformations import quaternion_from_euler

class JacoController:
    def __init__(self, robot_type="j2n6s300"):
        # Inicializando o moveit
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
        
        rospy.loginfo("Controlador do Jaco Inicializado")

    def move_para_pose(self, x, y, z, roll=0.0, pitch=pi/2, yaw=pi):
        """Move o efeutador para pose cartesiana especificada"""
        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        
        # Converte angulos para quaternion (convencao ZYZ)
        q = quaternion_from_euler(yaw, pitch, roll)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]
        
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
            
        return True

    def move_juntas(self, angulo_juntas):
        """Move para angulo de junta especificado (radianos)"""
        self.arm.set_joint_value_target(angulo_juntas)
        
        rospy.loginfo(f"Movendo para o angulo de juntas: {angulo_juntas}")
        
        plan = self.arm.plan()
        if not plan[0]:
            rospy.logerr("Planejamento Falhou!")
            return False
            
        result = self.arm.execute(plan[1], wait=True)
        if not result:
            rospy.logerr("Execucao Falhou!")
            return False
            
        return True

    def acao_gripper(self, posicao):
        """Controle do gripper (0.0=aberto, 1.0=fechado)"""
        self.gripper.set_named_target("Open" if posicao < 0.5 else "Close")
        self.gripper.go(wait=True)
        return True

if __name__ == '__main__':
    try:
        # Inicializando
        jaco = JacoController()
        
        # Examplo:
        # 1. Move as juntas para posicao base
        base_juntas = [0.0, 2.9, 1.3, 4.2, 1.4, 0.0]  # radianos
        jaco.move_juntas(base_juntas)
        
        # 2. Move para posicao cartesiana (Cinematica Inversa)
        # x=0.3m forward, y=0.1m left, z=0.5m up
        # roll=0, pitch=pi/2 (vertical), yaw=pi (facing forward)
        jaco.move_para_pose(0.3, 0.1, 0.5, 0.0, pi/2, pi)
        
        # 3. Fecha/Abre o gripper
        jaco.acao_gripper(0.0)
        
    except rospy.ROSInterruptException:
        pass
