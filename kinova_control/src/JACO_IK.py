#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import trajectory_msgs.msg
from std_srvs.srv import Empty
import time
import sys
import actionlib
import kinova_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import math
from tf.transformations import quaternion_from_euler

class JacoInverseKinematics:
    def __init__(self):
        rospy.init_node('jaco_inverse_kinematics')
        
        time.sleep(3)  
        # Unpause the physics (se estiver no Gazebo)
        try:
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=5.0)
            unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            resp = unpause_gazebo()
        except rospy.ROSException:
            rospy.logwarn("Gazebo physics unpause service not available - running without simulation?")
        
        self.prefix = 'j2n6s300'
        self.nbJoints = 6    
        self.nbfingers = 3    
        
        # Subscriber para a pose atual
        self.current_pose = None
        self.current_joint_state = None
        
        # Configuração do cliente de ação para cinemática inversa
        self.ik_action_address = '/' + self.prefix + '/arm_controller/query_ik'
        self.ik_client = actionlib.SimpleActionClient(self.ik_action_address, 
                                                     kinova_msgs.msg.ArmJointAnglesAction)
        
        # Configuração do publisher para comandos cartesianos
        self.cartesian_cmd_pub = rospy.Publisher('/' + self.prefix + '/arm_controller/cartesian_cmd', 
                                               kinova_msgs.msg.PoseVelocity, queue_size=1)
        
        # Configura subscribers
        self.setup_subscribers()
        
    def setup_subscribers(self):
        # Subscriber para a pose atual do efetuador
        self.pose_subscriber = rospy.Subscriber('/' + self.prefix + '/arm_controller/feedback', 
                                              kinova_msgs.msg.ArmPoseFeedback, 
                                              self.pose_feedback_callback, 
                                              queue_size=1)
        
        # Subscriber para o estado das juntas
        self.joint_state_subscriber = rospy.Subscriber('/' + self.prefix + '/joint_states', 
                                                     sensor_msgs.msg.JointState, 
                                                     self.joint_state_callback, 
                                                     queue_size=1)
        
        # Espera pelos primeiros dados
        rospy.wait_for_message('/' + self.prefix + '/arm_controller/feedback', 
                             kinova_msgs.msg.ArmPoseFeedback, timeout=5.0)
        rospy.wait_for_message('/' + self.prefix + '/joint_states', 
                             sensor_msgs.msg.JointState, timeout=5.0)
    
    def pose_feedback_callback(self, msg):
        """Armazena a pose atual do efetuador"""
        self.current_pose = msg.pose
    
    def joint_state_callback(self, msg):
        """Armazena o estado atual das juntas"""
        self.current_joint_state = msg
    
    def get_user_input(self):
        """Obtém a pose desejada do usuário"""
        print("\n=== Entrada de Posição Desejada ===")
        print("Informe as coordenadas (x, y, z) em metros e orientação (roll, pitch, yaw) em graus.")
        print("Exemplo: 0.4 -0.2 0.5 0 45 0")
        
        try:
            coords = input("Digite x y z roll pitch yaw: ").split()
            if len(coords) != 6:
                raise ValueError("Número incorreto de valores")
            
            x, y, z = map(float, coords[:3])
            roll, pitch, yaw = map(math.radians, map(float, coords[3:]))
            
            # Criar mensagem de pose
            target_pose = Pose()
            target_pose.position = Point(x, y, z)
            
            # Converter ângulos Euler para quaternion
            q = quaternion_from_euler(roll, pitch, yaw)
            target_pose.orientation = Quaternion(*q)
            
            return target_pose
            
        except ValueError as e:
            rospy.logerr(f"Entrada inválida: {e}")
            return None
    
    def solve_ik(self, target_pose):
        """Resolve a cinemática inversa para a pose desejada"""
        # Espera pelo servidor de IK
        if not self.ik_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Servidor de cinemática inversa não disponível!")
            return None
        
        # Cria o objetivo para IK
        goal = kinova_msgs.msg.ArmJointAnglesGoal()
        goal.pose = target_pose
        
        # Envia o objetivo
        self.ik_client.send_goal(goal)
        
        # Espera pelo resultado
        if self.ik_client.wait_for_result(rospy.Duration(10.0)):
            result = self.ik_client.get_result()
            return result.angles
        else:
            rospy.logerr("Tempo esgotado ao resolver IK")
            return None
    
    def move_to_pose(self, target_pose):
        """Move o braço para a pose desejada usando controle cartesiano"""
        # Primeiro tenta resolver IK para verificar se é alcançável
        joint_angles = self.solve_ik(target_pose)
        if joint_angles is None:
            rospy.logerr("Pose desejada não é alcançável!")
            return False
        
        # Se IK foi resolvido, envia comando cartesiano
        cmd = kinova_msgs.msg.PoseVelocity()
        cmd.pose = target_pose
        cmd.twist_linear_x = 0.01  # Velocidades pequenas para movimento suave
        cmd.twist_linear_y = 0.01
        cmd.twist_linear_z = 0.01
        cmd.twist_angular_x = 0.01
        cmd.twist_angular_y = 0.01
        cmd.twist_angular_z = 0.01
        
        rate = rospy.Rate(10)  # 10 Hz
        timeout = time.time() + 10.0  # Timeout de 10 segundos
        
        while not rospy.is_shutdown() and time.time() < timeout:
            # Publica o comando continuamente
            self.cartesian_cmd_pub.publish(cmd)
            
            # Verifica se alcançou a pose (simplificado - na prática precisaria de verificação mais robusta)
            current_pos = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ])
            target_pos = np.array([
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z
            ])
            
            if np.linalg.norm(current_pos - target_pos) < 0.01:  # Tolerância de 1 cm
                rospy.loginfo("Pose alcançada!")
                return True
            
            rate.sleep()
        
        rospy.logwarn("Timeout ao tentar alcançar a pose")
        return False
    
    def print_current_state(self):
        """Exibe o estado atual do braço"""
        if self.current_pose and self.current_joint_state:
            # Posição cartesiana
            pos = self.current_pose.position
            orient = self.current_pose.orientation
            
            # Ângulos das juntas (em graus)
            joint_positions = [math.degrees(p) for p in self.current_joint_state.position[:6]]
            
            rospy.loginfo(
                "\n=== Estado Atual ===\n"
                "Posição (x, y, z): %.3f, %.3f, %.3f\n"
                "Orientação (x, y, z, w): %.3f, %.3f, %.3f, %.3f\n"
                "Ângulos das Juntas (graus): %s\n"
                "====================" % (
                    pos.x, pos.y, pos.z,
                    orient.x, orient.y, orient.z, orient.w,
                    ["%.1f" % p for p in joint_positions]
                )
            )
        else:
            rospy.logwarn("Dados atuais não disponíveis")
    
    def run(self):
        """Loop principal"""
        try:
            while not rospy.is_shutdown():
                # Obtém a pose desejada do usuário
                target_pose = self.get_user_input()
                if target_pose is None:
                    continue
                
                # Move para a pose desejada
                success = self.move_to_pose(target_pose)
                
                # Exibe o estado atual
                self.print_current_state()
                
                # Pergunta se deseja continuar
                cont = input("Deseja enviar outra posição? (s/n): ").lower()
                if cont != 's':
                    break
                    
        except rospy.ROSInterruptException:
            rospy.loginfo("Interrupção do ROS recebida")
        except KeyboardInterrupt:
            rospy.loginfo("Execução interrompida pelo usuário")

if __name__ == "__main__":
    jaco_ik = JacoInverseKinematics()
    jaco_ik.run()