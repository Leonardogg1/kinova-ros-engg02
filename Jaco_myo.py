#!/usr/bin/env python3
import rospy
import sys
import os
import math
import time
import moveit_commander
from geometry_msgs.msg import Pose, Point
import threading
import numpy as np
from sensor_msgs.msg import JointState

# Adiciona o caminho do myo-raw ao sistema
sys.path.append(os.path.join(os.path.dirname(__file__), 'myo-raw'))
from myo_raw import MyoRaw, Pose as MyoPose

class JacoController:
    def __init__(self, robot_type="j2n6s300"):
        # Inicializando o moveit
        self.contg = 0
        self.conti = 0
        self.cont_gest = 50  # Limite mais sensível
        self.cont_imu = 100  # Limite mais sensível
        
        # Corrige os warnings de configuração antes de inicializar
        self.fix_moveit_warnings()
        
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('jaco_controller')
        
        # Configurar tempo de espera mais longo para servidores
        rospy.sleep(3)
        
        # Setup do moveit
        try:
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            self.arm = moveit_commander.MoveGroupCommander("arm", wait_for_servers=60.0)
            self.gripper = moveit_commander.MoveGroupCommander("gripper", wait_for_servers=60.0)
        except Exception as e:
            rospy.logerr(f"Erro ao inicializar MoveIt: {str(e)}")
            rospy.signal_shutdown("Erro crítico na inicialização")
            return
        
        # Configuração
        self.arm.set_end_effector_link(f"{robot_type}_end_effector")
        self.arm.set_planning_time(10.0)  # Tempo maior para planejamento
        self.arm.set_goal_position_tolerance(0.05)  # 5cm
        self.arm.set_goal_orientation_tolerance(0.3)  # ~17 graus
        
        # Estado inicial
        self.initial_position = None
        self.initial_quat = None
        self.ref_angle = 0.0  # Ângulo de referência do cotovelo
        
        # Parâmetros do plano de movimento
        self.x_range = 0.5  # Alcance máximo em X (metros)
        self.z_range = 0.5  # Alcance máximo em Z (metros)
        self.sensitivity = 0.5  # Sensibilidade do movimento
        
        # Captura posição inicial
        self.capture_initial_position()
        rospy.loginfo("Controlador do Jaco Iniciado")

    def fix_moveit_warnings(self):
        """Corrige avisos de configuração do MoveIt"""
        # Remove parâmetro obsoleto se existir
        if rospy.has_param('/robot_description_kinematics/arm/kinematics_solver_attempts'):
            rospy.delete_param('/robot_description_kinematics/arm/kinematics_solver_attempts')
            rospy.loginfo("Parâmetro obsoleto 'kinematics_solver_attempts' removido")
        
        # Configuração recomendada para evitar warnings
        rospy.set_param('/move_group/trajectory_execution/execution_duration_monitoring', False)
        rospy.loginfo("Configurações de warnings do MoveIt ajustadas")

    def capture_initial_position(self):
        """Captura a posição inicial com tratamento de erro robusto"""
        # Esperar pelo estado das juntas antes de capturar a posição
        rospy.loginfo("Aguardando estado das juntas...")
        try:
            rospy.wait_for_message('/joint_states', JointState, timeout=20)
        except rospy.ROSException:
            rospy.logwarn("Tempo excedido aguardando joint_states. Continuando...")
        
        attempts = 0
        while attempts < 15 and not rospy.is_shutdown():
            try:
                # Usar get_current_pose() com verificação de timestamp
                current_pose = self.arm.get_current_pose().pose
                if current_pose is not None:
                    self.initial_position = current_pose.position
                    rospy.loginfo("Posição inicial capturada com sucesso")
                    rospy.loginfo(f"Posição inicial: X={self.initial_position.x:.2f}, "
                                  f"Y={self.initial_position.y:.2f}, Z={self.initial_position.z:.2f}")
                    return
            except Exception as e:
                rospy.logwarn(f"Falha ao capturar posição inicial (tentativa {attempts+1}/15): {str(e)}")
                rospy.sleep(1)  # Espera mais tempo entre tentativas
                attempts += 1
        
        rospy.logerr("Não foi possível capturar a posição inicial após 15 tentativas")
        rospy.signal_shutdown("Erro crítico na inicialização")

    def calculate_elbow_angle(self, quat):
        """Calcula o ângulo do cotovelo baseado na orientação do Myo"""
        # Extrai componentes do quaternion
        w, x, y, z = quat
        
        # Vetor de direção do antebraço (considerando Myo no antebraço)
        # Vetor padrão apontando para longe do cotovelo (direção da mão)
        forearm_vector = np.array([-1, 0, -1])
        
        # Matriz de rotação do quaternion
        rotation_matrix = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
        
        # Aplica rotação ao vetor
        rotated_vector = rotation_matrix.dot(forearm_vector)
        
        # Ângulo com a vertical (gravidade)
        vertical = np.array([0, 0, -1])  # Vetor vertical para baixo
        dot_product = np.dot(rotated_vector, vertical)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        return angle
    
    def move_in_plane(self, quat):
        """Move o braço em um plano fixo baseado no ângulo do cotovelo"""
        if self.initial_position is None:
            return
            
        self.conti += 1
        if self.conti < self.cont_imu:
            return
            
        self.conti = 0  # Reset do contador
        
        try:
            # Primeira execução: captura orientação inicial
            if self.initial_quat is None:
                self.initial_quat = quat
                self.ref_angle = self.calculate_elbow_angle(quat)
                rospy.loginfo(f"Ângulo de referência: {math.degrees(self.ref_angle):.1f}°")
                return
            
            # Calcula o ângulo atual do cotovelo
            current_angle = self.calculate_elbow_angle(quat)
            
            # Calcula diferença angular em relação à referência
            angle_diff = current_angle - self.ref_angle
            
            # Obtém yaw (rotação em z) da orientação atual
            # Usando uma abordagem simplificada para evitar problemas de singularidade
            w, x, y, z = quat
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Obtém yaw da orientação inicial
            w0, x0, y0, z0 = self.initial_quat
            siny_cosp0 = 2 * (w0 * z0 + x0 * y0)
            cosy_cosp0 = 1 - 2 * (y0 * y0 + z0 * z0)
            init_yaw = math.atan2(siny_cosp0, cosy_cosp0)
            
            # Calcula diferença de yaw
            yaw_diff = yaw - init_yaw
            
            # Calcula variações de posição
            # Z: baseado na diferença angular do cotovelo
            delta_z = math.sin(angle_diff) * self.sensitivity
            
            # X: baseado na rotação do braço (yaw)
            delta_x = yaw_diff * self.sensitivity
            
            # Limita as variações aos limites definidos
            delta_z = max(-self.z_range, min(self.z_range, delta_z))
            delta_x = max(-self.x_range, min(self.x_range, delta_x))
            
            # Cria pose alvo
            target_pose = Pose()
            target_pose.position.x = self.initial_position.x + delta_x
            target_pose.position.y = self.initial_position.y  # Mantém Y fixo
            target_pose.position.z = self.initial_position.z + delta_z
            
            # Mantém orientação fixa (apenas posição muda)
            target_pose.orientation.x = self.initial_quat[0]
            target_pose.orientation.y = self.initial_quat[1]
            target_pose.orientation.z = self.initial_quat[2]
            target_pose.orientation.w = self.initial_quat[3]
            
            # Executa movimento
            self.arm.set_pose_target(target_pose)
            success = self.arm.go(wait=True)
            
            if not success:
                rospy.logwarn("Movimento falhou!")
                
        except Exception as e:
            rospy.logerr(f"Erro no movimento: {str(e)}")

    def acao_gripper(self, pose):
        """Controle do gripper baseado em gestos do Myo"""
        self.contg += 1
        if self.contg < self.cont_gest:
            return
            
        self.contg = 0  # Reset do contador
        
        try:
            if pose == MyoPose.FIST:
                self.gripper.set_named_target("Close")
                self.gripper.go(wait=True)
                rospy.loginfo("Gripper FECHADO")
            elif pose == MyoPose.FINGERS_SPREAD:
                self.gripper.set_named_target("Open")
                self.gripper.go(wait=True)
                rospy.loginfo("Gripper ABERTO")
                
        except Exception as e:
            rospy.logerr(f"Erro no gripper: {str(e)}")

if __name__ == '__main__':   
    try:
        # Inicializando o controlador do Jaco
        jaco = JacoController()
        
        # Inicializando o Myo
        def imu_handler(quat, acc, gyro):
            # Converte quaternion do Myo (valores inteiros para float)
            w, x, y, z = [q/16384.0 for q in quat]
            jaco.move_in_plane([x, y, z, w])
        
        def pose_handler(pose):
            jaco.acao_gripper(pose)
        
        myo = MyoRaw()
        myo.add_imu_handler(imu_handler)
        myo.add_pose_handler(pose_handler)
        myo.connect()
        
        rospy.loginfo("Conectado ao Myo. Controle ativo.")
        
        # Mantém o programa rodando
        rate = rospy.Rate(50)  # 50Hz (mais baixo para reduzir carga)
        while not rospy.is_shutdown():
            myo.run(0.02)  # Processa pacotes do Myo
            rate.sleep()
            
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Erro fatal: {str(e)}")
    finally:
        if 'myo' in locals():
            myo.disconnect()
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("Programa encerrado.")