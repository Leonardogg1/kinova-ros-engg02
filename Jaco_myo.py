#!/usr/bin/env python3
import rospy
import sys
import os
import math
import time
import moveit_commander
from geometry_msgs.msg import Pose
import threading

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
        self.arm.set_planning_time(5.0)  # Reduzido para resposta mais rápida
        self.arm.set_goal_position_tolerance(0.05)  # 5cm
        self.arm.set_goal_orientation_tolerance(0.3)  # ~17 graus
        
        # Estado inicial
        self.current_position = None
        rospy.loginfo("Controlador do Jaco Iniciado")

    def QuaternionNorm(self, Qx, Qy, Qz, Qw):
        qnorm = math.sqrt(Qx**2 + Qy**2 + Qz**2 + Qw**2)
        if qnorm < 1e-6:
            return Qx, Qy, Qz, Qw
        return Qx/qnorm, Qy/qnorm, Qz/qnorm, Qw/qnorm

    def Quaternion2Euler(self, Qx, Qy, Qz, Qw):
        qx, qy, qz, qw = self.QuaternionNorm(Qx, Qy, Qz, Qw)
        
        tx = math.atan2((2 * qw * qx - 2 * qy * qz), (qw**2 - qx**2 - qy**2 + qz**2))
        ty = math.asin(2 * qw * qy + 2 * qx * qz)
        tz = math.atan2((2 * qw * qz - 2 * qx * qy), (qw**2 + qx**2 - qy**2 - qz**2))
        
        return [tx, ty, tz]
    
    def move_para_pose(self, quat):
        """Atualiza a orientação do efetuador"""
        self.conti += 1
        if self.conti < self.cont_imu:
            return
            
        self.conti = 0  # Reset do contador
        
        try:
            # Primeira execução: captura posição atual
            if self.current_position is None:
                current_pose = self.arm.get_current_pose().pose
                self.current_position = current_pose.position
                rospy.loginfo("Posição inicial capturada")
            
            # Cria pose alvo
            target_pose = Pose()
            target_pose.position = self.current_position
            
            # Usa quaternion diretamente (sem converter para Euler)
            target_pose.orientation.x = quat[0]
            target_pose.orientation.y = quat[1]
            target_pose.orientation.z = quat[2]
            target_pose.orientation.w = quat[3]
            
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
            jaco.move_para_pose([x, y, z, w])
        
        def pose_handler(pose):
            jaco.acao_gripper(pose)
        
        myo = MyoRaw()
        myo.add_imu_handler(imu_handler)
        myo.add_pose_handler(pose_handler)
        myo.connect()
        
        rospy.loginfo("Conectado ao Myo. Controle ativo.")
        
        # Mantém o programa rodando
        rate = rospy.Rate(100)  # 100Hz
        while not rospy.is_shutdown():
            myo.run(0.01)  # Processa pacotes do Myo
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