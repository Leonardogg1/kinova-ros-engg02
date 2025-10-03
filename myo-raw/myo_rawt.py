from __future__ import print_function

import enum
import re
import struct
from struct import pack, unpack
import sys
import threading
import time
from common import *

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import serial
from serial.tools.list_ports import comports

# Adicionar importações para o EKF
from scipy.linalg import inv, block_diag

class ExtendedKalmanFilter:
    def __init__(self):
        # Estado: [px, py, pz, vx, vy, vz, ax, ay, az] - 9 dimensões
        self.n = 9
        self.m = 6  # Medições: [ax, ay, az, gx, gy, gz]
        
        # Estado inicial
        self.x = np.zeros(self.n)
        
        # Matriz de covariância do estado
        self.P = np.eye(self.n) * 0.1
        
        # Matriz de covariância do processo (ruído do modelo)
        self.Q = np.eye(self.n) * 0.01
        
        # Matriz de covariância da medição
        self.R = np.eye(self.m) * 0.1
        
        # Matriz de transição de estado (será calculada a cada passo)
        self.F = np.eye(self.n)
        
        # Matriz de observação (será calculada a cada passo)
        self.H = np.zeros((self.m, self.n))
        
        # Timestamp anterior
        self.last_time = time.time()
        
        # Gravidade
        self.gravity = np.array([0, 0, 0.969])
        
    def predict(self, dt):
        # Atualizar matriz de transição de estado
        # Posição: p = p + v*dt + 0.5*a*dt²
        # Velocidade: v = v + a*dt
        # Aceleração: a = a (modelo de velocidade constante)
        
        # Preencher matriz F
        self.F = np.eye(self.n)
        
        # Transição posição-velocidade
        for i in range(3):
            self.F[i, i+3] = dt
            self.F[i, i+6] = 0.5 * dt**2
            self.F[i+3, i+6] = dt
        
        # Predição do estado
        self.x = self.F @ self.x
        
        # Predição da covariância
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, acceleration, gyro, quaternion):
        # Converter medições para arrays numpy
        z = np.concatenate([acceleration, gyro])
        
        # Calcular matriz de observação H (jacobiana da função de observação)
        self._compute_observation_jacobian(quaternion)
        
        # Inovação (resíduo)
        y = z - self._observation_function(quaternion)
        
        # Covariância da inovação
        S = self.H @ self.P @ self.H.T + self.R
        
        # Ganho de Kalman
        K = self.P @ self.H.T @ inv(S)
        
        # Atualização do estado
        self.x = self.x + K @ y
        
        # Atualização da covariância (forma mais estável)
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x.copy()
    
    def _observation_function(self, quaternion):
        # Função de observação: transforma estado em medições esperadas
        h = np.zeros(self.m)
        
        # As acelerações esperadas são as acelerações do estado + gravidade no frame global
        # Rotacionar a gravidade para o frame do sensor usando o quaternion
        q = self._normalize_quaternion(quaternion)
        gravity_body = self._rotate_vector(self.gravity, self._quaternion_inverse(q))
        
        # Acelerações esperadas (primeiras 3 medições)
        h[0:3] = self.x[6:9] + gravity_body
        
        # Giroscópio esperado (últimas 3 medições) - assumindo pequenas variações
        # Em um modelo mais complexo, isso viria da derivada do quaternion
        h[3:6] = np.zeros(3)  # Simplificação
        
        return h
    
    def _compute_observation_jacobian(self, quaternion):
        # Jacobiana da função de observação em relação ao estado
        self.H = np.zeros((self.m, self.n))
        
        # Para as acelerações: dh/dacceleration = I
        self.H[0:3, 6:9] = np.eye(3)
        
        # Para o giroscópio: derivadas são zero na simplificação atual
        # Em uma implementação mais completa, aqui viriam as derivadas da orientação
        
        return self.H
    
    def _normalize_quaternion(self, q):
        q = np.array([x / 16384.0 for x in q])
        norm = np.linalg.norm(q)
        return q / norm if norm > 0 else np.array([1.0, 0, 0, 0])
    
    def _quaternion_inverse(self, q):
        # Inverso do quaternion (conjugado)
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _rotate_vector(self, v, q):
        # Rotacionar vetor v pelo quaternion q
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        uv *= (2.0 * q[0])
        uuv *= 2.0
        return v + uv + uuv
    
    def get_state(self):
        return {
            'position': self.x[0:3].copy(),
            'velocity': self.x[3:6].copy(),
            'acceleration': self.x[6:9].copy()
        }
    
    def reset(self):
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 0.1
        self.last_time = time.time()

class Trajectory3D:
    def __init__(self, max_history=100):
        # Usar Filtro de Kalman Estendido
        self.ekf = ExtendedKalmanFilter()
        
        # Configuração
        self.max_history = max_history
        self.history = []
        self.history_direct = []  # Histórico da integração direta
        self.last_time = time.time()
        
        # Inicializar atributos de estado
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        
        # Estado para integração direta (para comparação)
        self.position_direct = np.zeros(3)
        self.velocity_direct = np.zeros(3)
        self.acceleration_direct = np.zeros(3)
        self.prev_acceleration_direct = np.zeros(3)
        
        # Parâmetros para detecção de repouso
        self.stationary_threshold = 1.55
        self.stationary_count = 0
        self.accel_bias = np.zeros(3)
        self.bias_learning_rate = 0.05
        
        # Parâmetros para integração direta
        self.velocity_damping = 0.95  # Amortecimento para reduzir drift
        self.accel_filter = np.zeros(3)
        self.alpha = 0.8  # Fator de suavização

    def update(self, quat, raw_acc, gyro):
        # Converter inputs
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        dt = max(min(dt, 0.1), 0.001)  # Limitar dt entre 1ms e 100ms
        
        # Calibrar aceleração
        calibrated_acc = np.array(raw_acc, dtype=np.float32) / 2048.0
        
        # Detecção de repouso para calibração
        if np.linalg.norm(calibrated_acc - [0, 0, 1]) < self.stationary_threshold:
            self.stationary_count += 1
            if self.stationary_count > 10:
                self.accel_bias = 0.98 * self.accel_bias + 0.02 * (calibrated_acc - [0, 0, 1])
        else:
            self.stationary_count = 0
        
        calibrated_acc = calibrated_acc - self.accel_bias
        
        # Converter giroscópio
        calibrated_gyro = np.array(gyro, dtype=np.float32) / 16.0  # Escala aproximada
        
        # Executar Filtro de Kalman Estendido
        # Predição
        self.ekf.predict(dt)
        
        # Atualização com medições
        self.ekf.update(calibrated_acc, calibrated_gyro, quat)
        
        # Obter estado estimado
        state = self.ekf.get_state()
        self.position = state['position']
        self.velocity = state['velocity'] 
        self.acceleration = state['acceleration']
        
        # Calcular também a integração direta para comparação
        self._update_direct_integration(quat, calibrated_acc, dt)
        
        # Gerir histórico
        self.history.append(self.position.copy())
        self.history_direct.append(self.position_direct.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_direct.pop(0)

        return self.position.copy()
    
    def _update_direct_integration(self, quat, calibrated_acc, dt):
        # Integração direta para comparação (sem Kalman)
        q = self._normalize_quaternion(quat)
        rotated_acc = self._rotate_vector(calibrated_acc, q)
        
        # Remover gravidade (assumindo que sabemos a orientação)
        gravity = np.array([0, 0, 0.969])
        linear_acc = rotated_acc - gravity
        
        # Filtro de suavização
        self.acceleration_direct = self.alpha * linear_acc + (1 - self.alpha) * self.accel_filter
        self.accel_filter = self.acceleration_direct.copy()
        
        # Integração de velocidade
        avg_acc = 0.5 * (self.prev_acceleration_direct + self.acceleration_direct)
        self.velocity_direct += avg_acc * dt
        
        # Amortecimento para reduzir drift
        self.velocity_direct *= self.velocity_damping
        
        # Integração de posição
        prev_velocity = self.velocity_direct - avg_acc * dt
        avg_vel = 0.5 * (prev_velocity + self.velocity_direct)
        self.position_direct += avg_vel * dt
        
        self.prev_acceleration_direct = self.acceleration_direct.copy()

    def _normalize_quaternion(self, q):
        q = np.array([x / 16384.0 for x in q])
        norm = np.linalg.norm(q)
        return q / norm if norm > 0 else np.array([1.0, 0, 0, 0])
    
    def _rotate_vector(self, v, q):
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        uv *= (2.0 * q[0])
        uuv *= 2.0
        return v + uv + uuv

    def get_current_state(self):
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'position_direct': self.position_direct.copy()
        }

    def get_history(self):
        return np.array(self.history.copy())
    
    def get_direct_history(self):
        return np.array(self.history_direct.copy())

    def reset(self):
        self.ekf.reset()
        self.history = []
        self.history_direct = []
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.position_direct = np.zeros(3)
        self.velocity_direct = np.zeros(3)
        self.acceleration_direct = np.zeros(3)
        self.prev_acceleration_direct = np.zeros(3)
        self.last_time = time.time()
        self.accel_bias = np.zeros(3)


def multichr(ords):
    if sys.version_info[0] >= 3:
        return bytes(ords)
    else:
        return ''.join(map(chr, ords))

def multiord(b):
    if sys.version_info[0] >= 3:
        return list(b)
    else:
        return map(ord, b)

class Arm(enum.Enum):
    UNKNOWN = 0
    RIGHT = 1
    LEFT = 2

class XDirection(enum.Enum):
    UNKNOWN = 0
    X_TOWARD_WRIST = 1
    X_TOWARD_ELBOW = 2

class Pose(enum.Enum):
    REST = 0
    FIST = 1
    WAVE_IN = 2
    WAVE_OUT = 3
    FINGERS_SPREAD = 4
    THUMB_TO_PINKY = 5
    UNKNOWN = 255


class Packet(object):
    def __init__(self, ords):
        self.typ = ords[0]
        self.cls = ords[2]
        self.cmd = ords[3]
        self.payload = multichr(ords[4:])

    def __repr__(self):
        return 'Packet(%02X, %02X, %02X, [%s])' % \
            (self.typ, self.cls, self.cmd,
             ' '.join('%02X' % b for b in multiord(self.payload)))


class BT(object):
    '''Implements the non-Myo-specific details of the Bluetooth protocol.'''
    def __init__(self, tty):
        self.ser = serial.Serial(port=tty, baudrate=9600, dsrdtr=1)
        self.buf = []
        self.lock = threading.Lock()
        self.handlers = []

    ## internal data-handling methods
    def recv_packet(self, timeout=None):
        t0 = time.time()
        self.ser.timeout = None
        while timeout is None or time.time() < t0 + timeout:
            if timeout is not None: 
                remaining = t0 + timeout - time.time()
                if remaining <= 0:
                    return None
                self.ser.timeout = remaining
            c = self.ser.read()
            if not c: return None

            ret = self.proc_byte(ord(c))
            if ret:
                if ret.typ == 0x80:
                    self.handle_event(ret)
                return ret

    def recv_packets(self, timeout=.5):
        res = []
        t0 = time.time()
        while time.time() < t0 + timeout:
            p = self.recv_packet(t0 + timeout - time.time())
            if not p: return res
            res.append(p)
        return res

    def proc_byte(self, c):
        if not self.buf:
            if c in [0x00, 0x80, 0x08, 0x88]:
                self.buf.append(c)
            return None
        elif len(self.buf) == 1:
            self.buf.append(c)
            self.packet_len = 4 + (self.buf[0] & 0x07) + self.buf[1]
            return None
        else:
            self.buf.append(c)

        if self.packet_len and len(self.buf) == self.packet_len:
            p = Packet(self.buf)
            self.buf = []
            return p
        return None

    def handle_event(self, p):
        for h in self.handlers:
            h(p)

    def add_handler(self, h):
        self.handlers.append(h)

    def remove_handler(self, h):
        try: self.handlers.remove(h)
        except ValueError: pass

    def wait_event(self, cls, cmd):
        res = [None]
        def h(p):
            if p.cls == cls and p.cmd == cmd:
                res[0] = p
        self.add_handler(h)
        while res[0] is None:
            self.recv_packet()
        self.remove_handler(h)
        return res[0]

    ## specific BLE commands
    def connect(self, addr):
        return self.send_command(6, 3, pack('6sBHHHH', multichr(addr), 0, 6, 6, 64, 0))

    def get_connections(self):
        return self.send_command(0, 6)

    def discover(self):
        return self.send_command(6, 2, b'\x01')

    def end_scan(self):
        return self.send_command(6, 4)

    def disconnect(self, h):
        return self.send_command(3, 0, pack('B', h))

    def read_attr(self, con, attr):
        self.send_command(4, 4, pack('BH', con, attr))
        return self.wait_event(4, 5)

    def write_attr(self, con, attr, val):
        self.send_command(4, 5, pack('BHB', con, attr, len(val)) + val)
        return self.wait_event(4, 1)

    def send_command(self, cls, cmd, payload=b'', wait_resp=True):
        s = pack('4B', 0, len(payload), cls, cmd) + payload
        self.ser.write(s)

        while True:
            p = self.recv_packet()

            ## no timeout, so p won't be None
            if p.typ == 0: return p

            ## not a response: must be an event
            self.handle_event(p)


class MyoRaw(object):
    '''Implements the Myo-specific communication protocol.'''

    def __init__(self, tty=None):
        if tty is None:
            tty = self.detect_tty()
        if tty is None:
            raise ValueError('Myo dongle not found!')

        self.bt = BT(tty)
        self.conn = None
        self.emg_handlers = []
        self.imu_handlers = []
        self.arm_handlers = []
        self.pose_handlers = []

    def detect_tty(self):
        for p in comports():
            if re.search(r'PID=2458:0*1', p[2]):
                print('using device:', p[0])
                return p[0]

        return None

    def run(self, timeout=None):
        self.bt.recv_packet(timeout)

    def connect(self):
        ## stop everything from before
        self.bt.end_scan()
        self.bt.disconnect(0)
        self.bt.disconnect(1)
        self.bt.disconnect(2)

        ## start scanning
        print('scanning...')
        self.bt.discover()
        
        start_time = time.time()
        timeout = 10
        addr = None
        
        while time.time() - start_time < timeout:
            p = self.bt.recv_packet(0.5)
            if p is None:
                continue
                
            print('scan response:', p)

            if p.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                addr = list(multiord(p.payload[2:8]))
                print(f'Myo encontrado! Endereço: {addr}')
                break
        
        self.bt.end_scan()

        if addr is None:
            raise ValueError('Myo não encontrado após 10 segundos de scanning!')

        ## connect and wait for status event
        print('connecting...')
        conn_pkt = self.bt.connect(addr)
        self.conn = multiord(conn_pkt.payload)[-1]
        self.bt.wait_event(3, 0)
        print('connected!')

        ## get firmware version
        fw = self.read_attr(0x17)
        _, _, _, _, v0, v1, v2, v3 = unpack('BHBBHHHH', fw.payload)
        print('firmware version: %d.%d.%d.%d' % (v0, v1, v2, v3))

        self.old = (v0 == 0)

        if self.old:
            ## don't know what these do; Myo Connect sends them, though we get data
            ## fine without them
            self.write_attr(0x19, b'\x01\x02\x00\x00')
            self.write_attr(0x2f, b'\x01\x00')
            self.write_attr(0x2c, b'\x01\x00')
            self.write_attr(0x32, b'\x01\x00')
            self.write_attr(0x35, b'\x01\x00')

            ## enable EMG data
            self.write_attr(0x28, b'\x01\x00')
            ## enable IMU data
            self.write_attr(0x1d, b'\x01\x00')

            ## Sampling rate of the underlying EMG sensor, capped to 1000. If it's
            ## less than 1000, emg_hz is correct. If it is greater, the actual
            ## framerate starts dropping inversely. Also, if this is much less than
            ## 1000, EMG data becomes slower to respond to changes. In conclusion,
            ## 1000 is probably a good value.
            C = 1000
            emg_hz = 50
            ## strength of low-pass filtering of EMG data
            emg_smooth = 100

            imu_hz = 50

            ## send sensor parameters, or we don't get any data
            self.write_attr(0x19, pack('BBBBHBBBBB', 2, 9, 2, 1, C, emg_smooth, C // emg_hz, imu_hz, 0, 0))

        else:
            name = self.read_attr(0x03)
            print('device name: %s' % name.payload)

            ## enable IMU data
            self.write_attr(0x1d, b'\x01\x00')
            ## enable on/off arm notifications
            self.write_attr(0x24, b'\x02\x00')

            # self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
            self.start_raw()

        ## add data handlers
        def handle_data(p):
            if (p.cls, p.cmd) != (4, 5): return

            c, attr, typ = unpack('BHB', p.payload[:4])
            pay = p.payload[5:]

            if attr == 0x27:
                vals = unpack('8HB', pay)
                ## not entirely sure what the last byte is, but it's a bitmask that
                ## seems to indicate which sensors think they're being moved around or
                ## something
                emg = vals[:8]
                moving = vals[8]
                self.on_emg(emg, moving)
            elif attr == 0x1c:
                vals = unpack('10h', pay)
                quat = vals[:4]
                acc = vals[4:7]
                gyro = vals[7:10]
                self.on_imu(quat, acc, gyro)
            elif attr == 0x23:
                typ, val, xdir = unpack('3B', pay[:3])

                if typ == 1: # on arm
                    self.on_arm(Arm(val), XDirection(xdir))
                elif typ == 2: # removed from arm
                    self.on_arm(Arm.UNKNOWN, XDirection.UNKNOWN)
                elif typ == 3: # pose
                    self.on_pose(Pose(val))
            else:
                print('data with unknown attr: %02X %s' % (attr, p))

        self.bt.add_handler(handle_data)


    def write_attr(self, attr, val):
        if self.conn is not None:
            self.bt.write_attr(self.conn, attr, val)

    def read_attr(self, attr):
        if self.conn is not None:
            return self.bt.read_attr(self.conn, attr)
        return None

    def disconnect(self):
        if self.conn is not None:
            self.bt.disconnect(self.conn)

    def start_raw(self):
        '''Sending this sequence for v1.0 firmware seems to enable both raw data and
        pose notifications.
        '''

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def mc_start_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when starting data
        collection for v1.0 firmware; this enables raw data but disables arm and
        pose notifications.
        '''

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x09\x01\x01\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x00')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x00')

    def mc_end_collection(self):
        '''Myo Connect sends this sequence (or a reordering) when ending data collection
        for v1.0 firmware; this reenables arm and pose notifications, but
        doesn't disable raw data.
        '''

        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')
        self.write_attr(0x19, b'\x09\x01\x00\x00\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x00\x01\x01')
        self.write_attr(0x28, b'\x01\x00')
        self.write_attr(0x1d, b'\x01\x00')
        self.write_attr(0x24, b'\x02\x00')
        self.write_attr(0x19, b'\x01\x03\x01\x01\x01')

    def vibrate(self, length):
        if length in range(1, 4):
            ## first byte tells it to vibrate; purpose of second byte is unknown
            self.write_attr(0x19, pack('3B', 3, 1, length))


    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)

    def add_pose_handler(self, h):
        self.pose_handlers.append(h)

    def add_arm_handler(self, h):
        self.arm_handlers.append(h)


    def on_emg(self, emg, moving):
        for h in self.emg_handlers:
            h(emg, moving)

    def on_imu(self, quat, acc, gyro):
        for h in self.imu_handlers:
            h(quat, acc, gyro)

    def on_pose(self, p):
        for h in self.pose_handlers:
            h(p)

    def on_arm(self, arm, xdir):
        for h in self.arm_handlers:
            h(arm, xdir)

if __name__ == '__main__':
    try:
        import pygame
        from pygame.locals import *
        HAVE_PYGAME = False
    except ImportError:
        HAVE_PYGAME = False

    if HAVE_PYGAME:
        w, h = 1200, 400
        scr = pygame.display.set_mode((w, h))

    last_vals = None
    def plot(scr, vals):
        DRAW_LINES = True
        global last_vals
        if last_vals is None:
            last_vals = vals
            return

        D = 5
        scr.scroll(-D)
        scr.fill((0,0,0), (w - D, 0, w, h))
        for i, (u, v) in enumerate(zip(last_vals, vals)):
            if DRAW_LINES:
                pygame.draw.line(scr, (0,255,0),
                                 (w - D, int(h/8 * (i+1 - u))),
                                 (w, int(h/8 * (i+1 - v))))
                pygame.draw.line(scr, (255,255,255),
                                 (w - D, int(h/8 * (i+1))),
                                 (w, int(h/8 * (i+1))))
            else:
                c = int(255 * max(0, min(1, v)))
                scr.fill((c, c, c), (w - D, i * h / 8, D, (i + 1) * h / 8 - i * h / 8));

        pygame.display.flip()
        last_vals = vals

    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

    def proc_emg(emg, moving, times=[]):
        if HAVE_PYGAME:
            ## update pygame display
            plot(scr, [e / 2000. for e in emg])
        #else:
            
            #print(emg)
            #print(vals)

        ## print framerate of received data
        times.append(time.time())
        if len(times) > 20:
            #print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0)
            
    trajectory = Trajectory3D()
    
    # Configurar os plots
    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 3D
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Plots 2D para cada eixo
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    
    plt.tight_layout()
    
    last_plot_time = time.time()
    plot_active = True

    def on_imu(quat, acc, gyro):
        trajectory.update(quat, acc, gyro)
        
    m.add_imu_handler(on_imu)
    m.add_emg_handler(proc_emg)
    m.connect()

    m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
    m.add_pose_handler(lambda p: print('pose', p))

    def update_plots():
        # Limpar todos os plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        
        # Obter históricos
        hist_ekf = trajectory.get_history()
        hist_direct = trajectory.get_direct_history()
        current_state = trajectory.get_current_state()
        position = current_state['position']
        position_direct = current_state['position_direct']
        
        # Plot 3D - Trajetória
        if len(hist_ekf) > 1:
            # Trajetória EKF
            ax1.plot(hist_ekf[:,0], hist_ekf[:,1], hist_ekf[:,2], 'b-', linewidth=2, label='EKF')
            ax1.scatter([position[0]], [position[1]], [position[2]], c='b', s=50, marker='o')
            
            # Trajetória integração direta
            ax1.plot(hist_direct[:,0], hist_direct[:,1], hist_direct[:,2], 'r-', linewidth=1, alpha=0.7, label='Integração Direta')
            ax1.scatter([position_direct[0]], [position_direct[1]], [position_direct[2]], c='r', s=30, marker='x')
            
            # Configurar plot 3D
            b = 0.5
            ax1.set_xlim([-b, b])
            ax1.set_ylim([-b, b])
            ax1.set_zlim([-b, b])
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('Trajetória 3D - EKF vs Integração Direta')
            ax1.legend()
        
        # Plot 2D - Comparação por eixo
        time_axis = np.arange(len(hist_ekf))
        
        # Eixo X - só adicionar legenda se houver dados
        if len(hist_ekf) > 0:
            line_x_ekf, = ax2.plot(time_axis, hist_ekf[:,0], 'b-', label='EKF')
            line_x_direct, = ax2.plot(time_axis, hist_direct[:,0], 'r-', alpha=0.7, label='Integração Direta')
            ax2.legend()
        ax2.set_xlabel('Amostras')
        ax2.set_ylabel('X (m)')
        ax2.set_title('Posição X')
        ax2.grid(True)
        
        # Eixo Y - só adicionar legenda se houver dados
        if len(hist_ekf) > 0:
            line_y_ekf, = ax3.plot(time_axis, hist_ekf[:,1], 'b-', label='EKF')
            line_y_direct, = ax3.plot(time_axis, hist_direct[:,1], 'r-', alpha=0.7, label='Integração Direta')
            ax3.legend()
        ax3.set_xlabel('Amostras')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Posição Y')
        ax3.grid(True)
        
        # Eixo Z - só adicionar legenda se houver dados
        if len(hist_ekf) > 0:
            line_z_ekf, = ax4.plot(time_axis, hist_ekf[:,2], 'b-', label='EKF')
            line_z_direct, = ax4.plot(time_axis, hist_direct[:,2], 'r-', alpha=0.7, label='Integração Direta')
            ax4.legend()
        ax4.set_xlabel('Amostras')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('Posição Z')
        ax4.grid(True)
        
        # Diferenças entre EKF e integração direta - só adicionar legenda se houver dados
        if len(hist_ekf) > 0:
            diff = hist_ekf - hist_direct
            line_diff_x, = ax5.plot(time_axis, diff[:,0], 'g-', label='X')
            line_diff_y, = ax5.plot(time_axis, diff[:,1], 'orange', label='Y')
            line_diff_z, = ax5.plot(time_axis, diff[:,2], 'red', label='Z')
            ax5.legend()
        ax5.set_xlabel('Amostras')
        ax5.set_ylabel('Diferença (m)')
        ax5.set_title('Diferença EKF - Integração Direta')
        ax5.grid(True)
        
        # Valores atuais
        current_data = {
            'EKF X': position[0],
            'EKF Y': position[1], 
            'EKF Z': position[2],
            'Direct X': position_direct[0],
            'Direct Y': position_direct[1],
            'Direct Z': position_direct[2]
        }
        
        bars = ax6.bar(range(len(current_data)), list(current_data.values()))
        ax6.set_xticks(range(len(current_data)))
        ax6.set_xticklabels(list(current_data.keys()), rotation=45)
        ax6.set_ylabel('Posição (m)')
        ax6.set_title('Valores Atuais')
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.draw()
        plt.pause(0.001)

    try:
        while True:
            m.run(0.02)  # Loop mais rápido
            
            # Atualiza plots a cada 0.1 segundos
            if time.time() - last_plot_time >= 0.1 and plot_active:
                update_plots()
                last_plot_time = time.time()

            if HAVE_PYGAME:
                for ev in pygame.event.get():
                    if ev.type == QUIT or (ev.type == KEYDOWN and ev.unicode == 'q'):
                        raise KeyboardInterrupt()
                    elif ev.type == KEYDOWN:
                        if K_1 <= ev.key <= K_3:
                            m.vibrate(ev.key - K_0)
                        if K_KP1 <= ev.key <= K_KP3:
                            m.vibrate(ev.key - K_KP0)
                        if ev.key == K_r:  # Reset com tecla 'r'
                            trajectory.reset()
                        if ev.key == K_p:  # Pausa/continua com tecla 'p'
                            plot_active = not plot_active
    
    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print()