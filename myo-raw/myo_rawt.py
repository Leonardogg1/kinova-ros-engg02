from __future__ import print_function

import enum
import re
import struct
import sys
import threading
import time

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
        self.last_time = time.time()
        
        # Parâmetros para detecção de repouso
        self.stationary_threshold = 1.55
        self.stationary_count = 0
        self.accel_bias = np.zeros(3)
        self.bias_learning_rate = 0.05

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
        
        # Gerir histórico
        self.history.append(self.position.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return self.position.copy()

    def get_current_state(self):
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy()
        }

    def get_history(self):
        return np.array(self.history.copy())

    def reset(self):
        self.ekf.reset()
        self.history = []
        self.last_time = time.time()
        self.accel_bias = np.zeros(3)

# O resto do código permanece igual...

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
                if remaining <= 0:  # Evita valores negativos
                    return None
                self.ser.timeout = remaining
            else:
                self.ser.timeout = None
                
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
        while True:
            p = self.bt.recv_packet()
            print('scan response:', p)

            if p.payload.endswith(b'\x06\x42\x48\x12\x4A\x7F\x2C\x48\x47\xB9\xDE\x04\xA9\x01\x00\x06\xD5'):
                addr = list(multiord(p.payload[2:8]))
                break
        self.bt.end_scan()

        ## connect and wait for status event
        conn_pkt = self.bt.connect(addr)
        self.conn = multiord(conn_pkt.payload)[-1]
        self.bt.wait_event(3, 0)

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
        HAVE_PYGAME = True
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    last_plot_time = time.time()
    plot_active = True

    def on_imu(quat, acc, gyro):
        trajectory.update(quat, acc, gyro)
        
    m.add_imu_handler(on_imu)
    m.add_emg_handler(proc_emg)
    m.connect()

    m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
    m.add_pose_handler(lambda p: print('pose', p))

    def update_3d_plot():
        ax.clear()
        hist = trajectory.get_history()
        current_state = trajectory.get_current_state()
        position = current_state['position']
        
        if len(hist) > 1:
            ax.plot(hist[:,0], hist[:,1], hist[:,2], 'b-', linewidth=1.5)
            ax.scatter([position[0]], [position[1]], [position[2]], c='r', s=50)
        
        # Configura escala fixa
        b = 0.5
        ax.set_xlim([-b, b])
        ax.set_ylim([-b, b])
        ax.set_zlim([-b, b])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Trajetória 3D com Filtro de Kalman Estendido')
        
        plt.draw()
        plt.pause(0.001)

    try:
        while True:
            m.run(0.02)  # Loop mais rápido
            
            # Atualiza plot 3D a cada 0.1 segundos
            if time.time() - last_plot_time >= 0.1 and plot_active:
                update_3d_plot()
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