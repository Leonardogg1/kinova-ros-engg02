# Base com ROS Noetic e Gazebo 11
FROM ros:noetic

# Configurações de ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    QT_QPA_PLATFORM=xcb \
    LIBGL_ALWAYS_SOFTWARE=1 \
    DISPLAY=:0 \
    XAUTHORITY=/tmp/.docker.xauth \
    CATKIN_WS=/root/catkin_ws

# 1. Configuração otimizada de repositórios e dependências
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    terminator \
    nano\
    curl \
    wget \
    lsb-release \
    build-essential \
    python3-pip \
    python3-serial\
    python3-catkin-tools \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    ros-noetic-joint-state-publisher \
    ros-noetic-robot-state-publisher \
    ros-noetic-xacro \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-control \
    ros-noetic-moveit \
    ros-noetic-tf-conversions \
    ros-noetic-eigen-conversions \
    ros-noetic-controller-manager \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-trac-ik-kinematics-plugin \
    ros-noetic-trac-ik \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-randr0 \
    mesa-utils \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Inicializa rosdep
RUN [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ] && rosdep init || echo "rosdep already initialized" \
 && rosdep update

# Define o workspace
ENV CATKIN_WS=/root/catkin_ws

# Cria estrutura do workspace
RUN mkdir -p $CATKIN_WS/src

# Clona o repositório da Kinova (branch ros1)
WORKDIR $CATKIN_WS/src
RUN git clone -b noetic-devel https://github.com/Leonardogg1/kinova-ros-engg02.git

# Corrige permissões se necessário
RUN chmod -R a+rw $CATKIN_WS

# Compila o workspace
WORKDIR $CATKIN_WS
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Configurações finais de ambiente
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export QT_QPA_PLATFORM=xcb" >> /root/.bashrc && \
    echo "export LIBGL_ALWAYS_SOFTWARE=1" >> /root/.bashrc && \
    echo "export GAZEBO_IP=127.0.0.1" >> /root/.bashrc

# Comando padrão ao iniciar o container
CMD ["bash"]

