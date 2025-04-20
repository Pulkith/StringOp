#!/bin/bash
# IMAGE_NAME="osrf/ros:foxy-desktop"  # Replace with your image name
IMAGE_NAME="ros2_foxy_container_w_deps:latest"
HOST_DIR="/home/anoca/repos" # Host directory to mount
CONTAINER_DIR="/home/ros2_user/" # Directory inside container
USER_NAME="ros2_user"         # Non-root user name
USER_ID=$(id -u)              # Host user ID
GROUP_ID=$(id -g)             # Host group ID
CONTAINER_NAME="ros2_foxy_container"
USER_PASSWORD="eth"      # Password for ros2_user
DISPLAY=":0"


# Ensure the host directory exists
if [ ! -d "$HOST_DIR" ]; then
  echo "Error: Host directory $HOST_DIR does not exist."
  exit 1
fi

# xhost +local:docker > /dev/null

# Trap Ctrl+C to ensure cleanup (optional, as --rm handles most cases)
# trap 'echo "Caught Ctrl+C, exiting..."; exit 0' SIGINT

trap '' SIGINT

# # Check if container already exists
# if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
#   echo "Container $CONTAINER_NAME exists. Removing it..."
#   docker rm -f $CONTAINER_NAME > /dev/null
# fi

# Check if container is running
if [ "$(docker ps -q -f name=^${CONTAINER_NAME}$)" ]; then
  echo "Container $CONTAINER_NAME is already running. Joining..."
  docker exec -it -u $USER_NAME $CONTAINER_NAME bash -c "source /opt/ros/foxy/setup.bash && exec bash"
  exit 0
fi

# Run new container with --rm to auto-remove on exit
echo "Creating and starting new container $CONTAINER_NAME..."
docker run -dit \
  -v "$HOST_DIR:$CONTAINER_DIR" \
  -e DISPLAY=$DISPLAY \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --name $CONTAINER_NAME \
  --network host \
  -e USER_ID=$USER_ID \
  -e GROUP_ID=$GROUP_ID \
  -e USER_NAME=$USER_NAME \
  --privileged \
  --sig-proxy=false \
  $IMAGE_NAME \
  tail -f /dev/null
  # /bin/bash -c "
  #   # Create group if it doesn't exist
  #   groupadd -g \$GROUP_ID \$USER_NAME || true &&
  #   # Create user with matching UID/GID and home directory
  #   useradd -u \$USER_ID -g \$GROUP_ID -m -s /bin/bash \$USER_NAME || true &&
  #   # Ensure mounted directory is accessible
  #   chown \$USER_NAME:\$USER_NAME $CONTAINER_DIR &&
  #   # Install sudo if not present
  #   # Configure passwordless sudo for the user
  #   echo \"\$USER_NAME ALL=(ALL) NOPASSWD:ALL\" > /etc/sudoers.d/\$USER_NAME &&
  #   chmod 0440 /etc/sudoers.d/\$USER_NAME &&
  #   # Source ROS 2 setup and start bash
  #   # su - \$USER_NAME -c 'source /opt/ros/foxy/setup.bash && export DISPLAY=":0" && /bin/bash' 
    
  # "

docker exec -it $CONTAINER_NAME /bin/bash -c "
    # Create group if it doesn't exist
    groupadd -g \$GROUP_ID \$USER_NAME || true &&
    # Create user with matching UID/GID and home directory
    useradd -u \$USER_ID -g \$GROUP_ID -m -s /bin/bash \$USER_NAME || true &&
    # Ensure mounted directory is accessible
    chown \$USER_NAME:\$USER_NAME $CONTAINER_DIR &&
    # Install sudo if not present
    # Configure passwordless sudo for the user
    echo \"\$USER_NAME ALL=(ALL) NOPASSWD:ALL\" > /etc/sudoers.d/\$USER_NAME &&
    chmod 0440 /etc/sudoers.d/\$USER_NAME
    # Source ROS 2 setup and start bash
    su - \$USER_NAME -c 'source /opt/ros/foxy/setup.bash && export DISPLAY=":0" && /bin/bash' 
    
  "


# apt-get update && apt-get install -y sudo &&
