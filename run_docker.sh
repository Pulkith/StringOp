#!/bin/bash

# Define variables
IMAGE_NAME="osrf/ros:foxy-desktop"  # Replace with your image name
HOST_DIR="/home/ben" # Host directory to mount
CONTAINER_DIR="/home/ros2_user/" # Directory inside container
USER_NAME="ros2_user"         # Non-root user name
USER_ID=$(id -u)              # Host user ID
GROUP_ID=$(id -g)             # Host group ID
CONTAINER_NAME="ros2_foxy_container"

# Ensure the host directory exists
if [ ! -d "$HOST_DIR" ]; then
  echo "Error: Host directory $HOST_DIR does not exist."
  exit 1
fi

# Trap Ctrl+C to ensure cleanup (optional, as --rm handles most cases)
trap 'echo "Caught Ctrl+C, exiting..."; exit 0' SIGINT

# Check if container already exists
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
  echo "Container $CONTAINER_NAME exists. Removing it..."
  docker rm -f $CONTAINER_NAME > /dev/null
fi

# Run new container with --rm to auto-remove on exit
echo "Creating and starting new container $CONTAINER_NAME..."
docker run -it --rm \
  -v "$HOST_DIR:$CONTAINER_DIR" \
  --name $CONTAINER_NAME \
  --network host \
  -e USER_ID=$USER_ID \
  -e GROUP_ID=$GROUP_ID \
  -e USER_NAME=$USER_NAME \
  $IMAGE_NAME \
  /bin/bash -c "
    # Create group and user with matching UID/GID
    groupadd -g \$GROUP_ID \$USER_NAME &&
    useradd -u \$USER_ID -g \$GROUP_ID -m -s /bin/bash \$USER_NAME &&
    # Ensure mounted directory is accessible
    chown \$USER_NAME:\$USER_NAME $CONTAINER_DIR &&
    # Switch to non-root user and start bash
    su - \$USER_NAME -c 'source /opt/ros/foxy/setup.bash && /bin/bash'
  "
