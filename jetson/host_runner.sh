#!/usr/bin/env bash

set -x

USER_UID=$(id -u)
USER_NAME=$(whoami)

docker run -it --rm --net=host --runtime nvidia -v /dev/video0:/dev/video0 --volume=/run/user/${USER_UID}/pulse:/run/user/0/pulse -e DISPLAY=$DISPLAY -v /home/${USER_NAME}:/home/drowuser --privileged drow

