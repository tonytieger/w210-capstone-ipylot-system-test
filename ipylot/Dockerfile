#to build: build.sh
#to run: host_runner.sh

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r32.6.1-py3

FROM ${BASE_IMAGE} 

ENV UNAME drowuser

RUN apt update && apt install -y ffmpeg vim
RUN apt install -y pulseaudio && apt install -y pulseaudio-utils

# Set up the user
RUN export UNAME=$UNAME UID=1000 GID=1000 && \
    mkdir -p "/home/${UNAME}" && \
    echo "${UNAME}:x:${UID}:${GID}:${UNAME} User,,,:/home/${UNAME}:/bin/bash" >> /etc/passwd && \
    echo "${UNAME}:x:${UID}:" >> /etc/group && \
    mkdir -p /etc/sudoers.d && \
    echo "${UNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${UNAME} && \
    chmod 0440 /etc/sudoers.d/${UNAME} && \
    chown ${UID}:${GID} -R /home/${UNAME} && \
    gpasswd -a ${UNAME} audio

COPY pulse-client.conf /etc/pulse/client.conf

RUN pip3 install python-vlc
RUN pip3 install --upgrade nbconvert
RUN pip3 install tqdm
RUN apt install -y vlc
WORKDIR /home/drowuser

USER $UNAME
ENV HOME /home/drowuser

CMD ["bash"]

