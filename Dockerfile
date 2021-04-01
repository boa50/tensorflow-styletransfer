# For more information, please refer to https://aka.ms/vscode-docker-python
FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
ADD . /app

ARG USERNAME=boa50
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo git \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN pip install -r requirements.txt

RUN chown -R $USERNAME /app
USER $USERNAME