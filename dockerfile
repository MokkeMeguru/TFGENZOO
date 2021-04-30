FROM tensorflow/tensorflow:2.4.0-gpu
RUN apt-add-repository ppa:fish-shell/release-3
RUN apt-get update
RUN apt-get install -y zsh tmux wget
RUN mkdir /workspace
RUN pip install -U pip
ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
WORKDIR /workspace
