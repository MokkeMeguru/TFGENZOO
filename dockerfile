FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN mkdir /workspace
RUN pip install -U pip
COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
