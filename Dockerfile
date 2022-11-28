FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
COPY ./requirements.txt /requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r /requirements.txt