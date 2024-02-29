FROM python:3.11-buster as builder-image

RUN apt-get update

COPY ../install/requirements_py3.11.txt .
RUN pip3 install -U pip
RUN pip3 install --no-cache-dir -r requirements_py3.11.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /opt/code
ENV PYTHONPATH /opt/code/VectorDBBench
