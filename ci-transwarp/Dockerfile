FROM ubuntu:22.04

ARG build_dir=/opt/transwarp/vectordb_bench

RUN \
    # basics
    PKGS="software-properties-common vim sudo locales git" && \
    apt-get update && \
    apt-get install -y ${PKGS} && \
    locale-gen en_US.UTF-8 && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/timezone && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    # python
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3-pip python3.11 python3.11-dev && \
    apt-get clean && apt-get autoclean

ENV LC_ALL="en_US.UTF-8"
ENV LANG="en_US.UTF-8"

COPY ci-transwarp/pip.conf /root/.config/pip/pip.conf
COPY . ${build_dir}

RUN \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip && \
    python -m pip install hippo-api==1.2.0rc1 && \
    cd ${build_dir} && python -m pip install .[test] && cd ${build_dir} && rm -rf ./*

VOLUME [ "${build_dir}" ]

WORKDIR ${build_dir}
