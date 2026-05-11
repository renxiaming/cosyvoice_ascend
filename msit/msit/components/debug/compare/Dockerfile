FROM ubuntu:18.04
SHELL ["/bin/bash", "-c"]
ARG CANN_TOOLKIT_PATH
ADD $CANN_TOOLKIT_PATH ./
ARG CANN_AMCT_PATH
COPY $CANN_AMCT_PATH ./
ARG CAFFE_SRC
COPY $CAFFE_SRC ./
ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_X86_ARCHIVE
ARG UBUNTU_X86_SECURITY
ARG UBUNTU_ARM64 
ARG APT_PATH
ARG PYTHON_PATH
ARG PYPI_PATH
ARG PYPI_PATH_TRUST
ARG MSIT_PATH

#安装python、CANN_TOOLKIT,软件包、依赖，并配置环境变量写入.bashrc
RUN groupadd HwHiAiUser && useradd -rm -d /home/HwHiAiUser -s /bin/bash -g HwHiAiUser -G HwHiAiUser -u 1001 HwHiAiUser  &&\
    if [ "$(uname -m)" = "x86_64" ]; then \
        sed -i "s@${UBUNTU_X86_ARCHIVE}@${APT_PATH}@g" /etc/apt/sources.list \
        && sed -i "s@${UBUNTU_X86_SECURITY}@${APT_PATH}@g" /etc/apt/sources.list; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        sed -i "s@${UBUNTU_ARM64}@${APT_PATH}@g" /etc/apt/sources.list; \
    fi  &&\
    apt-get update && \
    apt-get install libbz2-dev -y &&\
    apt-get install liblzma-dev -y &&\
    apt-get install -y wget gnupg software-properties-common && \
    apt-get install -y sudo gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev libncursesw5-dev wget dkms &&\
    wget ${PYTHON_PATH}   && \
    tar -zxvf Python-3.7.5.tgz &&\
    cd Python-3.7.5 &&\
    ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared &&\
    make -j`nproc` &&\
    make install &&\
    cd - &&\ 
    ln -s /usr/local/python3.7.5/bin/python3 /usr/local/python3.7.5/bin/python3.7.5 &&\
    ln -s /usr/local/python3.7.5/bin/pip3 /usr/local/python3.7.5/bin/pip3.7.5 &&\
    echo python37_install_path=/usr/local/python3.7.5 > /etc/use_private_python.info &&\
    echo "export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH" >> /root/.bashrc &&\
    echo "export PATH=/usr/local/python3.7.5/bin:\$PATH" >> /root/.bashrc &&\
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH &&\
    export PATH=/usr/local/python3.7.5/bin:$PATH &&\
    pip3.7 config set global.index-url ${PYPI_PATH} && \
    pip3.7 config set global.trusted-host ${PYPI_PATH_TRUST} && \
    pip3.7 config set global.timeout 600 && \
    pip3.7 install --upgrade pip &&\
    pip3.7 install attrs &&\
    pip3.7 install numpy==1.17.2 &&\
    pip3.7 install decorator &&\
    pip3.7 install sympy &&\
    pip3.7 install cffi &&\
    pip3.7 install pyyaml &&\
    pip3.7 install pathlib2 &&\
    pip3.7 install psutil &&\
    pip3.7 install protobuf &&\
    pip3.7 install scipy &&\
    pip3.7 install requests &&\
    pip3.7 install cloudpickle &&\
    pip3.7 install synr==0.5.0 &&\
    pip3.7 install tornado absl-py &&\
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH &&\
    export PATH=/usr/local/python3.7.5/bin:$PATH &&\
    chmod +x ./$CANN_TOOLKIT_PATH && ./$CANN_TOOLKIT_PATH --check &&\
    echo "y" | ./$CANN_TOOLKIT_PATH --install-for-all --devel &&\
    echo "export PATH=\$PATH:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin" >> /root/.bashrc && \
    echo "export PATH=\$PATH:/usr/local/Ascend/ascend-toolkit/latest/atc/bin" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub" >> /root/.bashrc && \
    echo "export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest" >> /root/.bashrc && \
    echo "export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp" >> /root/.bashrc && \
    echo "export TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit" >> /root/.bashrc && \
    rm Ascend-*.run && rm -rf Python-3.7.5*

#CANN AMCT 量化工具
RUN apt-get update &&\
    apt install -y vim libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev git libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler &&\
    apt install -y --no-install-recommends libboost-all-dev &&\
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH &&\
    export PATH=/usr/local/python3.7.5/bin:$PATH &&\
    if [ "$(uname -m)" = "x86_64" ]; then \
        ln -sf /usr/lib/x86_64-linux-gnu/libhdf5_serial.so /usr/lib/x86_64-linux-gnu/libhdf5.so && \
        ln -sf /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so /usr/lib/x86_64-linux-gnu/libhdf5_hl.so; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        ln -sf /usr/lib/aarch64-linux-gnu/libhdf5_serial.so /usr/lib/aarch64-linux-gnu/libhdf5.so && \
        ln -sf /usr/lib/aarch64-linux-gnu/libhdf5_serial_hl.so /usr/lib/aarch64-linux-gnu/libhdf5_hl.so; \
    fi && \
    python3.7.5 -m pip install --upgrade pip  && \
    python3.7.5 -m pip install --trusted-host  opencv-python gnureadline scikit-image  && \
    unzip *.zip &&\
    tar -zxvf ./$CANN_AMCT_PATH && \
    rm *.zip && \
    pip3.7 install amct/amct_caffe/amct_caffe-*.whl && \
    export CAFFE_REL_PATH=`find . -maxdepth 1 -name caffe* -type d` && \
    export CAFFE_ABS_PATH=`realpath $CAFFE_REL_PATH` && \
    cd amct/amct_caffe/ && \
    tar -zxvf caffe_patch.tar.gz && \
    cd caffe_patch &&\
    python3.7 install.py --caffe_dir $CAFFE_ABS_PATH --custom_proto $CAFFE_ABS_PATH/custom.proto &&\
    cd $CAFFE_ABS_PATH && \
    make clean && \
    make all -j && make pycaffe -j &&\
    echo "export PYTHONPATH=\$PYTHONPATH:$CAFFE_ABS_PATH/python" >> ~/.bashrc &&\
    echo "export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/6.2.RC2/pyACL/python/site-packages/acl:\$PYTHONPATH" >> ~/.bashrc &&\
    echo "unset ASCEND_AICPU_PATH" >> ~/.bashrc &&\
    source /usr/local/Ascend/ascend-toolkit/set_env.sh &&\
    source ~/.bashrc

#安装msit以及依赖
RUN apt-get install -y git &&\
    git -c http.sslVerify=false clone ${MSIT_PATH} &&\
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH &&\
    export PATH=/usr/local/python3.7.5/bin:$PATH &&\
    export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH &&\
    export PATH=/usr/local/python3.7.5/bin:$PATH &&\
    cd msit/msit &&\
    chmod +x install.sh &&\
    ./install.sh

RUN apt-get autoclean &&\
    apt-get clean &&\
    apt-get autoremove


