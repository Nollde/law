FROM centos:7

# labels
LABEL law.version="0.0.41"
LABEL law.image_name="riga/law"
LABEL law.image_tag="py38"
LABEL law.image_dir="law-py38"
LABEL law.image_python_major="3"
LABEL law.image_python_minor="8"
LABEL law.image_python_patch="5"
LABEL law.image_python="3.8.5"
LABEL law.image_python_mm="3.8"

# law specific environment variables
ENV LAW_SANDBOX docker::riga/law:py38,docker::riga/law:py3,docker::riga/law:latest,docker::riga/law
ENV LAW_IMAGE_NAME law-py38
ENV LAW_IMAGE_TAG py38
ENV LAW_IMAGE_PYTHON_MAJOR 3
ENV LAW_IMAGE_PYTHON_MINOR 8
ENV LAW_IMAGE_PYTHON_PATCH 5
ENV LAW_IMAGE_PYTHON $LAW_IMAGE_PYTHON_MAJOR.$LAW_IMAGE_PYTHON_MINOR.$LAW_IMAGE_PYTHON_PATCH
ENV LAW_IMAGE_PYTHON_MM $LAW_IMAGE_PYTHON_MAJOR.$LAW_IMAGE_PYTHON_MINOR
ENV LAW_IMAGE_SOURCE_DIR /root/law

# basic environment variables
ENV CPATH /usr/local/include/python${LAW_IMAGE_PYTHON_MM}:/usr/local/include/python${LAW_IMAGE_PYTHON_MM}m:$CPATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV GFAL_PLUGIN_DIR /usr/lib64/gfal2-plugins

# expose ports
EXPOSE 8082

# installation workdir
WORKDIR /root/install

# prepare yum
RUN yum -y update; yum clean all
RUN yum -y install yum-plugin-priorities yum-utils; yum clean all
RUN yum install -y epel-release; yum clean all

# basic software
RUN yum -y groupinstall development; yum clean all
RUN yum -y install libffi-devel openssl-devel bzip2-devel json-c-devel curl-devel gcc-c++ which \
    wget nano screen git cmake cmake3; yum clean all

# python software
RUN wget -nv https://www.python.org/ftp/python/${LAW_IMAGE_PYTHON}/Python-${LAW_IMAGE_PYTHON}.tgz && \
    tar -xzf Python-${LAW_IMAGE_PYTHON}.tgz && \
    cd Python-${LAW_IMAGE_PYTHON} && \
    ./configure --enable-optimizations --enable-shared && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${LAW_IMAGE_PYTHON} && \
    rm Python-${LAW_IMAGE_PYTHON}.tgz && \
    ln -s -f "$( which python${LAW_IMAGE_PYTHON_MM} )" "$( dirname "$( which python${LAW_IMAGE_PYTHON_MM} )" )/python3" && \
    python3 --version && \
    ln -s -f "$( which pip${LAW_IMAGE_PYTHON_MM} )" "$( dirname "$( which pip${LAW_IMAGE_PYTHON_MM} )" )/pip3" && \
    pip3 --version

# boost for system python 2.7
RUN yum -y install boost-devel boost-python; yum clean all

# boost for custom python
RUN wget -nv https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.gz && \
    tar -xzf boost_1_73_0.tar.gz && \
    cd boost_1_73_0 && \
    ./bootstrap.sh --with-libraries=system,thread,python --with-python="$( which python3 )" && \
    perl -pi -e 's!(\Qincludes ?= $(prefix)/include/python$(version)\E)!\1m!' tools/build/src/tools/python.jam && \
    ./b2 install --prefix=/usr && \
    ldconfig && \
    cd .. && \
    rm -rf boost_1_73_0 && \
    rm boost_1_73_0.tar.gz

# WLCG software
RUN rpm -Uvh https://repo.opensciencegrid.org/osg/3.5/osg-3.5-el7-release-latest.rpm
RUN yum-config-manager --add-repo http://linuxsoft.cern.ch/cern/centos/7/cern/x86_64
RUN yum -y update; yum clean all
RUN yum -y install CERN-CA-certs osg-ca-certs osg-voms voms-clients; yum clean all

# gfal2
RUN yum -y install gfal2-all gfal2-devel gfal2-util; yum clean all
RUN git clone https://gitlab.cern.ch/mrieger/gfal2-bindings.git && \
    cd gfal2-bindings && \
    git checkout 00fbad0f && \
    python3 setup.py install && \
    cd .. && \
    rm -rf gfal2-bindings
RUN git clone https://github.com/cern-it-sdc-id/gfal2-dropbox.git && \
    cd gfal2-dropbox && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf gfal2-dropbox

# micro editor
RUN cd /usr/bin; curl https://getmic.ro | bash

# python packages
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir --upgrade setuptools
RUN pip3 install --no-cache-dir slackclient
RUN pip3 install --no-cache-dir python-telegram-bot
RUN pip3 install --no-cache-dir flake8

# cleanup installation workdir
WORKDIR /root
RUN rm -rf install

# install law master
RUN git clone https://github.com/riga/law "$LAW_IMAGE_SOURCE_DIR" && \
    cd "$LAW_IMAGE_SOURCE_DIR" && \
    pip3 install --no-cache-dir .
WORKDIR $LAW_IMAGE_SOURCE_DIR

# shell initialization
RUN echo 'source "$( law completion )" ""' >> /root/.bash_profile

# init command
CMD bash --login
