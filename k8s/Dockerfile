FROM docker.io/nvidia/cuda:10.1-runtime-ubi8

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

# install basic tools
RUN dnf -y install git wget gcc gcc-c++ make gettext unzip nodejs

# install yarn
RUN curl -sL https://dl.yarnpkg.com/rpm/yarn.repo | tee /etc/yum.repos.d/yarn.repo
RUN yum -y install nodejs yarn

# install python3
RUN yum -y install python3 \
	python3-numpy \
	python3-scipy \
	python3-pip \
	python3-devel

# install aws client
RUN pip3 install awscli

# download embeddings
RUN mkdir -p /usr/local/share/embeddings/trade-dst/glove && \
  wget https://parmesan.stanford.edu/glove/trade-dst/kazuma.db -O /usr/local/share/embeddings/trade-dst/kazuma.db && \
  wget https://parmesan.stanford.edu/glove/trade-dst/glove/common_crawl_840:300.db -O /usr/local/share/embeddings/trade-dst/glove/common_crawl_840:300.db
ENV EMBEDDINGS_ROOT=/usr/local/share/embeddings/trade-dst

# install some fixed deps
RUN pip3 install certifi==2019.3.9 \
    cffi==1.12.3 \
    chardet==3.0.4 \
    cuthon==0.5 \
    cycler==0.10.0 \
    Cython==0.29.10 \
    embeddings==0.0.6 \
    idna==2.8 \
    kiwisolver==1.1.0 \
    matplotlib==3.1.0 \
    numpy==1.16.4 \
    pycparser==2.19 \
    pyparsing==2.4.0 \
    python-dateutil==2.8.0 \
    requests==2.22.0 \
    six==1.12.0 \
    torch>=1.0.0 \
    tqdm==4.32.1 \
    urllib3==1.25.3 \
    transformers

# add user trade-dst
RUN useradd -ms /bin/bash -r trade-dst
USER trade-dst

# copy source and install
COPY .. /home/trade-dst/trade-dst
WORKDIR /home/trade-dst/trade-dst
RUN pip3 install --user -r requirements.txt
ENTRYPOINT ["python3"]