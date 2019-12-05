ARG COMMON_IMAGE=
FROM ${COMMON_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

# install genie-toolkit
ARG GENIE_VERSION=master
RUN git clone https://github.com/stanford-oval/genie-toolkit /opt/genie-toolkit/
WORKDIR /opt/genie-toolkit/
RUN git checkout ${GENIE_VERSION}
RUN yarn install

# copy source and install
RUN mkdir -p /home/trade-dst/trade-dst
COPY --chown=trade-dst:trade-dst k8s /home/trade-dst/trade-dst/k8s
COPY --chown=trade-dst:trade-dst models /home/trade-dst/trade-dst/models
COPY --chown=trade-dst:trade-dst utils /home/trade-dst/trade-dst/utils
COPY --chown=trade-dst:trade-dst *.py README.md requirements.txt /home/trade-dst/trade-dst/
WORKDIR /home/trade-dst/trade-dst
RUN pip3 install --user -r requirements.txt
