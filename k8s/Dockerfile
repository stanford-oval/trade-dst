ARG COMMON_IMAGE=
FROM ${COMMON_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

# copy source and install
COPY .. /home/trade-dst/trade-dst
WORKDIR /home/trade-dst/trade-dst
RUN pip3 install --user -r requirements.txt
