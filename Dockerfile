ARG gpu_tag
FROM "krakentechnologies/tensorflow:1.12.0${gpu_tag}-py36"

ARG gpu_tag

RUN python -m pip install --no-cache-dir -U pip pipenv

RUN mkdir /opt/timeserio/
ADD . /opt/timeserio
WORKDIR /opt/timeserio

RUN pipenv install --system --deploy 
