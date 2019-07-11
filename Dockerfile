ARG gpu_tag
FROM "octoenergy/tensorflow:1.13.1${gpu_tag}-py36"

ARG gpu_tag

RUN python -m pip install --no-cache-dir -U pip pipenv

RUN mkdir /opt/timeserio/
ADD . /opt/timeserio
WORKDIR /opt/timeserio

RUN pipenv install --system --deploy 
