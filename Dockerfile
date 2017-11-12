FROM python:3

WORKDIR /root
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt
ADD . facenet
CMD /bin/bash
