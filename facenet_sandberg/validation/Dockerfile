# https://getintodevops.com/blog/the-simple-way-to-run-docker-in-docker-for-ci
FROM python:3.6-slim
RUN apt-get update && \
apt-get -y install apt-transport-https \
     ca-certificates \
     curl \
     gnupg2 \
     software-properties-common && \
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")\
/gpg > /tmp/dkey; apt-key add /tmp/dkey && \
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/\
$(. /etc/os-release; echo "$ID") $(lsb_release -cs) stable" && \
apt-get update && \
apt-get -y install docker-ce
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/validation/src
WORKDIR /app/validation/src
ENTRYPOINT ["python", "validate.py"]
