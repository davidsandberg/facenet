FROM tensorflow/tensorflow:1.10.1-py3

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY align.py .
COPY model.pb .

ENTRYPOINT ["python3", "align.py"]