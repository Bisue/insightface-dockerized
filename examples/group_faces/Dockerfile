# python 3.9.17 기반
FROM python:3.9.17

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

RUN pip install insightface
RUN pip install opencv-contrib-python
RUN pip install onnxruntime
RUN pip install "numpy<1.24.0"

WORKDIR /code

COPY . .

CMD ["python", "main.py"]
# CMD ["python", "-u", "main.py"]
