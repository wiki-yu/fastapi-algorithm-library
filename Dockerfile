# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
FROM python:3.8
ENV PORT=$PORT

WORKDIR /code

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libgl1-mesa-dev

COPY ./requirements_cpu.txt /code/requirements_cpu.txt

# RUN pip install --no-cache-dir --upgrade -r requirements_cpu.txt
RUN pip install -r requirements_cpu.txt

COPY ./app /code/app 
COPY ./model_weights /code/model_weights


CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
# CMD uvicorn app.main:app --host 0.0.0.0 --port 80




