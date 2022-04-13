FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
WORKDIR /code
COPY requirements.txt /code/requirements.txt

#ARG NODE_ENV=development
ARG PORT=3000
ENV PORT=$PORT

#RUN python3 -m venv env
ENV VIRTUAL_ENV=/opt/env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#RUN  .env/bin/activate 
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get update && apt-get install -y python3-opencv
#RUN pip install opencv-python -y
#RUN apt-get install python3-opencv -y
RUN apt-get install python3-sqlalchemy -y

#RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html --user
RUN pip install -r requirements.txt




COPY ./app /code/app
COPY ./model_weights /code/model_weights

EXPOSE $PORT

#CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
#CMD web: uvicorn app.main:app --host=0.0.0.0 --port=${PORT}
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]