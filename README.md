# DS-AI Team Algorithm Library API

Official repository for the DS-AI Team Algorithm Library API deployed on Heroku with CICD pipeline.
Written in [Python3.8](https://www.python.org/) with [FastAPI](https://fastapi.tiangolo.com/).  



## Environment setup for manual installation

From project root, create virtual environment.

```bash
# Virtualenv modules installation (Unix based systems)
python -m venv env
source env/bin/activate

# Virtualenv modules installation (Windows based systems)
python -m venv env
.\env\Scripts\activate

# Virtualenv modules installation (Windows based systems if using bash)
python -m venv env
source ./env/Scripts/activate
```

If only using a CPU and no GPU is available, install requirements using:

```bash
pip3 install -r requirements_cpu.txt
```

---
---

For using NVIDIA GPUs, install needed dependancy versions here [Teams Shared Data](https://teams.microsoft.com/_#/files/IAI-AI?threadId=19%3A2887ad0aaac040a1b7ad4681f0b867be%40thread.tacv2&ctx=channel&context=NVIDIA%2520CUDA%2520Dependencies&rootfolder=%252Fsites%252FFiiUSA-iAIGroup-IAI-AI%252FShared%2520Documents%252FIAI-AI%252FAlgorithm%2520Library%2520API%252FNVIDIA%2520CUDA%2520Dependencies), or from official archive files here ([CUDA](https://developer.nvidia.com/cuda-zone) and [cuDNN](https://developer.nvidia.com/cudnn)) (working versions: CUDA verson 11.0.3 and cuDNN version 8.0.5 for CUDA 11.0), then install the CUDA version of PyTorch:

```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Next install remaining requirements:

```bash
pip3 install -r requirements.txt
```

---
---

### Download models here: [Teams Shared Data](https://teams.microsoft.com/_#/files/IAI-AI?threadId=19%3A2887ad0aaac040a1b7ad4681f0b867be%40thread.tacv2&ctx=channel&context=Algorithm%2520Library%2520API&rootfolder=%252Fsites%252FFiiUSA-iAIGroup-IAI-AI%252FShared%2520Documents%252FIAI-AI%252FAlgorithm%2520Library%2520API)

Place the DeepSORT checkpoint in

```bash
app/libs/deep_sort_pytorch/deep_sort/deep/checkpoint/
```

Place other various model weights in their respective folders

```bash
model_weights/<model name>/
```

---
Run API app.

```bash
uvicorn app.main:app
# Note: --reload flag restarts server after code or local files change.
#       Doesn't play nicely with uploads or local databases.
#       Remove for deployment.
```

By default,  
Backend running at [localhost:8000](http://127.0.0.1:8000)  
View docs at [localhost:8000/docs](http://127.0.0.1:8000/docs)

## Code-base structure

```bash
|-- Project Root/
   |-- .circleci
	  |-- config.yml
   |-- app/
      |-- api/
         |-- endpoints/
            |-- ...     # api endpoints
      |-- libs/
         |-- ...        # library files for app
      |-- schemas/
         |-- ...        # pydantic schema definitions
      |__init__.py
      |-- main.py
   |-- model_weights/   # model weights binary files
   |-- results/         # where model results are stored
   |-- uploads/         # where uploaded files are stored
   |-- tests/ 
	  |-- test_yolov5_endpoints.py
	  |-- test_rolling_average.py
   |-- requirements.txt
   |-- requirements_cpu.txt
   |-- Dockerfile
   |-- Procfile
   |-- setup.py
   |-- tox.ini
   |-- .gitattributes
   |-- .gitignore
```
