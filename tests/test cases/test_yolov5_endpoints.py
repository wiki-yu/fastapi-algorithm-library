from fastapi.testclient import TestClient
from fastapi import status,HTTPException, File, UploadFile
from app.api.api_v1.endpoints.yolov5_endpoints import router
import glob
from pathlib import Path
from collections import OrderedDict
import os

client = TestClient(router)

cur_path=os.getcwd()
cur_path=cur_path.replace('\\','/')

def test_yolov5_weights():
    response = client.get("/weights")
       
    #----- write for get method and test response
    assert response.status_code==200
    assert response.json()==[Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
            for f in glob.glob("model_weights/yolov5/*.pt")]
    



def test_upload_weights():
    url = "/uploadweights"
    filename=cur_path+'/model_weights/yolov5/yolov5s6.pt'
    filename_="yolov5s6.pt"
      
    up= {'file':(filename_, open(filename, 'rb'), 'application/octet-stream')}
    response=client.post(url,files=up)
    
    assert response.json()== {'filename': filename_}


