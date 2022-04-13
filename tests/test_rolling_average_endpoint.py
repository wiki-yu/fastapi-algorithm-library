from fastapi.testclient import TestClient
from fastapi import status,HTTPException, File, UploadFile
import glob
from pathlib import Path
from app.api.api_v1.endpoints.rolling_average_endpoint import router
import os

cur_path=os.getcwd()
cur_path=cur_path.replace('\\','/')

client = TestClient(router)

def test_rolling_average_upload_weights():
    url = "/uploadweights/"
    filename=cur_path+'/model_weights/rolling_average/vgg_tf_vgg_test_77.hdf5'
    filename_="vgg_tf_vgg_test_77.hdf5"
      
    up= {'file':(filename_, open(filename, 'rb'), 'application/octet-stream')}
    response=client.post(url,files=up)
    assert response.json()== {'filename': filename_}
    
    
    
    
    
def test_rolling_average_weights():
    response = client.get("/weights/")
    assert response.status_code==200
    assert response.json()==[Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
            for f in glob.glob("model_weights/rolling_average/*.hdf5")]
    
    
 
    
    
  