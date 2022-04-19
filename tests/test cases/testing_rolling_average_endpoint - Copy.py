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
    
    
    
def test_rolling_average_action_localization():
    url = "/actiondetection/"
    # ------------------ working ---------------
    multiple_files_ =[
    ('files', ('UCSP_fill_1_cut_compressed_mod.mp4', open(cur_path+'/input/videos/UCSP_fill_1_cut_compressed_mod.mp4', 'rb'), 'video/mp4'))]
    
    weights="vgg_tf_vgg_test_77"
    # data_={
    # "weights":"vgg_tf_vgg_test_77",
    # "save_upload_to_file":True,
    # "rollavg": True}
    
    data={
    "weights":weights,
    "save_upload_to_file":True,
    "rollavg": True}
    
    response=client.post(url,params=data,files=multiple_files_)
    result=response.json()
    # print(result)
    # print(result['model_weights'])
    # assert result['model_weights']=='vgg_tf_vgg_test_77'
    
    
  