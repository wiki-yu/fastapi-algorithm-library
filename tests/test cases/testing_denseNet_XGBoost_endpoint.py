from fastapi.testclient import TestClient
from fastapi import status,HTTPException, File, UploadFile
import glob
from pathlib import Path
from app.api.api_v1.endpoints.denseNet_XGBoost_endpoint import router



client = TestClient(router)

cur_path=os.getcwd()
cur_path=cur_path.replace('\\','/')


def test_densenet_upload_weights():
    url = "/uploadweightsdensenetXGB"
    filename=cur_path+'/model_weights/densenet_xgboost/densenet/denseXgB_model_myLayer.h5'
    filename_="denseXgB_model_myLayer.h5"
      
    up= {'file':(filename_, open(filename, 'rb'), 'application/octet-stream')}
    response=client.post(url,files=up)
    
    assert response.json()== {'filename': filename_}
    
 def test_xgboost_upload_weights():
    url = "/uploadweightsDNxgboost"
    filename=cur_path+'/model_weights/densenet_xgboost/xgboost/recognition_xgboost_prev_frames.joblib'
    filename_="recognition_xgboost_prev_frames.joblib"
      
    up= {'file':(filename_, open(filename, 'rb'), 'application/octet-stream')}
    response=client.post(url,files=up)
    
    assert response.json()== {'filename': filename_}   
    
    
    def test_upload_classes():
    url = "/uploadclasses"
    filename=cur_path+'/model_weights/densenet_xgboost/classes/classes.txt'
    filename_="classes.txt"
      
    up= {'file':(filename_, open(filename, 'rb'), 'application/octet-stream')}
    response=client.post(url,files=up)
    
    assert response.json()== {'filename': filename_}  
    
    
    
def test_densenet_xgboost_weights():
    response = client.get("/weights")
    assert response.status_code==200
    assert response.json()=={
        'denseNet': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                     for f in glob.glob("model_weights/densenet_xgboost/densenet/*.h5")],
        'XGBoost': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                    for f in glob.glob("model_weights/densenet_xgboost/xgboost/*.joblib")],
        'classNames': [Path(f).as_posix().rsplit('/', maxsplit=1)[-1].split('.')[0]
                       for f in glob.glob("model_weights/densenet_xgboost/classes/*.txt")]
    }
    
    
    
def test_densenet_xgboost_action_localization():
    # response = client.get("/actiondetection")
    url = "/actiondetection/"
    # ------------------ working ---------------
    multiple_files_ =[
    ('files', ('example1.png', open(cur_path+'/input/images/example1.png', 'rb'), 'image/png')),
    ('files', ('example2.png', open(cur_path+'/input/images/example2.png', 'rb'), 'image/png'))]
    weights_densenet="denseXgB_model_mylayer"
    weights_xgboost="recognition_xgboost_prev_frames"
    classNames="classes"
    data={
    "weights_densenet":weights_densenet,
    "weights_xgboost":weights_xgboost,
    "classNames":classNames,
    "save_upload_to_file":True}
    
    response=client.post(url,params=data,files=multiple_files_)
    result=response.json()
    print(result)
    assert response.json() ={
        "model_weights_rgb": weights_densenet,
        "model_weights_of": weights_xgboost,
        "classNames": classNames,
        "results_path": save_path
    }