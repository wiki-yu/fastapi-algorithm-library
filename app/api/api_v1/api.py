from fastapi import APIRouter

from app.api.api_v1.endpoints import (deepsort_endpoints,
                                      yolov5_endpoints,
                                      c3d_endpoints,
                                      rolling_average_endpoint,
                                      two_stream_endpoints,
                                      denseNet_XGBoost_endpoint,
                                      spatio_temporal_endpoint,
                                      )

api_router = APIRouter()

api_router.include_router(yolov5_endpoints.router, prefix="/yolov5", tags=["YOLOv5"])
api_router.include_router(deepsort_endpoints.router, prefix="/deepsort", tags=["DeepSORT"])
api_router.include_router(c3d_endpoints.router, prefix="/c3d", tags=["C3D"])
api_router.include_router(rolling_average_endpoint.router, prefix="/rollavg", tags=["Rolling Average"])
api_router.include_router(denseNet_XGBoost_endpoint.router, prefix="/denenetXgboost", tags=["DenseNet XGBoost"])
api_router.include_router(two_stream_endpoints.router, prefix="/two_stream", tags=["Two Stream"])
api_router.include_router(spatio_temporal_endpoint.router, prefix="/spatio_temporal", tags=["Spatio Temporal"])

# from app.api.api_v1.endpoints import (yolov5_endpoints)
# api_router = APIRouter()
# api_router.include_router(yolov5_endpoints.router, prefix="/yolov5", tags=["YOLOv5"])

