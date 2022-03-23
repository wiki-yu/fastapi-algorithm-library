from typing import Optional, List
from pydantic import BaseModel


class Yolov5InputData(BaseModel):
    """Shared properties while creating or reading data"""
    data: Optional[str] = None


class Yolov5Response(BaseModel):
    """Properties to return to client"""
    model_weights: str
    filenames: List[str]
    results_format: str
    results: List[dict]
    results_by_detection: List[dict]


yolov5_response_examples = {
    200: {
        "description": "Success",
        "content": {
            "application/json": {
                "examples": {
                    "xyxy": {
                        "summary": "xyxy detection format",
                        "value": {
                            "model_weights": "yolov5s",
                            "filenames": [
                                "test_image.jpg"
                            ],
                            "results_format": "xyxy",
                            "results": [
                                {
                                    "xmin": {
                                        "0": 90,
                                        "1": 58,
                                    },
                                    "ymin": {
                                        "0": 8,
                                        "1": 3,
                                    },
                                    "xmax": {
                                        "0": 220,
                                        "1": 103,
                                    },
                                    "ymax": {
                                        "0": 358,
                                        "1": 104,
                                    },
                                    "confidence": {
                                        "0": 0.25740930438041687,
                                        "1": 0.2785199284553528,
                                    },
                                    "class": {
                                        "0": 0,
                                        "1": 0,
                                    },
                                    "name": {
                                        "0": "person",
                                        "1": "person",
                                    }
                                }
                            ],
                            "results_by_detection": [
                                {
                                    "0": {
                                        "xmin": 90,
                                        "ymin": 8,
                                        "xmax": 220,
                                        "ymax": 358,
                                        "confidence": 0.25740930438041687,
                                        "class": 0,
                                        "name": "person"
                                    },
                                    "1": {
                                        "xmin": 58,
                                        "ymin": 3,
                                        "xmax": 103,
                                        "ymax": 104,
                                        "confidence": 0.2785199284553528,
                                        "class": 0,
                                        "name": "person"
                                    }
                                }
                            ]
                        }
                    },
                    "xywhn": {
                        "summary": "normalized xywh detection format",
                        "value": {
                            "model_weights": "yolov5s",
                            "filenames": [
                                "test_image.jpg"
                            ],
                            "results_format": "xywhn",
                            "results": [
                                {
                                    "xcenter": {
                                        "0": 0.1550000011920929,
                                        "1": 0.08049999922513962
                                    },
                                    "ycenter": {
                                        "0": 0.2747747600078583,
                                        "1": 0.08033032715320587
                                    },
                                    "width": {
                                        "0": 0.12999999523162842,
                                        "1": 0.04500000178813934
                                    },
                                    "height": {
                                        "0": 0.5255255103111267,
                                        "1": 0.15165165066719055
                                    },
                                    "confidence": {
                                        "0": 0.25740930438041687,
                                        "1": 0.2785199284553528
                                    },
                                    "class": {
                                        "0": 0,
                                        "1": 0
                                    },
                                    "name": {
                                        "0": "person",
                                        "1": "person"
                                    }
                                }
                            ],
                            "results_by_detection": [
                                {
                                    "0": {
                                        "xcenter": 0.1550000011920929,
                                        "ycenter": 0.2747747600078583,
                                        "width": 0.12999999523162842,
                                        "height": 0.5255255103111267,
                                        "confidence": 0.25740930438041687,
                                        "class": 0,
                                        "name": "person"
                                    },
                                    "1": {
                                        "xcenter": 0.08049999922513962,
                                        "ycenter": 0.08033032715320587,
                                        "width": 0.04500000178813934,
                                        "height": 0.15165165066719055,
                                        "confidence": 0.2785199284553528,
                                        "class": 0,
                                        "name": "person"
                                    }
                                }
                            ]
                        }
                    },
                    "multiple_xyxy": {
                        "summary": "Multiple images, xyxy format",
                        "value": {
                            "model_weights": "yolov5s",
                            "filenames": [
                                "test_image1.jpg",
                                "test_image2.jpg"
                            ],
                            "results_format": "xyxy",
                            "results": [
                                {
                                    "xmin": {
                                        "0": 90,
                                        "1": 58
                                    },
                                    "ymin": {
                                        "0": 8,
                                        "1": 3
                                    },
                                    "xmax": {
                                        "0": 220,
                                        "1": 103
                                    },
                                    "ymax": {
                                        "0": 358,
                                        "1": 104
                                    },
                                    "confidence": {
                                        "0": 0.25740930438041687,
                                        "1": 0.2785199284553528
                                    },
                                    "class": {
                                        "0": 0,
                                        "1": 0
                                    },
                                    "name": {
                                        "0": "person",
                                        "1": "person"
                                    }
                                },
                                {
                                    "xmin": {
                                        "0": 1071,
                                        "1": 723,
                                        "2": 824,
                                        "3": 1362,
                                        "4": 960
                                    },
                                    "ymin": {
                                        "0": 643,
                                        "1": 191,
                                        "2": 338,
                                        "3": 1,
                                        "4": 154
                                    },
                                    "xmax": {
                                        "0": 1151,
                                        "1": 800,
                                        "2": 882,
                                        "3": 1409,
                                        "4": 1030
                                    },
                                    "ymax": {
                                        "0": 802,
                                        "1": 379,
                                        "2": 500,
                                        "3": 87,
                                        "4": 318
                                    },
                                    "confidence": {
                                        "0": 0.26812422275543213,
                                        "1": 0.2944554388523102,
                                        "2": 0.3132147192955017,
                                        "3": 0.31873270869255066,
                                        "4": 0.3249199688434601
                                    },
                                    "class": {
                                        "0": 0,
                                        "1": 0,
                                        "2": 0,
                                        "3": 0,
                                        "4": 0
                                    },
                                    "name": {
                                        "0": "person",
                                        "1": "person",
                                        "2": "person",
                                        "3": "person",
                                        "4": "person"
                                    }
                                }
                            ],
                            "results_by_detection": [
                                {
                                    "0": {
                                        "xmin": 90,
                                        "ymin": 8,
                                        "xmax": 220,
                                        "ymax": 358,
                                        "confidence": 0.25740930438041687,
                                        "class": 0,
                                        "name": "person"
                                    },
                                    "1": {
                                        "xmin": 58,
                                        "ymin": 3,
                                        "xmax": 103,
                                        "ymax": 104,
                                        "confidence": 0.2785199284553528,
                                        "class": 0,
                                        "name": "person"
                                    }
                                },
                                {
                                    "0": {
                                        "xmin": 1071,
                                        "ymin": 643,
                                        "xmax": 1151,
                                        "ymax": 802,
                                        "confidence": 0.26812422275543213,
                                        "class": 0,
                                        "name": "person"
                                    },
                                    "1": {
                                        "xmin": 723,
                                        "ymin": 191,
                                        "xmax": 800,
                                        "ymax": 379,
                                        "confidence": 0.2944554388523102,
                                        "class": 0,
                                        "name": "person"
                                    },
                                    "2": {
                                        "xmin": 824,
                                        "ymin": 338,
                                        "xmax": 882,
                                        "ymax": 500,
                                        "confidence": 0.3132147192955017,
                                        "class": 0,
                                        "name": "person"
                                    },
                                    "3": {
                                        "xmin": 1362,
                                        "ymin": 1,
                                        "xmax": 1409,
                                        "ymax": 87,
                                        "confidence": 0.31873270869255066,
                                        "class": 0,
                                        "name": "person"
                                    },
                                    "4": {
                                        "xmin": 960,
                                        "ymin": 154,
                                        "xmax": 1030,
                                        "ymax": 318,
                                        "confidence": 0.3249199688434601,
                                        "class": 0,
                                        "name": "person"
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    },
    415: {
        "description": "Unsupported Media Type",
        "content": {
            "application/json": {
                "examples": {
                    "Invalid image file format": {
                        "value": {
                            "detail": "Please upload only .jpeg, .png, or .mp4 files."
                        }
                    }
                }
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "examples": {
                    "Invalid detection format": {
                        "value": {
                            "detail": "detection_format must be 'xyxy', 'xyxyn', 'xywh', or 'xywhn'."
                        }
                    },
                    "Invalid model weights file": {
                        "value": {
                            "detail": "Model weights not available."
                        }
                    }
                }
            }
        }
    },
}

yolov5getweights_response_examples = {
    200: {
        "description": "Success",
        "content": {
            "application/json": {
                "examples": {
                    "Standard YOLOv5 model weights": {
                        "summary": "Standard YOLOv5 model weights",
                        "value":
                            [
                                "yolov5l",
                                "yolov5l6",
                                "yolov5m",
                                "yolov5m6",
                                "yolov5s",
                                "yolov5s6",
                                "yolov5x",
                                "yolov5x6"
                            ]
                    }
                }
            }
        }
    },
}

yolov5uploadweights_response_examples = {
    200: {
        "description": "Success",
        "content": {
            "application/json": {
                "examples": {
                    "Success": {
                        "summary": "Success",
                        "value": {
                            "filename": "new_yolo_weights.pt"
                        }
                    }
                }
            }
        }
    },
    415: {
        "description": "Unsupported Media Type",
        "content": {
            "application/json": {
                "examples": {
                    "Invalid image file format": {
                        "value": {
                            "detail": "Please upload only .pt files."
                        }
                    }
                }
            }
        }
    },
}
