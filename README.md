## Training Pose Estimation Models for the App

### Setup
* Create python environment then install dependencies: `pip install -r requirements.txt`  
* Place COCO dataset in `data/` directory
  ```
   data
   ├── COCO
   │   ├── annotations
   │   ├── train2017
   │   └── val2017
   └── scripts
       ├── mpii2coco.py
       ├── posetrack2coco.py
       └── preprocess_coco.py
  ```
### Train
```
python -m pose_estimation.train 
```

### Acknowledgements
* Data conversion and dataset scripts from [TF-SimpleHumanPose](https://github.com/mks0601/TF-SimpleHumanPose)