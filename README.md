# Robotools
Collection of simple scripts for robotic experiments

## Usage
This repository contains the numbered scripts
```
1_handeye_calibration.py
2_scene_reconstruction.py
3_scene_labelling.py
4_dataset_acquisition.py
```
Each of theses high-level scripts can be easily modified to suit your needs and are intended to be run sequentially in order to obtain a high-quality fully annotated dataset to evaluate your robotic system.


## Setup
For script 3, you need to use [6IMPOSE_legacy](https://github.com/LukasDb/6IMPOSE_legacy)
- Clone the repository to some location and follow the instructions in the README.md
- Run the docker container with the provided ./run.sh
- Then you can run script #3