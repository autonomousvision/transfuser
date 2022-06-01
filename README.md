# Multi-Modal Fusion Transformer for End-to-End Autonomous Driving

### Update (06/2022): We have released an [extension to the original TransFuser CVPR paper on Arxiv](https://arxiv.org/abs/2205.15997). The code for this updated version will be released on the ["2022" branch](https://github.com/autonomousvision/transfuser/tree/2022).

## [Project Page](https://ap229997.github.io/projects/transfuser/) | [Paper](https://arxiv.org/pdf/2104.09224.pdf) | [Supplementary](http://www.cvlibs.net/publications/Prakash2021CVPR_supplementary.pdf) | [Video](https://youtu.be/WxadQyQ2gMs) | [Poster](https://ap229997.github.io/projects/transfuser/assets/poster.pdf) | [Blog](https://autonomousvision.github.io/transfuser)

<img src="transfuser/assets/teaser.svg" height="192" hspace=30> <img src="transfuser/assets/full_arch.svg" width="400">

This repository contains the code for the CVPR 2021 paper [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](http://www.cvlibs.net/publications/Prakash2021CVPR.pdf). If you find our code or paper useful, please cite
```bibtex
@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and Chitta, Kashyap and Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
```

## Contents
1. [Setup](#setup)
2. [Dataset](#dataset)
3. [Data Generation](#data-generation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [CARLA Leaderboard Submission](#carla-leaderboard-submission)
7. [Acknowledgements](#acknowledgements)

## Setup
Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment

```Shell
git clone https://github.com/autonomousvision/transfuser
cd transfuser
conda create -n transfuser python=3.7
conda activate transfuser
pip3 install -r requirements.txt
```

Download and setup CARLA 0.9.10.1
```Shell
chmod +x setup_carla.sh
./setup_carla.sh
```

## Dataset
The data is generated with ```leaderboard/team_code/auto_pilot.py``` in 8 CARLA towns using the routes and scenarios files provided at ```leaderboard/data``` on CARLA 0.9.10.1
```Shell
chmod +x download_data.sh
./download_data.sh
```

We used two datasets for different experimental settings:
- clear_weather_data: contains only `ClearNoon` weather. This dataset is used for the experiments described in the paper and generalization to new town results shown in the [video](https://youtu.be/WxadQyQ2gMs).
- 14_weathers_data: contains 14 preset weather conditions mentioned in ```leaderboard/team_code/auto_pilot.py```. This dataset is used for training models for the [leaderboard](https://leaderboard.carla.org/leaderboard) and the generalization to new weather results shown in the [video](https://youtu.be/WxadQyQ2gMs).

The dataset is structured as follows:
```
- TownX_{tiny,short,long}: corresponding to different towns and routes files
    - routes_X: contains data for an individual route
        - rgb_{front, left, right, rear}: multi-view camera images at 400x300 resolution
        - seg_{front, left, right, rear}: corresponding segmentation images
        - depth_{front, left, right, rear}: corresponding depth images
        - lidar: 3d point cloud in .npy format
        - topdown: topdown segmentation images required for training LBC
        - 2d_bbs_{front, left, right, rear}: 2d bounding boxes for different agents in the corresponding camera view
        - 3d_bbs: 3d bounding boxes for different agents
        - affordances: different types of affordances
        - measurements: contains ego-agent's position, velocity and other metadata
```

We have provided two versions of the datasets used in our work:
- Minimal dataset (63G): contains only `rgb_front`, `lidar` and `measurements` from the `14_weathers_data`. This is sufficient to train all the models (except LBC which also requires `topdown`).
- Large scale dataset (406G): contains multi-view camera data with different perception labels and affordances for both `clear_weather_data` and `14_weathers_data` to facilitate further development of imitation learning agents.

## Data Generation
In addition to the dataset, we have also provided all the scripts used for generating data and these can be modified as required for different CARLA versions.

### Running CARLA Server

#### With Display
```Shell
./CarlaUE4.sh --world-port=2000 -opengl
```

#### Without Display

Without Docker:
```
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh --world-port=2000 -opengl
```

With Docker:

Instructions for setting up docker are available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Pull the docker image of CARLA 0.9.10.1 ```docker pull carlasim/carla:0.9.10.1```.

Docker 18:
```
docker run -it --rm -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 -opengl
```

Docker 19:
```Shell
docker run -it --rm --net=host --gpus '"device=0"' carlasim/carla:0.9.10.1 ./CarlaUE4.sh --world-port=2000 -opengl
```

If the docker container doesn't start properly then add another environment variable ```-e SDL_AUDIODRIVER=dsp```.

### Run the Autopilot

Once the CARLA server is running, rollout the autopilot to start data generation.
```Shell
./leaderboard/scripts/run_evaluation.sh
```
The expert agent used for data generation is defined in ```leaderboard/team_code/auto_pilot.py```. Different variables which need to be set are specified in ```leaderboard/scripts/run_evaluation.sh```. The expert agent is based on the autopilot from [this codebase](https://github.com/bradyz/2020_CARLA_challenge).

### Routes and Scenarios

Each route is defined by a sequence of waypoints (and optionally a weather condition) that the agent needs to follow. Each scenario is defined by a trigger transform (location and orientation) and other actors present in that scenario (optional). The [leaderboard repository](https://github.com/carla-simulator/leaderboard/tree/master/data) provides a set of routes and scenarios files. To generate additional routes, spin up a CARLA server and follow the procedure below.

#### Generating routes with intersections
The position of traffic lights is used to localize intersections and (start_wp, end_wp) pairs are sampled in a grid centered at these points.
```Shell
python3 tools/generate_intersection_routes.py --save_file <path_of_generated_routes_file> --town <town_to_be_used>
```

#### Sampling individual junctions from a route
Each route in the provided routes file is interpolated into a dense sequence of waypoints and individual junctions are sampled from these based on change in navigational commands.
```Shell
python3 tools/sample_junctions.py --routes_file <xml_file_containing_routes> --save_file <path_of_generated_file>
```

#### Generating Scenarios
Additional scenarios are densely sampled in a grid centered at the locations from the [reference scenarios file](https://github.com/carla-simulator/leaderboard/blob/master/data/all_towns_traffic_scenarios_public.json). More scenario files can be found [here](https://github.com/carla-simulator/scenario_runner/tree/master/srunner/data).
```Shell
python3 tools/generate_scenarios.py --scenarios_file <scenarios_file_to_be_used_as_reference> --save_file <path_of_generated_json_file> --towns <town_to_be_used>
```

## Training
The training code and pretrained models are provided below.
```Shell
mkdir model_ckpt
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models.zip -P model_ckpt
unzip model_ckpt/models.zip -d model_ckpt/
rm model_ckpt/models.zip
```

Note that we have updated the pretrained TransFuser model with the improved checkpoint submitted to the leaderboard. This model contains multiple bug fixes and is trained on a different dataset than the one provided in this repository. (We are currently unable to share the entire dataset due to some issues.) 

- [CILRS](cilrs)
- [LBC](https://github.com/bradyz/2020_CARLA_challenge)
- [AIM](aim)
- [Late Fusion](late_fusion)
- [Geometric Fusion](geometric_fusion)
- [TransFuser](transfuser)

## Evaluation
Spin up a CARLA server (described above) and run the required agent. The adequate routes and scenarios files are provided in ```leaderboard/data``` and the required variables need to be set in ```leaderboard/scripts/run_evaluation.sh```.
```Shell
CUDA_VISIBLE_DEVICES=0 ./leaderboard/scripts/run_evaluation.sh
```

## CARLA Leaderboard Submission
CARLA also has an official [Autonomous Driving Leaderboard](https://leaderboard.carla.org/) on which different models can be evaluated. Refer to the [leaderboard_submission](https://github.com/autonomousvision/transfuser/tree/leaderboard_submission) branch in this repository for building docker image and submitting to the leaderboard.

## Acknowledgements
This implementation is based on code from several repositories.
- [2020_CARLA_challenge](https://github.com/bradyz/2020_CARLA_challenge)
- [OATomobile](https://github.com/OATML/oatomobile)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)

Also, check out other works on autonomous driving from our group.
- [Behl et al. - Label efficient visual abstractions for autonomous driving (IROS'20)](https://arxiv.org/pdf/2005.10091.pdf)
- [Ohn-Bar et al. - Learning Situational Driving (CVPR'20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ohn-Bar_Learning_Situational_Driving_CVPR_2020_paper.pdf)
- [Prakash et al. - Exploring Data Aggregation in Policy Learning for Vision-based Urban Autonomous Driving (CVPR'20)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prakash_Exploring_Data_Aggregation_in_Policy_Learning_for_Vision-Based_Urban_Autonomous_CVPR_2020_paper.pdf)
