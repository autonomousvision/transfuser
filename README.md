# TransFuser

![TransFuser](assets/transfuser.png)

This repository contains the code for the CVPR 2021 paper [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](http://www.cvlibs.net/publications/Prakash2021CVPR.pdf). If you find out code or paper useful, please cite
```bibtex
@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and Chitta, Kashyap and Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
```

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
pip3 install -r requirements.txt
conda activate transfuser
```

Download and setup CARLA 0.9.10.1
```Shell
chmod +x setup_carla.sh
./setup_carla.sh
```

## Data Generation
The training data is generated using ```leaderboard/team_code/auto_pilot.py``` in 8 CARLA towns and 14 weather conditions. The routes and scenarios files to be used for data generation are provided at ```leaderboard/data```.

### Running CARLA Server

#### With Display
```Shell
./CarlaUE4.sh -world-port=<port> -opengl
```

#### Without Display

Without Docker:
```
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=<gpu_id> ./CarlaUE4.sh -world-port=<port> -opengl
```

With Docker:

Instructions for setting up docker are available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Pull the docker image of CARLA 0.9.10.1 ```docker pull carlasim/carla:0.9.10.1```.

Docker 18:
```
docker run -it --rm -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<gpu_id> carlasim/carla:0.9.10.1 ./CarlaUE4.sh -world-port=2000 -opengl
```

Docker 19:
```Shell
docker run -it --rm --net=host --gpus '"device=<gpu_id>"' carlasim/carla:0.9.10.1 ./CarlaUE4.sh -world-port=2000 -opengl
```

If the docker container doesn't start properly then add another environment variable ```-e SDL_AUDIODRIVER=dsp```.

### Run the Autopilot

Once the CARLA server is running, rollout the autopilot to start data generation.
```Shell
./leaderboard/scripts/run_evaluation.sh
```
The expert agent used for data generation is defined in ```leaderboard/team_code/auto_pilot.py```. Different variables which need to be set are specified in ```leaderboard/scripts/run_evaluation.sh```. The expert agent is based on the autopilot from [this codebase](https://github.com/bradyz/2020_CARLA_challenge).

## Training
The training code and pretrained models for different models used in our paper are provided below.
- [CILRS](cilrs)
- [LBC](https://github.com/bradyz/2020_CARLA_challenge)
- [AIM](aim)
- [Late Fusion](late_fusion)
- [Geometric Fusion](geometric_fusion)
- [TransFuser](transfuser)

## Evaluation
Spin up a CARLA server (described above) and run the required agent. The adequate routes and scenarios files are provided in ```leaderboard/data``` and the required variables need to be set in ```leaderboard/scripts/run_evaluation.sh```.
```Shell
./leaderboard/scripts/run_evaluation.sh
```

## Acknowledgements
This implementation uses code from several amazing repositories.
- [2020_CARLA_challenge](https://github.com/bradyz/2020_CARLA_challenge)
- [OATomobile](https://github.com/OATML/oatomobile)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)
