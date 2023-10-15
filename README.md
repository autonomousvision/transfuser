# TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving

## [Paper](http://www.cvlibs.net/publications/Chitta2022PAMI.pdf) | [Supplementary](http://www.cvlibs.net/publications/Chitta2022PAMI_supplementary.pdf) | [Talk](https://www.youtube.com/watch?v=-GMhYcxOiEU) | [Poster](http://www.cvlibs.net/publications/Chitta2022PAMI_poster.pdf) | [Slides](https://kashyap7x.github.io/assets/pdf/talks/Chitta2022AIR.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfuser-imitation-with-transformer-based/autonomous-driving-on-carla-leaderboard)](https://paperswithcode.com/sota/autonomous-driving-on-carla-leaderboard?p=transfuser-imitation-with-transformer-based)

<img src="figures/demo.gif">

This repository contains the code for the PAMI 2023 paper [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving](https://arxiv.org/abs/2205.15997). This work is a journal extension of the CVPR 2021 paper [Multi-Modal Fusion Transformer for End-to-End Autonomous Driving](https://arxiv.org/abs/2104.09224). The code for the CVPR 2021 paper is available in the [cvpr2021](https://github.com/autonomousvision/transfuser/tree/cvpr2021) branch.

If you find our code or papers useful, please cite:

```bibtex
@article{Chitta2023PAMI,
  author = {Chitta, Kashyap and
            Prakash, Aditya and
            Jaeger, Bernhard and
            Yu, Zehao and
            Renz, Katrin and
            Geiger, Andreas},
  title = {TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving},
  journal = {Pattern Analysis and Machine Intelligence (PAMI)},
  year = {2023},
}
```

```bibtex
@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and
            Chitta, Kashyap and
            Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
```

Also, check out the code for other recent work on CARLA from our group:
- [Jaeger et al., Hidden Biases of End-to-End Driving Models (ICCV 2023)](https://github.com/autonomousvision/carla_garage)
- [Renz et al., PlanT: Explainable Planning Transformers via Object-Level Representations (CoRL 2022)](https://github.com/autonomousvision/plant)
- [Hanselmann et al., KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients (ECCV 2022)](https://github.com/autonomousvision/king)
- [Chitta et al., NEAT: Neural Attention Fields for End-to-End Autonomous Driving (ICCV 2021)](https://github.com/autonomousvision/neat)

## Contents

1. [Setup](#setup)
2. [Dataset and Training](#dataset-and-training)
3. [Evaluation](#evaluation)


## Setup

Clone the repo, setup CARLA 0.9.10.1, and build the conda environment:

```Shell
git clone https://github.com/autonomousvision/transfuser.git
cd transfuser
git checkout 2022
chmod +x setup_carla.sh
./setup_carla.sh
conda env create -f environment.yml
conda activate tfuse
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html
```

## Dataset and Training
Our dataset is generated via a privileged agent which we call the autopilot (`/team_code_autopilot/autopilot.py`) in 8 CARLA towns using the routes and scenario files provided in [this folder](./leaderboard/data/training/). See the [tools/dataset](./tools/dataset) folder for detailed documentation regarding the training routes and scenarios. You can download the dataset (210GB) by running:

```Shell
chmod +x download_data.sh
./download_data.sh
```

The dataset is structured as follows:
```
- Scenario
    - Town
        - Route
            - rgb: camera images
            - depth: corresponding depth images
            - semantics: corresponding segmentation images
            - lidar: 3d point cloud in .npy format
            - topdown: topdown segmentation maps
            - label_raw: 3d bounding boxes for vehicles
            - measurements: contains ego-agent's position, velocity and other metadata
```

### Data generation
In addition to the dataset itself, we have provided the scripts for data generation with our autopilot agent. To generate data, the first step is to launch a CARLA server:

```Shell
./CarlaUE4.sh --world-port=2000 -opengl
```

For more information on running CARLA servers (e.g. on a machine without a display), see the [official documentation.](https://carla.readthedocs.io/en/stable/carla_headless/) Once the server is running, use the script below for generating training data:
```Shell
./leaderboard/scripts/datagen.sh <carla root> <working directory of this repo (*/transfuser/)>
```

The main variables to set for this script are `SCENARIOS` and `ROUTES`. 

### Training script

The code for training via imitation learning is provided in [train.py.](./team_code_transfuser/train.py) \
A minimal example of running the training script on a single machine:
```Shell
cd team_code_transfuser
python train.py --batch_size 10 --logdir /path/to/logdir --root_dir /path/to/dataset_root/ --parallel_training 0
```
The training script has many more useful features documented at the start of the main function. 
One of them is parallel training. 
The script has to be started differently when training on a multi-gpu node:
```Shell
cd team_code_transfuser
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d train.py --logdir /path/to/logdir --root_dir /path/to/dataset_root/ --parallel_training 1
```
Enumerate the GPUs you want to train on with CUDA_VISIBLE_DEVICES.
Set the variable OMP_NUM_THREADS to the number of cpus available on your system.
Set OPENBLAS_NUM_THREADS=1 if you want to avoid threads spawning other threads.
Set --nproc_per_node to the number of available GPUs on your node.

The evaluation agent file is build to evaluate models trained with multiple GPUs. 
If you want to evaluate a model trained with a single GPU you need to remove [this line](https://github.com/autonomousvision/transfuser/blob/a7d4db684c160095dec03851aff5ce92e36b2387/team_code_transfuser/submission_agent.py#LL95C18-L95C18).


## Evaluation

### Longest6 benchmark
We make some minor modifications to the CARLA leaderboard code for the Longest6 benchmark, which are documented [here](./leaderboard). See the [leaderboard/data/longest6](./leaderboard/data/longest6/) folder for a description of Longest6 and how to evaluate on it.

### Pretrained agents
Pre-trained agent files for all 4 methods can be downloaded from [AWS](https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip):

```Shell
mkdir model_ckpt
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip -P model_ckpt
unzip model_ckpt/models_2022.zip -d model_ckpt/
rm model_ckpt/models_2022.zip
```

### Running an agent
To evaluate a model, we first launch a CARLA server:

```Shell
./CarlaUE4.sh --world-port=2000 -opengl
```

Once the CARLA server is running, evaluate an agent with the script:
```Shell
./leaderboard/scripts/local_evaluation.sh <carla root> <working directory of this repo (*/transfuser/)>
```

By editing the arguments in `local_evaluation.sh`, we can benchmark performance on the Longest6 routes. You can evaluate both privileged agents (such as [autopilot.py]) and sensor-based models. To evaluate the sensor-based models use [submission_agent.py](./team_code_transfuser/submission_agent.py) as the `TEAM_AGENT` and point to the folder you downloaded the model weights into for the `TEAM_CONFIG`. The code is automatically configured to use the correct method based on the args.txt file in the model folder.

You can look at qualitative examples of the expected driving behavior of TransFuser on the Longest6 routes [here](https://www.youtube.com/watch?v=DZS-U3-iV0s&list=PL6LvknlY2HlQG3YQ2nMIx7WcnyzgK9meO).

### Parsing longest6 results
To compute additional statistics from the results of evaluation runs we provide a parser script [tools/result_parser.py](./tools/result_parser.py).

```Shell
${WORK_DIR}/tools/result_parser.py --xml ${WORK_DIR}/leaderboard/data/longest6/longest6.xml --results /path/to/folder/with/json_results/ --save_dir /path/to/output --town_maps ${WORK_DIR}/leaderboard/data/town_maps_xodr
```

It will generate a results.csv file containing the average results of the run as well as additional statistics. It also generates town maps and marks the locations where infractions occurred.

### Submitting to the CARLA leaderboard
To submit to the CARLA leaderboard you need docker installed on your system.
Edit the paths at the start of [make_docker.sh](./leaderboard/scripts/make_docker.sh).
Create the folder *team_code_transfuser/model_ckpt/transfuser*.
Copy the *model.pth* files and *args.txt* that you want to evaluate to *team_code_transfuser/model_ckpt/transfuser*.
If you want to evaluate an ensemble simply copy multiple .pth files into the folder, the code will load all of them and ensemble the predictions.

```Shell
cd leaderboard
cd scripts
./make_docker.sh
```
The script will create a docker image with the name transfuser-agent.
Follow the instructions on the [leaderboard](https://leaderboard.carla.org/submit/) to make an account and install alpha.

```Shell
alpha login
alpha benchmark:submit  --split 3 transfuser-agent:latest
```
The command will upload the docker image to the cloud and evaluate it.

<!-- ### Building docker image

Add the following paths to your ```~/.bashrc```
```
export CARLA_ROOT=<path_to_carla_root>
export SCENARIO_RUNNER_ROOT=<path_to_scenario_runner_in_this_repo>
export LEADERBOARD_ROOT=<path_to_leaderboard_in_this_repo>
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```

Edit the contents of ```leaderboard/scripts/Dockerfile.master``` to specify the required dependencies, agent code and model checkpoints. Add all the required information in the area delimited by the tags ```BEGINNING OF USER COMMANDS``` and ```END OF USER COMMANDS```. The current Dockerfile works for all the models in this repository.

Specify a name for the docker image in ```leaderboard/scripts/make_docker.sh``` and run:
```
leaderboard/scripts/make_docker.sh
```

Refer to the Transfuser example for the directory structure and where to include the code and checkpoints.

### Testing the docker image locally

Spin up a CARLA server:
```
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh -world-port=2000 -opengl
```

Run the docker container:  
Docker 19:  
```
docker run -it --rm --net=host --gpus '"device=0"' -e PORT=2000 <docker_image> ./leaderboard/scripts/run_evaluation.sh
```
If the docker container doesn't start properly, add another environment variable ```SDL_AUDIODRIVER=dsp```.

### Submitting docker image to the leaderboard

Register on [AlphaDriver](https://app.alphadrive.ai/), create a team and apply to the CARLA Leaderboard.

Install AlphaDrive cli:
```
curl http://dist.alphadrive.ai/install-ubuntu.sh | sh -
```

Login to alphadrive and submit the docker image:
```
alpha login
alpha benchmark:submit --split <2/3> <docker_image>
```
Use ```split 2``` for MAP track and ```split 3``` for SENSORS track. -->
