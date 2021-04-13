import os

class GlobalConfig:
	# Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    root_dir = '/is/rg/avg/aprakash/carla9-10_data/opengl/all_towns_data'
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
    val_towns = ['Town05']
    train_data, val_data = [], []
    for town in train_towns:
        train_data.append(os.path.join(root_dir, town+'_tiny'))
        train_data.append(os.path.join(root_dir, town+'_short'))
    for town in val_towns:
        val_data.append(os.path.join(root_dir, town+'_short'))

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate

    # Encoder
    vert_anchors = 8
    horz_anchors = 8
    anchors = vert_anchors * horz_anchors
    n_embd = 512
    n_scale = 4

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
