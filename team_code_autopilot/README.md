# Expert Driver (autopilot.py)

The rule-based expert algorithm used for generating training data is provided in `autopilot.py`. Its performance is an upper bound for the learning-based TransFuser agent. The autopilot has access to the complete state of the environment including vehicle and pedestrian locations and actions. The expert also has access to a dense set of waypoints along the route to be followed, terminating at the agent's destination.

The expert is described in Section 4.3 of the [paper](https://arxiv.org/abs/2205.15997). For additional details, check out [Bernhard Jaeger's Master Thesis](https://kait0.github.io/assets/pdf/master_thesis_bernhard_jaeger.pdf) which describes and analyzes various building blocks of the autopilot. The expert driver in this repository has some minor logical changes and additional hyper-parameter tuning compared to the expert from the thesis. 
