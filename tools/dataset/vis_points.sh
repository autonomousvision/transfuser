export CARLA_ROOT=${1:-/home/kchitta/Documents/CARLA_0.9.10.1}
export WORK_DIR=${2:-/home/kchitta/Documents/misc/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

python3 ${WORK_DIR}/tools/dataset/vis_points.py \
--in_path=${WORK_DIR}/leaderboard/data/training/routes/Scenario8/Town03_Scenario8.xml \
--save_dir=${WORK_DIR}/figures/vis_points/Scenario8/

python3 ${WORK_DIR}/tools/dataset/vis_points.py \
--in_path=${WORK_DIR}/leaderboard/data/training/scenarios/Scenario8/Town03_Scenario8.json \
--save_dir=${WORK_DIR}/figures/vis_points/Scenario8/

python3 ${WORK_DIR}/tools/dataset/vis_points.py \
--in_path=${WORK_DIR}/leaderboard/data/training/routes/lr/Town03_lr.xml \
--save_dir=${WORK_DIR}/figures/vis_points/lr/

python3 ${WORK_DIR}/tools/dataset/vis_points.py \
--in_path=${WORK_DIR}/leaderboard/data/longest6/longest6_split/longest_weathers_15.xml \
--save_dir=${WORK_DIR}/figures/vis_points/