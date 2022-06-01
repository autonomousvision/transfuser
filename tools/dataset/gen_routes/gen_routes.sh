export CARLA_ROOT=${1:-/home/kchitta/Documents/CARLA_0.9.10.1}
export WORK_DIR=${2:-/home/kchitta/Documents/misc/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

python3 ${WORK_DIR}/tools/dataset/gen_routes/gen_routes_for_scen_1_3_4.py \
--save_dir=${WORK_DIR}/leaderboard/data/training/routes/ \
--scenarios_dir=${WORK_DIR}/leaderboard/data/training/scenarios/Scenario1/ \
--road_type=curved 

python3 ${WORK_DIR}/tools/dataset/gen_routes/gen_routes_for_scen_1_3_4.py \
--save_dir=${WORK_DIR}/leaderboard/data/training/routes/ \
--scenarios_dir=${WORK_DIR}/leaderboard/data/training/scenarios/Scenario3/ \
--road_type=curved 

python3 ${WORK_DIR}/tools/dataset/gen_routes/gen_routes_for_scen_1_3_4.py \
--save_dir=${WORK_DIR}/leaderboard/data/training/routes/ \
--scenarios_dir=${WORK_DIR}/leaderboard/data/training/scenarios/Scenario4/ \
--road_type=junction 

python3 ${WORK_DIR}/tools/dataset/gen_routes/gen_routes_for_scen_7_8_9.py \
--save_dir=${WORK_DIR}/leaderboard/data/training/routes/ \
--scenarios_dir=${WORK_DIR}/leaderboard/data/training/scenarios/

python3 ${WORK_DIR}/tools/dataset/gen_routes/gen_routes_for_scen_10.py \
--save_dir=${WORK_DIR}/leaderboard/data/training/routes/ \
--scenarios_dir=${WORK_DIR}/leaderboard/data/training/scenarios/Scenario10/

python3 ${WORK_DIR}/tools/dataset/gen_routes/gen_routes_lane_change.py \
--save_dir=${WORK_DIR}/leaderboard/data/training/routes/