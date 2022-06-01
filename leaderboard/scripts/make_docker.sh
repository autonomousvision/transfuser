#!/bin/bash
export CARLA_ROOT=/home/hiwi/CARLA
export SCENARIO_RUNNER_ROOT=/home/hiwi/challenge_submission_2021/scenario_runner
export LEADERBOARD_ROOT=/home/hiwi/challenge_submission_2021/leaderboard
export TEAM_CODE_ROOT=/home/hiwi/challenge_submission_2021/team_code_latest
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

if [ -z "$CARLA_ROOT" ]
then
    echo "Error $CARLA_ROOT is empty. Set \$CARLA_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$SCENARIO_RUNNER_ROOT" ]
then echo "Error $SCENARIO_RUNNER_ROOT is empty. Set \$SCENARIO_RUNNER_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$LEADERBOARD_ROOT" ]
then echo "Error $LEADERBOARD_ROOT is empty. Set \$LEADERBOARD_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$TEAM_CODE_ROOT" ]
then echo "Error $TEAM_CODE_ROOT is empty. Set \$TEAM_CODE_ROOT as an environment variable first."
    exit 1
fi

mkdir -p .tmp

cp -fr ${CARLA_ROOT}/PythonAPI  .tmp
mv .tmp/PythonAPI/carla/dist/carla*-py2*.egg .tmp/PythonAPI/carla/dist/carla-leaderboard-py2.7.egg
mv .tmp/PythonAPI/carla/dist/carla*-py3*.egg .tmp/PythonAPI/carla/dist/carla-leaderboard-py3x.egg

cp -fr ${SCENARIO_RUNNER_ROOT}/ .tmp
cp -fr ${LEADERBOARD_ROOT}/ .tmp
cp -fr ${TEAM_CODE_ROOT}/ .tmp/team_code

# build docker image
docker build --force-rm --network host -t transfuser-agent -f ${LEADERBOARD_ROOT}/scripts/Dockerfile.master .

rm -fr .tmp
