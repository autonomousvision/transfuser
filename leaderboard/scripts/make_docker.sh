#!/bin/bash

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

mkdir .tmp

cp -fr ${CARLA_ROOT}/PythonAPI  .tmp
mv .tmp/PythonAPI/carla/dist/carla-*.egg .tmp/PythonAPI/carla/dist/carla-leaderboard.egg
cp -fr ${SCENARIO_RUNNER_ROOT}/ .tmp
cp -fr ${LEADERBOARD_ROOT}/ .tmp
cp -fr ${TEAM_CODE_ROOT}/ .tmp/team_code

# build docker image
docker build --force-rm -t leaderboard-user -f ${LEADERBOARD_ROOT}/scripts/Dockerfile.master .

rm -fr .tmp
