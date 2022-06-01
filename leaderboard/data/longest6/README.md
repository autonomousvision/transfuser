# Longest6 benchmark

The Longest6 benchmark consists of 36 routes with an average route length of 1.5km, which is similar to the average route length of the official leaderboard (~1.7km). During evaluation, we ensure a high density of dynamic agents by spawning vehicles at every possible spawn point permitted by the CARLA simulator. Following the [NEAT evaluation benchmark](https://github.com/autonomousvision/neat/blob/main/leaderboard/data/evaluation_routes/eval_routes_weathers.xml), each route has a unique environmental condition obtained by combining one of 6 weather conditions (Cloudy, Wet, MidRain, WetCloudy, HardRain, SoftRain) with one of 6 daylight conditions (Night, Twilight, Dawn, Morning, Noon, Sunset).

To evaluate an agent on Longest6, please use the following environment variables in `leaderboard/scripts/local_evaluation.sh`:

```Shell
export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6.xml
```

We additionally provide a separate xml file for each of the 36 routes in the benchmark in the folder [longest6_split](./longest6_split/). To get the driving score of a single route, you can use the environment variables:

```Shell
export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6_split/longest_weathers_#.xml
```

| ![ Route](../../../figures/longest6/route00.png) | ![ Route](../../../figures/longest6/route01.png) | ![ Route](../../../figures/longest6/route02.png) | ![ Route](../../../figures/longest6/route03.png) | ![ Route](../../../figures/longest6/route04.png) | ![ Route](../../../figures/longest6/route05.png) |
|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| ![ Route](../../../figures/longest6/route06.png) | ![ Route](../../../figures/longest6/route07.png) | ![ Route](../../../figures/longest6/route08.png) | ![ Route](../../../figures/longest6/route09.png) | ![ Route](../../../figures/longest6/route10.png) | ![ Route](../../../figures/longest6/route11.png) |
| ![ Route](../../../figures/longest6/route12.png) | ![ Route](../../../figures/longest6/route13.png) | ![ Route](../../../figures/longest6/route14.png) | ![ Route](../../../figures/longest6/route15.png) | ![ Route](../../../figures/longest6/route16.png) | ![ Route](../../../figures/longest6/route17.png) |
| ![ Route](../../../figures/longest6/route18.png) | ![ Route](../../../figures/longest6/route19.png) | ![ Route](../../../figures/longest6/route20.png) | ![ Route](../../../figures/longest6/route21.png) | ![ Route](../../../figures/longest6/route22.png) | ![ Route](../../../figures/longest6/route23.png) |
| ![ Route](../../../figures/longest6/route24.png) | ![ Route](../../../figures/longest6/route25.png) | ![ Route](../../../figures/longest6/route26.png) | ![ Route](../../../figures/longest6/route27.png) | ![ Route](../../../figures/longest6/route28.png) | ![ Route](../../../figures/longest6/route29.png) |
| ![ Route](../../../figures/longest6/route30.png) | ![ Route](../../../figures/longest6/route31.png) | ![ Route](../../../figures/longest6/route32.png) | ![ Route](../../../figures/longest6/route33.png) | ![ Route](../../../figures/longest6/route34.png) | ![ Route](../../../figures/longest6/route35.png) |