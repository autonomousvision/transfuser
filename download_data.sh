#!/usr/bin/env bash

mkdir data
cd data

# Download license file
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/LICENSE.txt

# Download clear_weather data
mkdir clear_weather_data
cd clear_weather_data
for town in Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10
do
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/clear_weather_data/${town}.zip
	unzip -q ${town}.zip
	rm ${town}.zip
done

cd ..

# Download 14_weathers data
mkdir 14_weathers_data
cd 14_weathers_data
for town in Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10
do
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/14_weathers_data/${town}.zip
	unzip -q ${town}.zip
	rm ${town}.zip
done
