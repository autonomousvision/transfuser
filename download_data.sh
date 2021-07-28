#!/usr/bin/env bash

mkdir data
cd data

echo "This script downloads datasets used in the TransFuser project."
echo "Choose from the following options:"
echo "0 - Minimal Dataset (63G)"
echo "1 - Large Scale Dataset (406G)"
read -p "Enter dataset ID to download: " ds_id

if [ ${ds_id} == 0 ]
then
	# Download license file
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/LICENSE.txt

	# Download minimal version of 14_weathers_data
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/14_weathers_minimal_data.zip
	unzip 14_weathers_minimal_data.zip
	rm 14_weathers_minimal_data.zip

elif [ ${ds_id} == 1 ]
then
	# Download license file
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/LICENSE.txt

	# Download clear_weather_data
	mkdir clear_weather_data
	cd clear_weather_data
	for town in Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10
	do
		wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/clear_weather_data/${town}.zip
		unzip -q ${town}.zip
		rm ${town}.zip
	done

	cd ..

	# Download 14_weathers_data
	mkdir 14_weathers_data
	cd 14_weathers_data
	for town in Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10
	do
		wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/14_weathers_data/${town}.zip
		unzip -q ${town}.zip
		rm ${town}.zip
	done
else
	echo "Invalid ID!"
fi