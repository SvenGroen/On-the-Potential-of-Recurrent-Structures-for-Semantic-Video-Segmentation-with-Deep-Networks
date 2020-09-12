#!/bin/bash
#$ -N env_test
#$ -l mem=4G
#$ -cwd
#$ -pe default 2

echo "Getting latest Miniconda version"
wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh


echo "Installing Miniconda"
bash Miniconda3-latest-Linux-x86_64.sh -b -p "miniconda" -u
rm Miniconda3-latest-Linux-x86_64.sh


echo "Create and activate eviroment torch"
source miniconda/bin/activate
conda create -y -q --name torch python=3.6
source activate torch


echo "Install requirements"
conda info --envs | grep '*'
pip install --upgrade -r requirements.txt


echo "Miniconda enviroment torch created, activate by:"
echo "source miniconda/bin/activate torch"
