#!/usr/bin/env bash

# Source bashrc
source $HOME/.bashrc

# Activate the environment
source activate deleutri

# Run the script
python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt