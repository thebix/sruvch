#/bin/sh

rm -f ./storage/stats/events.out.tfevents.*
python ./guide4.py
tensorboard --logdir=./storage/stats