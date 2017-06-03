#/bin/sh

rm -f ./storage/stats/events.out.tfevents.*
python ./start.py
tensorboard --logdir=./storage/stats