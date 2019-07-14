#!/bin/bash
epochs=12500

while [ $epochs -le 12500 ]
do
	echo /tmp/logs/SqueezeDet/train/model_$epochs.ckpt-$epochs
	# python ./src/demo.py --mode image --demo_net zynqDet --checkpoint /tmp/logs/SqueezeDet/train/model_$epochs.ckpt-$epochs
	python ./src/eval_det.py --eval_dir ./data/out/temp/ --adv
	((epochs = epochs + 500)) 
done