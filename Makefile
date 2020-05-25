EXPERIMENT=v1_4

train:
	python deeparc.py --experiment_name=$(EXPERIMENT) --gpu=0

notebook:
	nohup jupyter notebook --no-browser --ip=0.0.0.0 &

pretrained:
	zip -o arc_pretrained.zip *.py

tensorboard:
	nohup tensorboard --host 0.0.0.0 --logdir ./experiments/$(EXPERIMENT) --samples_per_plugin images=10000 &
