EXPERIMENT=baseline

train:
	python deeparc.py --experiment_name=$(EXPERIMENT) --gpu=0

notebook:
	nohup jupyter notebook --no-browser --ip=0.0.0.0 &

pretrained:
	zip -o arc_pretrained.zip *.py
