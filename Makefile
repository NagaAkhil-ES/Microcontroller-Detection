SHELL := /bin/bash

setup_env: requirements.txt
	python3 -m venv env
	
	source env/bin/activate; \
	pip install -r requirements.txt; \
	pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
	
	PYVER=$$(ls env/lib/ | grep python3 | head -n 1); \
	echo "$$(pwd)/src" > env/lib/$${PYVER}/site-packages/src.pth