#!/bin/sh
# use --enc [50, 101, 152] for ResNet-[50, 101, 152] respectively
PYTHONPATH=$(pwd):$PYTHONPATH python src/test.py --enc 101