#!/bin/bash

sudo yum install -y graphviz-devel.x86_64 htop
source activate pytorch_p36
pip install pygraphviz sklearn-hierarchical-classification --user