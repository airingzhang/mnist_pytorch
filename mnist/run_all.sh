#!/bin/bash
dataset_dir=$1
save_dir=$2
cur_dir=$PWD

# Make directory for dataset and models
mkdir -p $dataset_dir
mkdir -p $save_dir

# Download MNIST
cd $dataset_dir
echo "Downloading Training set images"
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

echo "Downloading Training set labels"
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

echo "Downloading Testing set images"
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

echo "Downloading Testing set labels"
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz


# Extract files
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

# come back to code folder
cd $cur_dir

# initial training (the original training set is split for training and validation set, default ratio is 0.9)
python main.py --save-dir ${save_dir} --data-dir ${dataset_dir}

# full training (with all training set fed in)
python main.py --save-dir ${save_dir} --data-dir ${dataset_dir} --resume ${save_dir}/020.ckpt --train-full 1 --epochs 40



