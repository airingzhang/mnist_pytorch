# Toy experiment using CNN on MNIST dataset.
We use Pytorch as our Deep Learning library and our testing Environment be Centos 7.  
We showcase the common practice of DL pipeline. This idea may be shared across different libraries such as Tensorflow, Caffe, etc.
Specifically, in practice, major components of DL pipelines can be generally divided into 4 parts:
1. **dataset**, the input end (`mnist_dataset.py`). Datasets vary much. Therefore, specific behaviors of one dataset can be defined there.  
2. **model**, the algorithm (`net.py`). Strictly speaking, model should change along with dataset. Nevertheless, decoupling it from datasets is a good idea at the stage of researching good algorithms.  
3. **loss**, the output end (`loss.py`). The loss not only defines how we train the model but possibly indicates how we evaluate the performance.  
4. **main**, the whole pipeline (`main.py`), assemble together all the components above. Often people would define different behaviors for training and testing phase here.   


##  Major Dependency
* python 2.7 (anaconda is more preferable)
* PyTorch 0.4.0
* CUDA 8.0
* CUDNN 7.0 (Other CUDNN versions are also possible as long as they are compatible with CUDA 8.0)

##  Install Dependency
* PyTorch 0.4:  
	1. install anaconda which can be downloaded from `https://www.anaconda.com`  
	2. install PyTorch 0.4.0 `conda install pytorch=0.4.0 cuda80 -c pytorch` more details can be found `https://pytorch.org/get-started/previous-versions/`
* CUDA and CUDNN:  
	1. CUDA 8.0 can be found `https://developer.nvidia.com/cuda-80-ga2-download-archive`  
	2. CUDNN 7.0, follow the steps from 	`https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html`

	
## Code Arrangement
* model structure, loss and dataset are defined respectively in `net.py`, `loss.py` and `mnist_dataset.py`
* bash scripts `run_all.sh` cover the whole pipeline from data downloading, data preparation, initial training and full training
* `main.py` defines all training and testing behaviors

## Running
* full pipeline: `sh run_all.sh ${save_dir_for_dataset} ${save_dir_for_model}`  
* initial training: the original training set is split for training and validation set (default ratio is 0.1)
`python main.py --save-dir ${save_dir} --data-dir ${dataset_dir}`

* full training: with all training set being fed in
`python main.py --save-dir ${save_dir} --data-dir ${dataset_dir} --resume ${save_dir}/020.ckpt --train-full 1 --epochs 40`

* more setting details can be changed by adding more arguments. Check the details in `main.py`

* training logs and all `.py` files at the moment would be save to `${save_dir}` and `${save_dir}/code` 


## Experiment Logs
* No that much hyper-parameter tuning happens here. Most of the hyper-parameters were selected based on common practice.
 
* Step 1, initial training:  
	1. settings: training the models with 90% training set and 10% training set as the validation set  
	2. training models with 20 epochs  
	3. initial learning rate is 0.01 in first 2 epoch, then 0.001 from epoch 3 to 4, and 0.0001 afterwards  
	4. Training Loss: 0.034722. Validation Accuracy: 5910.0/6000 (98.50%)  
	
* Step 2, full training:  
	1. settings: all training set used for training  
	2. training models with extra 20 epochs on basis of the model already trained  
	3. initial learning rate is 0.0001
	4. Training Loss: 0.023376. Validation Accuracy: 9866.0/10000 (98.66%)



	 

