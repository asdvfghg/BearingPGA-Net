# BearingPGA-Net: A Lightweight and Deployable Bearing Fault Diagnosis Network via Decoupled Knowledge Distillation and FPGA Acceleration
## Overview
This is the repository of our submission for IEEE Transactions on Instrumentation and Measurement   "BearingPGA-Net: A Lightweight and Deployable Bearing Fault Diagnosis Network via Decoupled Knowledge Distillation and FPGA Acceleration". Our preprint version will be coming soon.
In this work,

* We build BearingPGA-Net, a lightweight neural network tailored for bearing fault diagnosis. This network is characterized by a single convolutional layer and is enhanced using decoupled knowledge distillation.

* We employ dynamic fixed-point quantization to compress the parameters of BearingPGA-Net by 50\% and propose a CNN accelerators scheme, where we utilize parallel computing and module multiplexing techniques to fully leverage the computational resources of the Kintex-7 FPGA.


* Compared to lightweight competitors, our proposed method demonstrates exceptional performance in noisy environments, achieving an average F1 score of over 98\% on CWRU datasets. Moreover, it offers a smaller model size, occupying only 2.83K parameters. Our FPGA deployment solution not only minimizes performance loss and maximizes speed but also is translatable to other FPGA boards..



![BearingPGA-Net](https://raw.githubusercontent.com/asdvfghg/image/master/小书匠/1690447137832.png)


[^_^]:
	## Citing
	If you find this repo useful for your research, please consider citing it:
	```
	@article{liao2022attention,
	  title={Attention-embedded Quadratic Network (Qttention) for Effective and Interpretable Bearing Fault Diagnosis},
	  author={Liao, Jing-Xiao and Dong, Hang-Cheng and Sun, Zhi-Qi and Sun, Jinwei and Zhang, Shiping and Fan, Feng-Lei},
	  journal={arXiv preprint arXiv:2206.00390},
	  year={2022}
	}
	```



## Repository organization

### Requirements
#### 1. Python environments
Our experiments are conducted in Windows 10 with Intel(R) Core(TM) 11th Gen i5-1135G7 at 2.4GHz CPU and one NVIDIA RTX 3080 8GB GPU.

We use PyCharm 2023.1 to be a coding IDE, if you use the same, you can run this program directly. Other IDE we have not yet tested, maybe you need to change some settings.
* Python == 3.10
* PyTorch == 2.1.0
* CUDA == 11.8 if use GPU
* wandb == 0.15.4
* anaconda == 2021.05
 #### 2. FPGA environments
 The FPGA chip we used is  Kintex-7 XC7K325T, the following software are required for parameters quantization and FPGA deployment.
 * Matlab for converting number formats.
 * CodeBlocks or other C++ IDE for convert complement number to original number.
 * vivado2018 for FPGA RTL simulation and evaluation.
### Organization
```
BearingPGA-Net
│   train_dkd.py # training BearingPGA-Net via decoupled knowledge distillation
│   train_single.py training BearingPGA-Net without KD
│   inference_only.py # test bearing data 
│	 weight_convert.py # convert weight parameters to .txt
│   fixed_point_quantization.m # convert 32-bit float to 16-bit fixed-point format
│   convert_complement_number_to_original.cpp # convert complement number to original number for deployment to FPGA
└─  data # bearing fault datasets 
     │   
     └─ 0HP # example dataset
└─  utils
     │   data_split.py # spliting dataset to train set and test set, then add noise to the raw signal. 
     │   DatasetLoader.py # dataloder class for pytorch
     │	  loss_fun.py # Loss function of DKD and KD
     │   Preprocess.py # preprocessing signal
└─  Model
     │   Student.py # student model
     │   Teacher.py # teacher model
└─  Pth # Model saving folder
└─  Weight_Parameters # Store float format and fixed-point format parameters
└─  1D_CONV_RTL # Verilog code of all layers in BearingPGA-Net
```

### Datasets
We use the CWRU dataset and HIT dataset in our article. The CWRU dataset is a public bearing fault dataset  that can be found in [CWRU Dataset](https://github.com/s-whynot/CWRU-dataset).

### How to Use
#### 1. Training BearingPGA-Net on Pytorch

Our deep learning models use **Weight & Bias** for training and fine-tuning. It is the machine learning platform for developers to build better models faster. [Here](https://docs.wandb.ai/quickstart) is a quick start for Weight & Bias. You need to create an account and install the CLI and Python library for interacting with the Weights and Biases API:
```
pip install wandb
```
Then login 
```
wandb login
```

After that, you can run our code for training.

1. For noisy data, you need to generate train dataset and test dataset by running ```data_split.py```.

2. Run ```train_dkd.py``` to train a BearingPGA-Net. Notably, you need to fill in your username in the wandb initialization function:
 ```
wandb.init(project="DKD", entity="username")
```
#### 2. Deploying on FPGA

Before deploying to the FPGA, the parameters of our model needs to be quantized.

1. Run ```weight_convert.py```  to convert weight parameters .pth to .txt. The .txt file will be saved to this folder:  ```Weight_Parameters/Float```.
2. Run ```fixed_point_quantization.m ```  using Matlab to convert 32-bit float to 16-bit fixed-point format. The .txt file will be saved to this folder:  ```Weight_Parameters/Fixed_Point```.
3. Run ```convert_complement_number_to_original.cpp``` converting complement number to original number. Notably, you need to modify the file path of this code to convert the parameters for each layer. Fortunately, we only have 4 layers of parameters need to be converted.

```
ifstream infile("./Weight_Parameters/Fixed_Point/scnn_layer_0.txt");
ofstream outfile("./Weight_Parameters/Fixed_Point/scnn_layer_0_new.txt");
```

 4. Because the paramters in .txt file cannot directly use in Verilog, the format has to be transformed again. However, it is quite simple. All you need to do is add these two lines to the file header and change the file extension to .coe:
```
memory_initialization_radix=2;
memory_initialization_vector=
```
5. Due to the entire FPGA project is too large, we only provide our Verilog code.  The entire process, including register transfer level (RTL) design, simulation and verification, synthesis, implementation (place and route, P\&R), editing timing constraints, generation of bitstream files, and the FPGA configuration, has to be done in vivado. **Notably, you need to change the bit-width in Verilog for different model.**
- FFT module: ```shiftFunction.v```line5 - SHIFT_BIT=*decimal bit-width*
- Convolution layer: ```convUnit.v```line21 - RIGHT_SHIFT_BIT(*decimal bit-width-1*)
- FC layer:```layer.v```line8 - RIGHT_SHIFT_BIT=*decimal bit-width*
## Contact
If you have any questions , please contact the following email address:

jingxiaoliao[at]hit[dot]edu[dot]cn

Enjoy your coding!

[^_^]:
	## Reference

