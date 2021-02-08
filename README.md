# Cycle_PUGAN
Cycle_PUGAN(Cycle Point cloud Upsampling Generative Adversarial Network) by Wei-Cheng LIN, MMLAB.

#### Introduction 
This repository is for my undergraduate final project. The code is modified from [PUGAN-pytorch](https://github.com/UncleMEDM/PUGAN-pytorch) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 
#### Usage
1. Clone the repository:
    ```shell
    git clone https://github.com/stevenlin510/Cycle_PUGAN.git
    cd Cycle_PUGAN
    ```
2. Install dependencies and pointnet2 module:
    ```shell
    pip install -r requirements.txt
    cd pointnet2
    python setup.py install
    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
    ```
3. Train the model 
    ```shell
    cd train
    python train.py --exp_name=the_project_name --gpu=gpu_number --batch_size=batch_size
    ```

#### Evaluation
Install CGAL library first.
```shell
cd evaluation
cmake .
make
./evaluation <input>.off <pred>.xyz

```

#### Dataset
I use the PU-GAN dataset for training, you can refer to https://github.com/liruihui/PU-GAN to download the .h5 dataset file, which can be directly used in this project.
#### Note
Change opt['project_dir'] to where this project is located, and change opt['dataset_dir'] to where you store the dataset.




