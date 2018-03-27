----------
# Hashing-based Deep Neural Network
This project provides the coresponding code to the paper "Resisting Adversarial Examples using Hashing-based Deep Neural Networks in Malware Detection"

## Dependencies:
* Python 2.7
* Numpy 1.11.3
* Matplotlib 2.0.0
* Tensorflow 1.3 or 1.4
* Scikit-Learn 1.0.0
* Jupyter notebook

## Usage && Files Descriptions
In root direcotory:
* generate adversarial examples:
  gen-adv-smps.ipynb
* construct DNN graphs
  graphs.py
* Multi-index hashing based DNNs
  InH.ipynb
* Local forest hashing based DNNs
$\quad$ LFH.ipynb
* Joint index hashing and Denoising auto-encoder
$\quad$ JID.ipynb
* Joint locah forest hashing and Denoising auto-encoder
$\quad$ JFD.ipynb
* other files
$\quad$ utils.py learning_hashing_by_RF.py

We recommend to run *gen-adv-smps.ipynb* first to obain adversarial examples, and then perform any of *InH.ipynb*, *LFH.ipynb*, *JID.ipynb* and *JFD.ipynb*

In **drebin** directory:
There are source codes for a series expriments on drebin. We need apply the **DREBIN** dataset at here: [drebin]:https://www.sec.cs.tu-bs.de/~danarp/drebin/. 

In **pdfrate** directory:
We provide the robust support vector machine method.

## FAQ
We conduct the experiments on the CPU server which 64 cores CPU (2.4GHz) and shared 150G RAM.


