# Adversarial-Attacks-Pytorch

This is a lightweight repository of adversarial attacks for Pytorch.

There are popular attack methods and some utils.

Here is a [documentation](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/index.html) for this package.



## Table of Contents
1. [Usage](#Usage)
2. [Attacks and Papers](#Attacks-and-Papers)
3. [Demos](#Demos)
4. [Frequently Asked Questions](#Frequently-Asked-Questions)
5. [Update Records](#Update-Records)
6. [Recommended Sites and Packages](#Recommended-Sites-and-Packages)

## Usage

### Dependencies

- torch 1.2.0
- python 3.6



### Installation

- `pip install torchattacks` or
- `git clone https://github.com/Harry24k/adversairal-attacks-pytorch`

```python
import torchattacks
pgd_attack = torchattacks.PGD(model, eps = 4/255, alpha = 8/255)
adversarial_images = pgd_attack(images, labels)
```



### Precautions

* **WARNING** :: All images should be scaled to [0, 1] with transform[to.Tensor()] before used in attacks.
* **WARNING** :: All models should return ONLY ONE vector of `(N, C)` where `C = number of classes`.




## Attacks and Papers

The papers and the methods with a brief summary and example.
All attacks in this repository are provided as *CLASS*.
If you want to get attacks built in *Function*, please refer below repositories.

* **Explaining and harnessing adversarial examples** : [Paper](https://arxiv.org/abs/1412.6572), [Repo](https://github.com/Harry24k/FGSM-pytorch)
  
  - FGSM
  
* **DeepFool: a simple and accurate method to fool deep neural networks** : [Paper](https://arxiv.org/abs/1511.04599)
  
  - DeepFool
  
* **Adversarial Examples in the Physical World** : [Paper](https://arxiv.org/abs/1607.02533), [Repo](https://github.com/Harry24k/AEPW-pytorch)
  - BIM or iterative-FSGM
  - StepLL

* **Towards Evaluating the Robustness of Neural Networks** : [Paper](https://arxiv.org/abs/1608.04644), [Repo](https://github.com/Harry24k/CW-pytorch)
  
  - CW(L2)
  
* **Ensemble Adversarial Traning : Attacks and Defences** : [Paper](https://arxiv.org/abs/1705.07204), [Repo](https://github.com/Harry24k/RFGSM-pytorch)
  
  - RFGSM
  
* **Towards Deep Learning Models Resistant to Adversarial Attacks** : [Paper](https://arxiv.org/abs/1706.06083), [Repo](https://github.com/Harry24k/PGD-pytorch)
  
  - PGD(Linf)
  
* **Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"** : [Paper](https://arxiv.org/abs/1907.00895)
  
  - APGD(EOT + PGD)
  
* **Fast is better than free: Revisiting adversarial training"** : [Paper](https://arxiv.org/abs/2001.03994)
  
  - FFGSM(Fast's FGSM)
  
* **Theoretically Principled Trade-off between Robustness and Accuracy"** : [Paper](https://arxiv.org/abs/1901.08573)
  - TPGD(TRADES' PGD)
  
  
Attack | Clean | Adversarial
:---: | :---: | :---:
FGSM | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/fgsm.png" width="300" height="300">
BIM | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/bim.png" width="300" height="300">
StepLL | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/stepll.png" width="300" height="300">
RFGSM | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/rfgsm.png" width="300" height="300">
CW | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/cw.png" width="300" height="300">
PGD(w/o random starts) | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/pgd.png" width="300" height="300">
PGD(w/ random starts) | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/rpgd.png" width="300" height="300">
DeepFool | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/clean.png" width="300" height="300"> | <img src="https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/pic/deepfool.png" width="300" height="300">



## Demos

* **White Box Attack with Imagenet** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20with%20Imagenet.ipynb)): 
To make adversarial examples with the Imagenet dataset to fool [Inception v3](https://arxiv.org/abs/1512.00567). However, the Imagenet dataset is too large, so only '[Giant Panda](http://www.image-net.org/)' is used.
* **Black Box Attack with CIFAR10** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
This demo provides an example of black box attack with two different models. First, make adversarial datasets from a holdout model with CIFAR10 and save it as torch dataset. Second, use the adversarial datasets to attack a target model.
* **Adversairal Training with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/Adversairal%20Training%20with%20MNIST.ipynb)): 
This code shows how to do adversarial training with this repository. The MNIST dataset and a custom model are used in this code. The adversarial training is performed with PGD, and then FGSM is applied to test the model.
* **Targeted PGD with Imagenet** ([code](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Targeted%20PGD%20with%20Imagenet.ipynb)): 
It shows we can perturb images to be classified into the labels we want with targeted PGD.
* **MultiAttack with MNIST** ([code](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/MultiAttack%20with%20MNIST.ipynb)): 
This code shows an example of PGD with N-random-restarts.




## Frequently Asked Questions

* **I want to use image normalization.** : In this case, you have to put normalize layer in the model. Please refer to [DEMO:White Box Attack with Imagenet](https://github.com/Harry24k/adversairal-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20with%20Imagenet.ipynb).
* **There is no randomize process in my model, but attacks return different results.** : [Some operations are non-deterministic with float tensors on GPU](https://discuss.pytorch.org/t/inconsistent-gradient-values-for-the-same-input/26179). If you want to get same results with same inputs, please run ["torch.backends.cudnn.deterministic = True".](https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch)



## Update Records

### ~Version 1.2 (DON'T USE)
* **Pip packages were corrupted by accumulating previous versions.**

### Version 1.3
* **Pip Package Re-uploaded.**

### Version 1.4
* **PGD** :
    * Now it supports targeted mode.
    
### Version 1.5
* **MultiAttack** :
    * MultiAttack added.
    * With it, you can use PGD with N-random-restarts or stronger attacks with different methods.
    
### Version 2.4 (Stable)
* **steps instead of iters** :
    * For compatibility reasons, all iters are changed to steps.
    
* **FFGSM** :
    * New FGSM proposed by [Eric Wong et al.](https://arxiv.org/abs/2001.03994) added.
   
* **TPGD** :
    * PGD(Linf) based on KL-Divergence loss proposed by [Hongyang Zhang et al.](https://arxiv.org/abs/1901.08573) added.
    

## Recommended Sites and Packages

* **Other Adversarial Attack Packages :**
    * [https://github.com/IBM/adversarial-robustness-toolbox](https://github.com/IBM/adversarial-robustness-toolbox) : Adversarial attack and defense package made by IBM. **TensorFlow, Keras, Pyotrch available.**
    * [https://github.com/bethgelab/foolbox](https://github.com/bethgelab/foolbox) : Adversarial attack package made by [Bethge Lab](http://bethgelab.org/). **TensorFlow, Pyotrch available.**
    * [https://github.com/tensorflow/cleverhans](https://github.com/tensorflow/cleverhans) : Adversarial attack package made by Google Brain. **TensorFlow available.**
    * [https://github.com/BorealisAI/advertorch](https://github.com/BorealisAI/advertorch) : Adversarial attack package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    * [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust) : Adversarial attack (especially on GNN) package made by [BorealisAI](https://www.borealisai.com/en/). **Pytorch available.**
    
* **Adversarial Defense Leaderboard :**
    * [https://github.com/MadryLab/mnist_challenge](https://github.com/MadryLab/mnist_challenge)
    * [https://github.com/MadryLab/cifar10_challenge](https://github.com/MadryLab/cifar10_challenge)
    * [https://www.robust-ml.org/](https://www.robust-ml.org/)
    * [https://robust.vision/benchmark/leaderboard/](https://robust.vision/benchmark/leaderboard/)
    
* **Adversarial Attack and Defense Papers:**
    * https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html : A Complete List of All (arXiv) Adversarial Example Papers made by Nicholas Carlini.
    * https://github.com/chawins/Adversarial-Examples-Reading-List : Adversarial Examples Reading List made by Chawin Sitawarin.