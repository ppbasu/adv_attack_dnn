# AdvNet Team 04 ANC 

```                                                                                                                 
To create the conda environment                                                 
>conda create -n new environment --file req.txt                                 
```          
Once the git repository has been copied to your workspace, as shown in our Final Report paper, navigate to the AdvNet_Team04 folder since that is where all of the needed files are located.

Only two files will ever need to be changed to reproduce our results. These files include the main.py file and the resnet file.

In the main.py file, ensure that the attack you are wanting to investigate is listed on line 20. For example, if line 20 says the following: "from torchattacks import PGD, FGSM", that means that the PGD attack and FGSM attack are able to be used.

Line 320 is another part of the main.py file that will typically need to be changed. For example, if this line reads, "pgd_attack = PGD(model, eps=0.031, alpha=0.008, iters=pgditers)" then that means that the PGD attach will be implemented in the network. Switching "PGD" to say "FGSM" or "IFGSM" instead will apply one of those attacks to the network instead.

Once the necessary changes are made in the main.py file, save the changes and exit out of the file back to the command line. 

If you will be running either the ResNet-18 or the ResNet-50 network then the resnet.py file in the models folder may need to be changed.
```
To run clean image training and inference for cifar10 dataset use: 
>python main.py --model <model_name> --save <save_folder_name> --dataset cifar10 --trpgd 0 --tspgd 0
```
The options for <model_name> are resnet,


```
To run adversarial image training and inference for cifar10 dataset use: 
>python main.py --model <model_name> --save <save_folder_name> --dataset cifar10 --trpgd 1 --tspgd 1
```
```
--trpgd = training pgd image flag
--tspgd = test pgd image flag
--save = creates a folder with provided name to store training/evaluation results
--model = DNN model: alexnet, resnet, vgg_cifar10
```

```
To run clean image inference with previously trained network: 
>python main.py --model <model_name> -e ./results/<save_folder_name>/model_best.pth.tar --dataset cifar10 --trpgd 0 --tspgd 0
```

```
To run adversarial image inference with previously trained network: 
>python main.py --model <model_name> -e ./results/<save_folder_name>/model_best.pth.tar --dataset cifar10 --trpgd 0 --tspgd 1
```
