# AdvNet Team 04 ANC

Current code only uses pgd adversarial attack. PGD iterations locked to 10. 

To run clean image training and inference for cifar10 dataset use: 
>python main.py --model <model_name> --save <save_folder_name> --dataset cifar10 --trpgd 0 --tspgd 0

To run adversarial image training and inference for cifar10 dataset use: 
>python main.py --model <model_name> --save <save_folder_name> --dataset cifar10 --trpgd 1 --tspgd 1

--trpgd = training pgd image flag
--tspgd = test pgd image flag
--save = creates a folder with provided name to store training/evaluation results
--model = DNN model: alexnet, resnet, vgg_cifar10

To run clean image inference with previously trained network: 
>python main.py --model <model_name> -e ./results/<save_folder_name>/model_best.pth.tar --dataset cifar10 --trpgd 0 --tspgd 0

To run adversarial image inference with previously trained network: 
>python main.py --model <model_name> -e ./results/<save_folder_name>/model_best.pth.tar --dataset cifar10 --trpgd 0 --tspgd 1
