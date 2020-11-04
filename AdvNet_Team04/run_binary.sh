prob_file=${1:-prob_0p6_0p3_linear_chip7.mat}

for i in {1..5}
do
python main_binary_noise.py -e results/vgg_cifar10_binarynoisetrain/model_best.pth.tar --common_prob 0 --bitwise_prob 1 --noise_inject True --model vgg_cifar10_binary --dataset cifar10 -b 2 -p 50 -pf /home/syin11/shared/pythonsim/XNORNET_SRAM/$prob_file
done
	