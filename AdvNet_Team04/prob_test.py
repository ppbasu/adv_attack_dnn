import argparse
import scipy.io as sio
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')


parser.add_argument('-pf', '--prob_file', type=str, default='/home/syin11/shared/pythonsim/XNORNET_SRAM/prob_0p6_0p3_linear_chip7.mat',
                    metavar='FILE', help='probability table mat file')
  
args = parser.parse_args()
  
fname = args.prob_file  
  
prob_data = sio.loadmat(args.prob_file)
probs = prob_data['prob']
levels = prob_data['levels']

print(levels)

fname = fname.replace('/home/syin11/shared/pythonsim/XNORNET_SRAM/', '')
fname = fname.replace('.mat', '')


ideal = 

print(probs.shape)

fig, ax = plt.subplots()
im = ax.imshow(probs)
ax.set_aspect(aspect=1/20)
ax.set_xlabel('levels')
ax.set_ylabel('bit-count')
# ax.set_xticks(range(-60,60,12))
# ax.set_yticks(range(-256,256,2))

# fig.colorbar(im, cax=ax, orientation='vertical')
fig.tight_layout()

plt.savefig(fname+'.png')
plt.close(fig)