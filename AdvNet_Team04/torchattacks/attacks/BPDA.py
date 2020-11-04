import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from ..attack import Attack

def normalize(image, mean, std):
    return (image - mean)/std

def preprocess(image):
    image = image / 255
    image = np.transpose(image, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image = normalize(image, mean, std)
    return image

def image2tensor(image):
    img_t = torch.Tensor(image)
    img_t = img_t.unsqueeze(0)
    img_t.requires_grad_()
    return img_t

def label2tensor(label):
    target = np.array([label])
    # target = torch.from_numpy(target).long()
    target = torch.tensor(target)
    return target

def get_img_grad_given_label(image, label, model):
    logits = model(image)
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, target)
    loss.backward()
    ret = image.grad.clone()
    model.zero_grad()
    image.grad.data.zero_()
    return ret

def get_cw_grad(adv, origin, label, model):
    origin = torch.tensor(origin)
    model = model.cuda()
    label = label.cuda()
    logits = model(adv.cuda())
    ce = nn.CrossEntropyLoss()
    l2 = nn.MSELoss()
    loss = ce(logits, label) + l2(torch.tensor(0), torch.tensor(origin - adv)) / l2(torch.tensor(0), origin)
    loss.backward()
    ret = adv.grad.clone()
    model.zero_grad()
    adv.grad.data.zero_()
    # origin.grad.data.zero_()
    return ret

def l2_norm(adv, img):
    adv = adv.detach().numpy()
    img = img.detach().numpy()
    ret = np.sum(np.square(adv - img))/np.sum(np.square(img))
    return ret

def clip_bound(adv):
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    adv = adv * std + mean
    adv = np.clip(adv, 0., 1.)
    adv = (adv - mean) / std
    return adv.astype(np.float32)

def identity_transform(x):
    return x.detach().clone()




class BPDA(Attack):
    """
    PGD attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Arguments:
        model (nn.Module): a model to attack.
        eps (float): epsilon in the paper. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): max iterations. (DEFALUT : 40)
        
    """
    def __init__(self, model, step_size = 1., iters = 10, linf=False):
        super(BPDA, self).__init__("BPDA", model)
        self.step_size = step_size
        self.linf = linf
        self.iters = iters
        self.transform_func = identity_transform
    
    def forward(self, images, labels):
        targets = label2tensor(labels)
        targets = torch.tensor(targets)
        adv = images.detach().numpy()
        adv = torch.from_numpy(adv)
        adv.requires_grad_()
        for _ in range(self.iters):
            adv_def = adv.detach().clone()
            adv_def.requires_grad_()
            l2 = nn.MSELoss()
            loss = l2(torch.tensor(0), adv_def)
            loss.backward()
            g = get_cw_grad(adv_def, images, targets, self.model)
            if self.linf:
                g = torch.sign(g)
            # print(g.numpy().sum())
            adv = adv.detach().numpy() - self.step_size * g.numpy()
            adv = clip_bound(adv)
            adv = torch.from_numpy(adv)
            adv.requires_grad_()
            # if linf:
                # print('label', torch.argmax(model(adv)), 'linf', torch.max(torch.abs(adv - image)).detach().numpy())
            # else:
                # print('label', torch.argmax(model(adv)), 'l2', l2_norm(adv, image))

            
        return adv