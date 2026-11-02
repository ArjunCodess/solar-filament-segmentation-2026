"""Binary Dice/focal and differentiable clDice with auxiliary supervision."""
import torch
import torch.nn.functional as F


def erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_skeleton(x, iterations=15):
    opened = F.max_pool2d(erode(x), 3, 1, 1)
    skeleton = F.relu(x-opened)
    for _ in range(iterations):
        x = erode(x)
        delta = F.relu(x-F.max_pool2d(erode(x), 3, 1, 1))
        skeleton = skeleton+F.relu(delta-skeleton*delta)
    return skeleton


def cldice_loss(probability, mask, skeleton, valid):
    pred_skeleton = soft_skeleton(probability)*valid
    precision = ((pred_skeleton*mask).sum((1, 2, 3))+1e-6)/(pred_skeleton.sum((1, 2, 3))+1e-6)
    recall = ((skeleton*probability*valid).sum((1, 2, 3))+1e-6)/((skeleton*valid).sum((1, 2, 3))+1e-6)
    value = 1-(2*precision*recall+1e-6)/(precision+recall+1e-6)
    # Background-only crops need false-positive suppression, not undefined topology.
    return torch.where(mask.sum((1, 2, 3)) > 0, value, (probability*valid).sum((1, 2, 3))/valid.sum((1, 2, 3)).clamp_min(1)).mean()


def composite_loss(logits, target, topology=True, auxiliary=True, focal_alpha=.75):
    mask, spine, boundary, skeleton, valid = [target[:, i:i+1] for i in range(5)]
    prob = logits[:, :1].sigmoid()
    dims = (1, 2, 3)
    dice = (1-(2*(prob*mask*valid).sum(dims)+1)/((prob*valid).sum(dims)+(mask*valid).sum(dims)+1)).mean()
    bce = F.binary_cross_entropy_with_logits(logits[:, :1], mask, reduction='none')
    pt = torch.where(mask > .5, prob, 1-prob)
    alpha = torch.where(mask > .5, focal_alpha, 1-focal_alpha)
    focal = (alpha*(1-pt).square()*bce*valid).sum()/valid.sum().clamp_min(1)
    cl = cldice_loss(prob, mask, skeleton, valid) if topology else logits.sum()*0
    loss = .4*dice+.3*focal+.3*cl if topology else .7*dice+.3*focal
    aux = logits.sum()*0
    if auxiliary:
        aux = ((F.binary_cross_entropy_with_logits(logits[:, 1:2], spine, reduction='none')+
                F.binary_cross_entropy_with_logits(logits[:, 2:3], boundary, reduction='none'))*valid).sum()/valid.sum().clamp_min(1)
        loss = loss+.1*aux
    return loss, {'dice_loss':float(dice.detach()), 'focal_loss':float(focal.detach()), 'cldice_loss':float(cl.detach()), 'auxiliary_loss':float(aux.detach())}
