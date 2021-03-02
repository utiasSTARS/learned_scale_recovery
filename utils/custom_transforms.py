from __future__ import division
import torch
import numpy as np
import random
import torch.utils.data
# from scipy.misc import imresize
import torchvision.transforms.functional as F

def get_data_transforms(config):
                ## specify transforms for data augmentation.  Only transform the training data and keep test and val data as-is

    data_transforms = {
        'train': Compose([
            RandomJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            PILtoNumpy(),
            RandomHorizontalFlip(p=0.5),
            ArrayToTensor(),
        ]),
        'val': Compose([
            RandomJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            PILtoNumpy(),
            # RandomHorizontalFlip(p=0.5),
            ArrayToTensor(),
        ]),
        'test': Compose([
            PILtoNumpy(),
            # RandomHorizontalFlip(p=1),
            ArrayToTensor(),
        ])
    }
    return data_transforms

class tofloatTensor(object): #transform that converts a numpy array to a FloatTensor
    def __init__(self):
        self.x=0
    def __call__(self, input):
        out = torch.FloatTensor(input)#.type(torch.LongTensor)
        return out  

    ###Visual Transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, original, transformed):
        for t in self.transforms:
            original, transformed = t(original, transformed)
        return original, transformed


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                
                t.sub_(m).div_(s)
        return images, intrinsics

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __call__(self, original, transformed):
        tensors = []
        tensors_transformed = []
        for im, im_transformed in zip(original[0], transformed[0]):
            if im.ndim == 2:
                im = im.reshape((1,im.shape[0],im.shape[1]))
                im_transformed = im_transformed.reshape((1,im_transformed.shape[0],im_transformed.shape[1]))
            else:
                im = np.transpose(im, (2, 0, 1))
                im_transformed = np.transpose(im_transformed, (2, 0, 1))
            tensors.append(torch.from_numpy(im).float()/255)
            tensors_transformed.append(torch.from_numpy(im_transformed).float()/255)
        return (tensors, original[1], original[2]), (tensors_transformed, transformed[1], transformed[2])
    
class PILtoNumpy(object):
    def __call__(self, original, transformed):
        np_imgs, np_imgs_transformed = [], []
        for im, im_transformed in zip(original[0], transformed[0]):
            np_imgs.append(np.array(im))
            np_imgs_transformed.append(np.array(im_transformed))

        return (np_imgs, original[1], original[2]), (np_imgs_transformed, transformed[1], transformed[2])
    
class RandomJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, original, transformed):
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        if random.random() > 0.5:
            output_imgs = [F.adjust_hue(F.adjust_saturation(F.adjust_contrast(F.adjust_brightness(im, \
                brightness_factor), contrast_factor), saturation_factor), hue_factor) for im in transformed[0]]
        else:
            output_imgs = transformed[0]
        return original, (output_imgs, transformed[1], transformed[2])    
    
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    Horizontal flip changes (makes negative) the angular velocity of the y and z axes (upward and forward)
    We do this for original and transformed, because flipping shouldn't change the photometric reprojection error"""
    def __init__(self, p=None):
        self.prob = p
    def __call__(self, original, transformed):
        if self.prob is not None:
            prob = np.random.uniform(low=0, high=1)
            if prob < self.prob:
                output_intrinsics = np.copy(original[1])
                w = original[0][0].shape[1]

                output_intrinsics[:,0,2] = w - output_intrinsics[:,0,2]
                output_images = [np.copy(np.fliplr(im)) for im in original[0]]

                transformed_output_intrinsics = np.copy(transformed[1])
                w = transformed[0][0].shape[1]
                transformed_output_intrinsics[:,0,2] = w - transformed_output_intrinsics[:,0,2]
                transformed_output_images = [np.copy(np.fliplr(im)) for im in transformed[0]]                
                for i in range(0,len(original[2])):
                    output_gt_lie_alg = np.copy(original[2][i][0])

                    output_gt_lie_alg[5] = -np.copy(output_gt_lie_alg[5]) #roll
                    output_gt_lie_alg[4] = -np.copy(output_gt_lie_alg[4]) #yaw
                    
                    output_vo_lie_alg = np.copy(original[2][i][1])
                    output_vo_lie_alg[5] = -np.copy(output_vo_lie_alg[5]) #roll
                    output_vo_lie_alg[4] = -np.copy(output_vo_lie_alg[4]) #yaw
                    original[2][i][0] = np.copy(output_gt_lie_alg)
                    original[2][i][1] = np.copy(output_vo_lie_alg)

                    transformed_output_gt_lie_alg = np.copy(transformed[2][i][0])
                    transformed_output_gt_lie_alg[5] = -np.copy(transformed_output_gt_lie_alg[5]) #roll
                    transformed_output_gt_lie_alg[4] = -np.copy(transformed_output_gt_lie_alg[4]) #yaw
                    
                    transformed_output_vo_lie_alg = np.copy(transformed[2][i][1])
                    transformed_output_vo_lie_alg[5] = -np.copy(transformed_output_vo_lie_alg[5]) #roll
                    transformed_output_vo_lie_alg[4] = -np.copy(transformed_output_vo_lie_alg[4]) #yaw
                    transformed[2][i][0] = np.copy(transformed_output_gt_lie_alg)
                    transformed[2][i][1] = np.copy(transformed_output_vo_lie_alg)


                return (output_images, output_intrinsics, original[2]), \
                    ( transformed_output_images, transformed_output_intrinsics, transformed[2] )
            else:
                return original, transformed

        else:
            return original, transformed

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before.
        TODO: check that target doesn't actually need to change based on the scaling/cropping"""

    def __call__(self, images, intrinsics, targets):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, output_intrinsics, targets

class Resize(object):
    def __init__(self, new_dim=(120,400)):
        self.new_dim=new_dim
    def __call__(self, images, intrinsics, targets):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)
        downscale_y = float(images[0].shape[0])/self.new_dim[0]
        downscale_x = float(images[0].shape[1])/self.new_dim[1]

        output_intrinsics[:,0] = intrinsics[:,0]/downscale_x
        output_intrinsics[:,1] = intrinsics[:,1]/downscale_y
        resized_imgs = [imresize(im, self.new_dim) for im in images]
        return resized_imgs, output_intrinsics, targets




