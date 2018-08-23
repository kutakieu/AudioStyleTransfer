import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import argparse
import numpy as np
from PIL import Image

from util import load_image, convertRGB2BGR

GG_MEAN = [103.939, 116.779, 123.68]

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, kernel, bias):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.conv.weight = torch.nn.Parameter(torch.tensor(kernel.transpose((3,2,0,1))), requires_grad=False)
        self.conv.bias = torch.nn.Parameter(torch.tensor(bias), requires_grad=False)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return F.relu(x, inplace=True)

class Vgg19(nn.Module):
    def __init__(self, vgg19_npy_path):
        super(Vgg19, self).__init__()
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        self.vgg_net_components = {}

        for layer_name, value in self.data_dict.items():
            if "conv" in layer_name:
                kernel = value[0].astype(np.float32)
                bias = value[1].astype(np.float32)
                print(layer_name)
                print(kernel.shape)
                self.vgg_net_components[layer_name] = BasicConv2d(kernel.shape[2], kernel.shape[3], (kernel.shape[0], kernel.shape[1]), kernel, bias)

        self.pooling_layer = nn.AvgPool2d(kernel_size=[2, 2], stride=[2, 2])

        self.VGG_MEAN = [103.939, 116.779, 123.68]


    def forward(self, x, var=True):
        # if var:
        #     rgb = x.detach().numpy() * 255.0
        # else:
        #     rgb = x.numpy() * 255.0

        # x *= 255.0
        x = torch.mul(x, 255.0)
        # VGG_MEAN = [103.939, 116.779, 123.68]
        x[:,:,0] -= self.VGG_MEAN[0]
        x[:,:,1] -= self.VGG_MEAN[1]
        x[:,:,2] -= self.VGG_MEAN[2]

        self.conv1_1 = self.vgg_net_components["conv1_1"](x)
        self.conv1_2 = self.vgg_net_components["conv1_2"](self.conv1_1)
        pool_1 = self.pooling_layer(self.conv1_2)
        self.conv2_1 = self.vgg_net_components["conv2_1"](pool_1)
        self.conv2_2 = self.vgg_net_components["conv2_2"](self.conv2_1)
        pool_2 = self.pooling_layer(self.conv2_2)
        self.conv3_1 = self.vgg_net_components["conv3_1"](pool_2)
        self.conv3_2 = self.vgg_net_components["conv3_2"](self.conv3_1)
        self.conv3_3 = self.vgg_net_components["conv3_3"](self.conv3_2)
        self.conv3_4 = self.vgg_net_components["conv3_4"](self.conv3_3)
        pool_3 = self.pooling_layer(self.conv3_4)
        self.conv4_1 = self.vgg_net_components["conv4_1"](pool_3)
        self.conv4_2 = self.vgg_net_components["conv4_2"](self.conv4_1)
        self.conv4_3 = self.vgg_net_components["conv4_3"](self.conv4_2)
        self.conv4_4 = self.vgg_net_components["conv4_4"](self.conv4_3)
        pool_4 = self.pooling_layer(self.conv4_4)
        self.conv5_1 = self.vgg_net_components["conv5_1"](pool_4)


def main(args):

    print(args.content_image_path)

    # prepare input images
    content_image = load_image(args.content_image_path, scale=float(args.content_scale), args=args)
    WIDTH, HEIGHT = content_image.shape[1], content_image.shape[0]
    content_image = content_image.reshape((1, HEIGHT, WIDTH, 3))
    style_image = load_image(args.style_image_path, (WIDTH, HEIGHT), args=args)
    style_image = style_image.reshape((1, HEIGHT, WIDTH, 3))

    images = np.concatenate((content_image, style_image), 0).astype(np.float32)
    print(images.shape)
    images = images.transpose(0,3,1,2)
    constant_image = torch.tensor(images, requires_grad=False)
    print(constant_image.size())
    vgg_const = Vgg19("./vgg19.npy")
    vgg_const(constant_image)

    input_img = torch.tensor(np.expand_dims(images[0,:,:,:], 0), requires_grad=True)
    print(input_img.size())
    vgg_var = Vgg19("./vgg19.npy")
    vgg_var(input_img)



    content_layer_const = vgg_const.conv4_2
    content_layer_const = torch.unsqueeze(content_layer_const[0,:,:,:], 0)

    style_layers_const = [vgg_const.conv1_1, vgg_const.conv2_1, vgg_const.conv3_1, vgg_const.conv4_1, vgg_const.conv5_1]
    style_layers_const = [torch.unsqueeze(layer[1,:,:,:], 0) for layer in style_layers_const]

    optimizer = torch.optim.Adam([input_img], lr=args.learning_rate)

    for i in range(500):
        vgg_var(input_img)

        style_layers_var = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1, vgg_var.conv5_1]
        content_layer_var = vgg_var.conv4_2

        # compose the loss function
        content_style_ratio = 1e-4
        loss_content = content_loss(content_layer_const, content_layer_var)
        loss_style = style_loss(style_layers_const, style_layers_var)
        style_weight = float(args.style_weight)
        overall_loss = (1 - style_weight) * content_style_ratio * loss_content + style_weight * loss_style

        # Backward and optimize
        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     input_img -= args.learning_rate * input_img.grad
        #     input_img.grad.zero_()

        if i % 10 == 0:
            print('Iteration {}: {}'.format(i, overall_loss))
            output_img = input_img.detach().numpy()[0].transpose(1,2,0)
            output_tmp = np.clip(output_img, 0, 1)
            output_img = convertRGB2BGR(output_tmp)
            print(output_img.shape)
            result = Image.fromarray(np.uint8(output_img*255))
            result.save("out_" + str(i) + ".jpg")

def gram_matrix(activations):
    # print(activations.size())
    height = activations.size()[2]
    width = activations.size()[3]
    num_channels = activations.size()[1]
    gram_matrix = activations.permute(0, 3, 1, 2)
    gram_matrix = torch.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = torch.matmul(gram_matrix, torch.t(gram_matrix))
    return gram_matrix

def content_loss(const_layer, var_layer):
    diff = const_layer - var_layer
    diff_squared = diff * diff
    sum = torch.sum(diff_squared) / 2.0
    return sum

def style_loss(const_layers, var_layers):
    loss_style = 0.0
    layer_count = float(len(const_layers))
    for const_layer, var_layer in zip(const_layers, var_layers):
        gram_matrix_const = gram_matrix(const_layer)
        gram_matrix_var = gram_matrix(var_layer)

        size = const_layer.numpy().shape[-1]
        diff_style = gram_matrix_const - gram_matrix_var
        diff_style_sum = torch.sum(diff_style * diff_style) / (4.0 * size * size)
        loss_style += diff_style_sum
    return loss_style / layer_count


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("content_image_path", help="Path to the content image")
    parser.add_argument("style_image_path", help="Path to the style image")
    parser.add_argument("output_image", nargs='?', help='Path to output the stylized image', default="out.jpg")
    parser.add_argument('crop', nargs='?', help='Where ', default='center', choices=('top', 'center', 'bottom', 'left', 'right'))
    parser.add_argument("content_scale", nargs='?', help='Optional scaling of the content image', default=1.0)
    parser.add_argument("style_weight", nargs='?', help="Number between 0-1 specifying influence of the style image", default=0.5)
    parser.add_argument("learning_rate", default=0.05, nargs='?',)
    return parser.parse_args(args)


if __name__ == "__main__":
    # main(sys.argv[1:])
    main(parse_arguments(sys.argv[1:]))
