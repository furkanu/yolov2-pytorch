from __future__ import division

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import detector

class _ReorgModule(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B, C, H, W = x.shape
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x
class _RegionModule(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors
    def forward(self, x):
        return x
class _EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
class _RouteModule(nn.Module):
    def __init__(self, start, end=None):
        super().__init__()
        self.start = start
        self.end = end
    def forward(self, x, outputs, idx):
        map1 = outputs[idx + self.start]
        if self.end:
            map2 = outputs[idx + self.end]
            return torch.cat([map1, map2], 1)
        return map1
class _MaxPoolStride1(nn.Module):
    def __init__(self):
        super(_MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

def _transform_predictions(preds, inp_size, anchors, num_classes, device):
    N, C, H, W = preds.shape
    grid_size = H
    stride = inp_size // grid_size
    num_anchors = len(anchors)
    bbox_attrs = num_classes + 5


    preds = preds.view(N, bbox_attrs*num_anchors, grid_size*grid_size) #This seems to be the same as preds.view(N, C, H*W)
    preds = preds.permute(0, 2, 1).contiguous()                        #N, (H*W), (5+C)*n_anchors
    preds = preds.view(N, grid_size*grid_size*num_anchors, bbox_attrs) #N, (H*W)*n_anchors, (5+C)

    #anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the center_x, center_y and object confidence
    preds[..., :2] = torch.sigmoid(preds[..., :2])
    obj_probs = torch.sigmoid(preds[..., 4])

    #Add the grid offsets to the center coordinates prediction
    grid = np.arange(grid_size)
    x_offsets, y_offsets = np.meshgrid(grid, grid)

    x_offsets = torch.tensor(x_offsets, dtype=torch.float, device=device).view(-1, 1) #a column vector
    y_offsets = torch.tensor(y_offsets, dtype=torch.float, device=device).view(-1, 1) #a column vector

    x_y_offsets = torch.cat([x_offsets, y_offsets], 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    preds[..., :2] += x_y_offsets #at this stage, (bx = sigmoid(tx) + cx ) and (by = sigmoid(ty) + cy) is done

    #Now that we're done with the center coordinates of the bounding boxes, let's process the widths and heights of them.
    #Apply the anchors to the dimensions of the bounding box.
    #log space transform height and width
    anchors = torch.tensor(anchors, dtype=torch.float, device=device)
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    preds[..., 2:4] = torch.exp(preds[..., 2:4])*anchors

    #Now we'are done with the center of the boxes and the height and width of the boxes.
    #Let's now apply softmax function to the class scores.
    #class_probs = F.softmax(preds[..., 5:], -1)
    class_scores = preds[..., 5:]

    #Resize detection map to the size of the input image
    boxes = preds[..., :4]*stride

    #return preds #(center_x, center_y, height, width)
    return boxes, obj_probs, class_scores #(center_x, center_y, height, width)
def _read_cfg_lines(cfgfile):
    with open(cfgfile, 'r') as f:
        lines = f.read().splitlines()
        lines = list(filter(None, lines)) #remove empty lines
        lines = [l for l in lines if l[0] != '#']
        lines = [x.strip() for x in lines] #get rid of whitespaces
    return lines
def _create_block_dicts(lines):
    block, blocks = {}, []
    for line in lines:
        if line[0] == '[':                              #This marks the start of a new block
            if len(block) != 0:                         #If block is not empty, implies it is storing values of previous block.
                blocks.append(block)                    #Add it to the block list
                block = {}                              #Re-init the block.
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()

    blocks.append(block)
    return blocks
def _make_conv_module(idx, block, prev_filters):
    module = nn.Sequential()
    activation = block['activation']
    batch_normalize = block.get('batch_normalize')
    filters = int(block['filters'])
    padding = int(block['pad'])
    kernel_size = int(block['size'])
    stride = int(block['stride'])

    if padding:
        pad = (kernel_size - 1) // 2
    else:
        pad = 0

    conv = nn.Conv2d(in_channels=prev_filters, out_channels=filters, kernel_size=kernel_size,
                     stride=stride, padding=pad, bias=False if batch_normalize else True)
    module.add_module(f'conv_{idx}', conv)

    if batch_normalize:
        bn = nn.BatchNorm2d(filters)
        module.add_module(f'batch_norm_{idx}', bn)

    #Activation is either Linear or a Leaky ReLU for YOLO
    if activation == 'leaky':
        leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        module.add_module(f'leaky_{idx}', leaky_relu)

    return module, filters
def _make_maxpool_module(block):
    #Both YOLO f/ PASCAL and COCO don't use 2X2 pooling with stride 1
    #Tiny-YOLO does use it
    kernel_size = int(block['size'])
    stride = int(block['stride'])

    if stride > 1:
        pool = nn.MaxPool2d(kernel_size, stride=stride)
    else:
        pool = _MaxPoolStride1()

    return pool
def _read_region_block(block, net_info):
    anchors = block['anchors'].split(',')
    anchors = [float(a) for a in anchors]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]

    net_info['anchors'] = anchors
    net_info['num_classes'] = int(block['classes'])


def _make_region_module(block):
    anchors = block['anchors'].split(',')
    anchors = [float(a) for a in anchors]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]

    region = _RegionModule(anchors)

    return region
def _make_route_module(idx, block, output_filters):
    block['layers'] = [int(l) for l in block['layers'].split(',')]
    start = block['layers'][0]
    #end, if there exists one
    if len(block['layers']) > 1:
        end = block['layers'][1]
    else:
        end = 0 #0 means that we should stay on the current layer. There is no route.

    #Positive annotation
    if start > 0:
        start = start - idx
    if end > 0:
        end = end - idx

    if end < 0:
        filters = output_filters[idx + start] + output_filters[idx + end]
    else:
        filters = output_filters[idx + start]

    module = _RouteModule(start, end)

    return module, filters
def _make_reorg_module(block, prev_filters):
    stride = int(block['stride'])
    reorg = _ReorgModule(stride)
    prev_filters = stride*stride*prev_filters
    return reorg, prev_filters
def _parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each block describes a block in the neural
    network to be built. Block is represented as a dictionary in the list.
    """
    lines = _read_cfg_lines(cfgfile)
    blocks =_create_block_dicts(lines)
    return blocks
def _create_modules(block_dicts):
    net_info = block_dicts[0]       #Captures the 'net' block, which has information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3                #We initialize this to 3 since the image has 3 channels
    output_filters = []

    for idx, block in enumerate(block_dicts[1:]):
        if block['type'] == 'convolutional':
            module, prev_filters = _make_conv_module(idx, block, prev_filters)
        elif block['type'] == 'maxpool':
            module = _make_maxpool_module(block)
        elif block['type'] == 'shortcut':
            module = _EmptyModule()
        elif block['type'] == 'route':
            module, prev_filters = _make_route_module(idx, block, output_filters)
        elif block['type'] == 'reorg':
            module, prev_filters = _make_reorg_module(block, prev_filters)
        elif block['type'] == 'region':
            _read_region_block(block, net_info)
            break
            #module = _make_region_module(block)

        module_list.append(module)
        output_filters.append(prev_filters)

    return module_list, net_info
class Darknet(nn.Module):
    def __init__(self, cfgfile, device, input_size=None):
        super().__init__()
        self.block_dicts = _parse_cfg(cfgfile)
        self.module_list, self.net_info = _create_modules(self.block_dicts)
        self.device = device
        if input_size is not None:
            self.input_size = input_size
        else:
            self.input_size = int(self.net_info['height'])

    def forward(self, x):
        outputs = {} #We cache the outputs since route and shortcut layers will need them
        for i, block in enumerate(self.block_dicts[1:-1]): #blocks except net block and region block
            if block['type'] in ('convolutional', 'maxpool', 'reorg'):
                x = self.module_list[i](x)
            elif block['type'] == 'route':
                x = self.module_list[i](x, outputs, i)
            elif block['type'] == 'shortcut':
                from_ = int(block['from'])
                x = outputs[i-1] + outputs[i + from_]
            outputs[i] = x

        boxes, obj_probs, class_probs = _transform_predictions(x, self.input_size, self.net_info['anchors'],
                                        self.net_info['num_classes'], self.device)

        return boxes, obj_probs, class_probs
    def load_weights(self, weightfile):
        #The first 5 values are header information (These might not be correct.)
        # 1. Major version number
        # 2. Minor version number
        # 3. Subversion number
        # 4 Images seen by the network (during training)
        with open(weightfile) as f:
            header = np.fromfile(f, dtype=np.int32, count=4)
            self.header = torch.tensor(header, dtype=torch.int)
            self.seen = self.header[3]
            weights =  np.fromfile(f, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.block_dicts[i+1]['type'] #i+1 because this includes the 'net' block
            if module_type == 'convolutional':
                module = self.module_list[i]
                batch_normalize = self.block_dicts[i+1].get('batch_normalize')  #i+1 because this includes the 'net' block
                conv = module[0]

                if batch_normalize:
                    bn = module[1]

                    #Get the number of weights of the Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    #Convert the shape of the loaded weights to those of the model's
                    bn_biases = bn_biases.view_as(bn.bias.detach())
                    bn_weights = bn_weights.view_as(bn.weight.detach())
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the datasets to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases

                    #Reshape the loaded weigths according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the datasets
                    conv.bias.data.copy_(conv_biases)

                #Load the weights for the convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def full_forward(self, x):
        boxes, obj_probs, class_scores = self.forward(x)
        return boxes, obj_probs, torch.softmax(class_scores, -1)

    def visualize(self, data, cats, threshold=0.5, nms_threshold=0.5, num_images=9, figsize=(20,20)):
        i = 0
        plot_size = np.sqrt(num_images)
        if not float.is_integer(plot_size):
            plot_size +=1
        plot_size = int(plot_size)
        fig, axs = plt.subplots(ncols=plot_size, nrows=plot_size, figsize=figsize)
        for (x, _) in data.val_dl:
            inp = x.to(self.device)
            p_boxes_b, p_obj_probs_b, p_class_probs_b =  self.full_forward(inp)
            for (im, p_boxes, p_obj_probs, p_class_probs) in zip(inp, p_boxes_b, p_obj_probs_b, p_class_probs_b):
                detections = detector.filter_predictions(p_boxes, p_obj_probs, p_class_probs, cats,
                                                         threshold=threshold, nms_threshold=nms_threshold)
                ax = axs.flat[i]
                ax.axis('off')
                im = data.denorm(im)
                im = im.clip(0, 1)

                out_img = detector.draw_predictions(im, detections)
                out_img = out_img.clip(0, 1)
                ax.imshow(out_img)
                i+= 1
                if i == num_images:
                    for j in range(i, plot_size*plot_size):
                        fig.delaxes(axs.flat[j])
                    return
