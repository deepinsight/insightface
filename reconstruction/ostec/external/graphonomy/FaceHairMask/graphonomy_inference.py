import cv2
import torch
import timeit
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


from FaceHairMask import graph
from FaceHairMask import graphonomy_process as tr



label_colours = [(0, 0, 0) for i in range(20)]
label_colours[2] = (255, 0, 0)
label_colours[13] = (0, 0, 255)


def custom_decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape

    # import ipdb; ipdb.set_trace()
    assert (
        n >= num_images
    ), "Batch size %d should be greater or equal than number of images to save %d." % (
        n,
        num_images,
    )

    hair_mask = torch.where(mask == 2, torch.ones_like(mask), torch.zeros_like(mask))

    face_mask = torch.where(mask == 13, torch.ones_like(mask), torch.zeros_like(mask))

    return hair_mask, face_mask


def overlay(frame, mask):

    mask = np.array(mask)
    frame = np.array(frame)

    tmp = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(mask)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    # overlay mask on frame
    overlaid_image = cv2.addWeighted(frame, 0.4, dst, 0.1, 0)
    return overlaid_image


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]


def flip_cihp(tail_list):
    """

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    """
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev, dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (
        n >= num_images
    ), "Batch size %d should be greater or equal than number of images to save %d." % (
        n,
        num_images,
    )
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new("RGB", (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def read_img(img_path):
    _img = Image.open(img_path).convert("RGB")  # return is RGB pic
    return _img


def img_transform(img, transform=None):
    sample = {"image": img, "label": 0}

    sample = transform(sample)
    return sample


def inference(net, img=None, device=None):
    """

    :param net:
    :return:
    """
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = (
        adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).to(device).transpose(2, 3)
    )

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).to(device)

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).to(device)

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    #scale_list = [1, 0.5, 0.75, 1.25]
    # NOTE: this part of the code assumes img is PIL image in RGB color space
    # We provide torch tensor in range [-1, 1]
    # Bring range to [0, 255]
    img = torch.clamp(img, -1, 1)
    img = (img + 1.0) / 2.0
    img *= 255

    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose(
            [
                tr.Scale_only_img(pv),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img(),
            ]
        )

        composed_transforms_ts_flip = transforms.Compose(
            [
                tr.Scale_only_img(pv),
                tr.HorizontalFlip_only_img(),
                tr.Normalize_xception_tf_only_img(),
                tr.ToTensor_only_img(),
            ]
        )
        # NOTE: img [1, 3, 256, 256], (min, max) = (0, 255)

        # print("original:", img.shape, img.min(), img.max())
        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    # net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    # NOTE: testloader_list[0]['image'].shape = 3, 420, 620

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]["image"], sample_batched[0]["label"]
        inputs_f, _ = sample_batched[1]["image"], sample_batched[1]["label"]
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch

        # TODO: check requires grad functionality
        # inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if device is not None:
                inputs = inputs.to(device)
            # outputs = net.forward(inputs)
            outputs = net.forward(
                inputs, adj1_test.to(device), adj3_test.to(device), adj2_test.to(device)
            )
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)
            if iii > 0:
                outputs = F.upsample(
                    outputs, size=(h, w), mode="bilinear", align_corners=True
                )
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    # results = predictions.cpu().numpy()
    # vis_res = decode_labels(results)
    # parsing_im = Image.fromarray(vis_res[0])
    # return parsing_im

    hair_mask, face_mask = custom_decode_labels(predictions)

    return outputs_final, hair_mask, face_mask

    # parsing_im.save(output_path+'/{}.png'.format(output_name))
    # cv2.imwrite(output_path+'/{}_gray.png'.format(output_name), results[0, :, :])

    # end_time = timeit.default_timer()
    # print('time used for the multi-scale image inference' + ' is :' + str(end_time - start_time))