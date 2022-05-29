import torch
from PIL import Image
from torchvision.transforms import transforms

from FaceHairMask import deeplab_xception_transfer
from FaceHairMask.graphonomy_inference import inference

import numpy as np
import cv2

def preprocess(image, size=256, normalize=1):
    if size is None:
        image = transforms.Resize((1024, 1024))(image)
    else:
        image = transforms.Resize((size, size))(image)
    image = transforms.ToTensor()(image)
    if normalize is not None:
        image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
    return image

def postProcess(faceMask, hairMask):
    hairMask = hairMask.cpu().permute(1,2,0).detach().numpy()
    faceMask = faceMask.cpu().permute(1,2,0).detach().numpy()
    return faceMask, hairMask

class MaskExtractor:
    def __init__(self):
        
        #? Hair Face Extractors
        self.net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20, hidden_layers=128, source_classes=7)
        stateDict = torch.load("models/Graphonomy/inference.pth")
        self.net.load_source_model(stateDict)
        self.net.to("cuda")
        self.net.eval()
        

    def processInput4(self, image):
        preprocessedImage = preprocess(image, size=256, normalize=1)
        preprocessedImage = preprocessedImage.unsqueeze(0).to("cuda")
        return preprocessedImage
    
    def getMask(self, image):
        preprocessedImage = self.processInput4(image)
        _, hairMask, faceMask = inference(net=self.net, img=preprocessedImage, device="cuda")
        faceMask, hairMask = postProcess(faceMask, hairMask)
        return hairMask, faceMask
        
    def main(self, image):
        image = (image.pixels_with_channels_at_back()[:, :, ::-1] * 255).astype('uint8')
        hairMask, faceMask = self.getMask(Image.fromarray(image))
        hairMask = transforms.Resize((Image.fromarray(image).size[1], Image.fromarray(image).size[0]))(Image.fromarray((hairMask[:,:,0]* 255).astype('uint8')))
        faceMask = transforms.Resize((Image.fromarray(image).size[1], Image.fromarray(image).size[0]))(Image.fromarray((faceMask[:,:,0]* 255).astype('uint8')))

        # Additional Morphology
        hairMask = np.array(hairMask) / 255
        faceMask = np.array(faceMask) / 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        faceMask = cv2.erode(faceMask, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        hairMask = cv2.dilate(hairMask, kernel, iterations=1)
        faceMask = faceMask * (1 - hairMask)


        return hairMask, faceMask