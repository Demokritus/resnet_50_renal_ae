import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2
from typing import *


class Dataset:
    def __init__(self, path, IMG_SIZE = (512, 512)):
        self.path = path
        self.img_names = [x for x in filter(lambda x: x.endswith('.tif'), os.listdir(path))]
        self.IMG_SIZE = IMG_SIZE

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # img = Image.open(os.path.join(self.path, self.img_names[idx]))
        img = self.gen_cropped_image(os.path.join(self.path, self.img_names[idx]))
        # item_cat = self.img_labels[idx][2]
        # item_name = self.img_labels[idx][3]
        # image = random_crop[0]
        # x, y, box_size = random_crop[1]
        
        if type(img) == np.ndarray:
            img = Image.fromarray(img.astype(np.uint8))
        
        img = self.transform(img)
        min_img = torch.min(img)
        max_img = torch.max(img)
        eps = 1e-6
        norm_img = (img - min_img + eps) / (max_img - min_img + eps)
        
        return norm_img, self.img_names[idx]

    
    def transform(self, x):
        transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.IMG_SIZE)
        ])
        # x2 = transform(x)
        # x_dummy = torch.zeros((1, *self.IMG_SIZE))
        # x3 = torch.cat([x2, x_dummy, x_dummy], dim=0)
        x3 = transform(x)
        return x3
    

    def getROIPointsV2(self, img_path : str, 
                       min_share : float = 0.001) -> Tuple[List,List]:
        '''
        img_path : str
            a path to an image,
        min_share : float
            a minimal share of non-black pixels in a given segment (filtering)
            
        ______
        Return:
            a tuple of tuples, first tuple - X and Y of top left corner, 
            second tuple - X and Y of bottom right corner
        '''
        img = cv2.imread(img_path)
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cp_img_gray = np.copy(img_gray)
        img_mean = np.mean(img_gray)
        
        img_gray[img_gray < 2 * img_mean] = 0
        img_gray[img_gray >= 2 * img_mean] = 255
        
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        ret, thresh = cv2.threshold(cp_img_gray, 2 * img_mean, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)

        # cv2.fillPoly(img_gray, pts = contours, color=(255,0,255))
        cv2.fillPoly(thresh, pts = contours, color=(255,0,255))

        #===DILATION===
        kernel = np.ones((3,3), np.uint8)   # set kernel as 3x3 matrix from numpy
        # Create dilation image from the original image
        # img_alt = cv2.dilate(img_gray, kernel, iterations=15)
        img_alt = cv2.dilate(thresh, kernel, iterations=15)

        ret, thresh = cv2.threshold(img_alt, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)

        hull = []

        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], returnPoints=True))
        
        # ===DRAWING===
        # create an empty black image
        islands = np.zeros((len(contours), thresh.shape[0], thresh.shape[1]))
        # segments = np.zeros((len(contours), thresh.shape[0], thresh.shape[1]))
        segments = []
        for i in range(len(contours)):
            color = 255
            cv2.drawContours(islands[i, :, :], hull, i, color, -1, 8)
            
            share = lambda piece: np.sum(piece == 255) / np.size(piece)
            # share = np.sum(islands[i,:,:] == color) / np.size(islands[i,:,:])
            if share(islands[i, :, :]) >= min_share:
                segments.append(islands[i, :, :])
        
        positions = []
        for k in range(len(segments)):
            x1 = np.min(np.where(segments[k])[0])
            y1 = np.min(np.where(segments[k])[1])
            x2 = np.max(np.where(segments[k])[0])
            y2 = np.max(np.where(segments[k])[1])
            positions.append(((x1,y1), (x2,y2)))

        # return segments, islands
        return positions, segments  
        

    def gen_cropped_image(self, img_path : str,
                          min_share : float = 0.2) -> Tuple[np.array, Tuple]:
        '''
        The function generates a randomly cropped image from a randomly chosen ROI on original image
        
        Arguments:
        img_path : str
            a path to an image file,
        min_share : float
            minimal share of tissue island on a given cropped image (threshold),
        box_size : int
            the size of a cropped image
        
        Return:
        cropped_image : np.array
            a cropped image
        '''
        
        # box size
        box_size : int = self.IMG_SIZE[0]
        
        # checking a randomly assigned box on an image for number of white pixels
        share = lambda img, x1, y1, x2, y2: \
                        np.sum(img[x1:x2,y1:y2] == 255) / np.size(img[x1:x2,y1:y2])
        
        # checking a whole image for the size of foreground
        check = lambda img, img_mean, box_size, min_share: \
                        np.sum(img >= 1.05 * img_mean) / box_size ** 2 >= min_share
        # check = lambda img, img_mean, box_size, min_share: np.sum(img >= 1.5 * img_mean) / box_size ** 2 >= min_share
        
        # acquiring positions of isles and isles themselves
        positions, isles = self.getROIPointsV2(img_path, min_share = min_share)
        
        # opening an image using img_path argument
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_mean = np.mean(img)

        # function returns nothing if the whole image does not meet the criterion
        if not check(img, img_mean, box_size, min_share):
            # print("EMPTY IMAGE")
            return None
        
        # picking a random position from a number of positions
        L = len(positions)
        k = np.random.randint(0, L)
        
        # generating random coordinates of top left corner of a box 
        # inside of an area around tissue blob with coord-s ((x1,y1), (x2,y2))
        pt1, pt2 = positions[k]
        x1, y1 = pt1
        x2, y2 = pt2
        x = np.random.randint(x1, x2 - box_size)
        y = np.random.randint(y1, y2 - box_size)
        
        if share(isles[k], x1, y1, x2, y2) >= min_share:
            # return (img[x:x+box_size, y:y+box_size], (x, y, box_size))
            # print("INPUT IMAGE SHAPE: {}".format(img.shape))
            return img[x:x+box_size, y:y+box_size]
        else:
            gen_cropped_image(img_path, min_share=min_share, box_size=box_size)
