import os 
import re 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset 
from PIL import Image
import ast

class ImageCaptionDataset(Dataset):

    def __init__(self, imageFolder, dataFrame, tokenizer, transform, seq_len = 100, columns = ('imgName', 'captions')):

        super(ImageCaptionDataset, self).__init__()

        self.imageFolder = imageFolder 
        self.dataFrame = dataFrame 
        self.tokenizer = tokenizer 
        self.transform = transform 
        self.seqLen = seq_len 
        self.columns = columns #(str, list)



    def __len__(self):

        return 2 * len(self.dataFrame) - 1

    def __getitem__(self, ix): 

        isgrt = False
        if ix > len(self.dataFrame) - 1:
            ix = ix - len(self.dataFrame)
            isgrt = True
        ## image stuff 
        
        imageName = self.dataFrame.iloc[ix][self.columns[0]]

        imagePath = os.path.join(self.imageFolder, imageName)
        image = Image.open(imagePath).convert("RGBA")
        image = self.transform(image)

        ## caption stuff 
        if isgrt:
            # ix = ix - len(self.dataFrame)
            caption = ast.literal_eval(self.dataFrame.iloc[ix][self.columns[1]])[1]
        else:
            caption = ast.literal_eval(self.dataFrame.iloc[ix][self.columns[1]])[0]
        # caption = self.dataFrame.iloc[ix].captions 
        tok = self.tokenizer.encode(caption)
        tok.pad(self.seqLen)
        tokens = torch.FloatTensor(tok.ids)

        return {'src': image, 'trg': tokens}
