"""Deterministic image-only bright-region hypothesis; not an annotated disc detector."""
import numpy as np
from PIL import Image,ImageFilter
import torch
from dataset import FundusDataset


def bright_box(image):
    small=image.copy();small.thumbnail((256,256),Image.Resampling.BILINEAR)
    rgb=np.asarray(small,dtype=np.float32);green=rgb[:,:,1]
    valid=(rgb[:,:,0]>20)&(green>5)
    yy,xx=np.where(valid)
    if len(xx)<64 or green[valid].std()<1:return None
    x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1
    if min(x1-x0,y1-y0)<16:return None
    channel=small.getchannel('G')
    score=np.asarray(channel.filter(ImageFilter.GaussianBlur(3)),dtype=np.float32)-.7*np.asarray(channel.filter(ImageFilter.GaussianBlur(15)),dtype=np.float32)
    inside=np.zeros(valid.shape,dtype=bool);dx=max(1,round(.1*(x1-x0)));dy=max(1,round(.1*(y1-y0)))
    inside[y0+dy:y1-dy,x0+dx:x1-dx]=True;inside&=valid
    if not inside.any():return None
    score[~inside]=-np.inf;cy,cx=np.unravel_index(score.argmax(),score.shape)
    # An achromatic bright maximum may be glare; retain the full view instead.
    if green[cy,cx]-rgb[cy,cx,2]<=5:return None
    width,height=image.size;side=max(1,round(.45*min((x1-x0)*width/small.width,(y1-y0)*height/small.height)))
    left=max(0,min(width-side,round((cx+.5)*width/small.width-side/2)));top=max(0,min(height-side,round((cy+.5)*height/small.height-side/2)))
    return left,top,left+side,top+side


def read_bright_image(path,size,jpeg_draft):
    with Image.open(path) as image:
        if jpeg_draft and image.format=='JPEG':image.draft('RGB',(2*size,2*size))
        image=image.convert('RGB');box=bright_box(image)
        return image.crop(box) if box is not None else image.copy()


class BrightRegionDataset(FundusDataset):
    def __getitem__(self,index):
        name,path,label=self.samples[index]
        image=read_bright_image(path,self.size,self.jpeg_draft)
        return self.transform(image),torch.tensor(float(label)),name
