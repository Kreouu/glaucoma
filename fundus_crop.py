"""Conservative label-independent near-black border trimming, before letterboxing."""
import numpy as np


class TrimBlackBorder:
    def box(self,image):
        pixels=np.asarray(image)
        active=pixels.max(axis=2)>7
        rows=np.flatnonzero(active.mean(axis=1)>=.01)
        cols=np.flatnonzero(active.mean(axis=0)>=.01)
        width,height=image.size
        if not len(rows) or not len(cols):return (0,0,width,height)
        pad=max(2,round(.02*max(width,height)))
        left=max(0,int(cols[0])-pad);right=min(width,int(cols[-1])+1+pad)
        top=max(0,int(rows[0])-pad);bottom=min(height,int(rows[-1])+1+pad)
        if (right-left)*(bottom-top)<.25*width*height:return (0,0,width,height)
        return left,top,right,bottom

    def __call__(self,image):return image.crop(self.box(image))
