import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
import skimage.transform as transform
import skimage.io as io

    
@torch.no_grad()
def search(q_img,k_img,v_img,patch_size,patch_step,alpha=0.1):
    r'''
    input image : H W C
    
    '''
    data=torch.stack([
        torch.from_numpy(q_img).permute(2,0,1).cuda().float(),
        torch.from_numpy(k_img).permute(2,0,1).cuda().float(),
        torch.from_numpy(v_img).permute(2,0,1).cuda().float()
    ])
    patch=F.unfold(data,patch_size,stride=patch_step).permute(0,2,1)
    q,k,v=patch

    dis=(q[:,None,...]-k[None,...])**2
    dis=dis.sum(dim=-1)/dis.shape[-1]
    thr=torch.min(dis,dim=0,keepdim=True)[0]+alpha
    dis=dis/thr
    value_min_id=torch.argmin(dis,dim=1).view(-1)
    knn_out=torch.index_select(v,0,value_min_id).transpose(1,0)
    knn_count=torch.ones_like(knn_out)
    out=F.fold(knn_out,q_img.shape[:2],patch_size,stride=patch_step)
    out_count=F.fold(knn_count,q_img.shape[:2],patch_size,stride=patch_step)
    out=out/out_count
    out=out.permute(1,2,0).cpu().numpy()
    return out

class PNN:
    def __init__(self,img_size=1024,patch_size=7,ratio=0.75,*args,**dargs) -> None:
        self.img_size=img_size
        self.patch_size=patch_size
        import math
        self.layer_num=int(math.log(patch_size*2/img_size,ratio)+0.5)
        self.ratio=ratio
        self.patch_step=1
        self.patch_layer=[]
        self.noise_base=None


    def add_image(self,img_path):
        def read_img(img_path):
            img=Image.open(img_path)
            img_data=np.array(img)/127-1
            img.close()
            H,W,C=img_data.shape
            if H>W:
                scale=self.img_size/W
            
            else:
                scale=self.img_size/H
            img_data=transform.resize(img_data,(int(H*scale),int(W*scale)))
            return img_data
        
        img_data=read_img(img_path)
        for id,i in enumerate(transform.pyramid_gaussian(img_data,downscale=1/self.ratio,max_layer=self.layer_num)):
            io.imsave(f'{id}.jpg',((i+1)*255/2).astype(np.uint8))
            print(id,i.shape)
            self.noise_base=i
            self.patch_layer.append(i)

    def generator_img(self):
        out=np.random.randn(*self.noise_base.shape)*0.3
        # out=np.random.randn(*self.noise_base.shape)*0.1+self.noise_base
        # out=self.noise_base
        v_img=None
        for layer_id,layer_img in enumerate(reversed(self.patch_layer)):
            if v_img is None:
                k_img=layer_img
            else:
                k_img=transform.resize(v_img,output_shape=layer_img.shape[:2])
                out=transform.resize(out,output_shape=layer_img.shape[:2])
            v_img=layer_img

            for run_id in range(3):
                out=search(out,k_img,v_img,self.patch_size,self.patch_step)

                io.imsave(f'{layer_id}_{run_id}.jpg',((out+1)*255/2).astype(np.uint8))
            

if __name__=='__main__':
    # test_patch()
    a_path=r'D:\data\imagenet-mini\train\n01498041\n01498041_10992.JPEG'
    a=PNN(img_size=32)
    a.add_image(a_path)
    a.generator_img()