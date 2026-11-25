"""Compose selected owner-provided 2023 presentation images without copying raw assets."""
from io import BytesIO
from zipfile import ZipFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from legacy_figures import ROOT
import style
plt.rcParams['savefig.bbox'] = None

def main():
    deck=ROOT/'knowledge/papers/pdf/nalla-gereige-2023-slides.pptx'
    panels=[('image22.png','Expansion: velocity field'),('image24.png','U-bend: shear stress (Pa)'),('image18.png','Maximum shear'),('image25.png','Exposure time'),('image33.png','Encapsulation index')]
    fig,axes=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
    with ZipFile(deck) as z:
        for ax,(name,label) in zip(axes.flat,panels):
            img=Image.open(BytesIO(z.read('ppt/media/'+name))).convert('RGB')
            if name == 'image24.png':
                legend=Image.open(BytesIO(z.read('ppt/media/image41.png'))).convert('RGB')
                canvas=Image.new('RGB',(img.width+legend.width,max(img.height,legend.height)),'white'); canvas.paste(img,(0,0)); canvas.paste(legend,(img.width,0)); img=canvas
            ax.imshow(img); ax.set_title(label); ax.axis('off')
    axes.flat[-1].axis('off');fig.suptitle('2023 COMSOL results — Nalla & Gereige 2023')
    for ext in ['png','svg']: fig.savefig(ROOT/f'docs/figures/fig09_comsol_2023.{ext}',dpi=200)
    plt.close(fig)
if __name__=='__main__': main()
