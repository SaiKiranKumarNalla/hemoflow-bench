"""Render illustrative steady-field tracer animations from legacy M1 grids."""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates, binary_erosion
from matplotlib import colormaps
from legacy_figures import rows, load, FAMILIES, ROOT
import style  # shared plotting defaults

OUT = ROOT / 'docs/animations'
FPS, FRAMES = 24, 240
W, H = 1600, 900
FONT = str(Path(__import__('matplotlib').get_data_path()) / 'fonts/ttf/DejaVuSans.ttf')

def sample(arr, x, y, gx, gy, order=1):
    coords = [(y-gy[0])/(gy[1]-gy[0]), (x-gx[0])/(gx[1]-gx[0])]
    return map_coordinates(arr.astype(float), coords, order=order, mode='constant', cval=np.nan)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows() if r['status'] == 'ok']
    rng = np.random.default_rng(20261125)
    for family in FAMILIES:
        r = sorted((r for r in ok if r['family']==family and r['params']['rheology']=='Newtonian'), key=lambda a:a['run_id'])[0]
        z = load(r); gx, gy = z['grid_x'], z['grid_y']; mask=z['grid_mask']; ux=z['grid_ux']; uy=z['grid_uy']; tau=z['grid_tau']
        speed=np.hypot(ux,uy); vmax=np.nanpercentile(speed[mask],99)
        rgba=(colormaps['viridis'](np.clip(speed/vmax,0,1))*255).astype('uint8')[:,:,:3]
        rgba=(rgba*.25+255*.75).astype('uint8'); rgba[~mask]=[228,228,228]; rgba[mask & ~binary_erosion(mask)] = [85,85,85]
        # Preserve equal spatial aspect inside a fixed 1600 x 900 canvas.
        scale=min(1450/(gx[-1]-gx[0]), 650/(gy[-1]-gy[0])); pw=int((gx[-1]-gx[0])*scale); ph=int((gy[-1]-gy[0])*scale); left=(W-pw)//2; top=(H-ph)//2
        base=Image.new('RGB',(W,H),'white'); base.paste(Image.fromarray(rgba).transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((pw,ph)),(left,top))
        d=ImageDraw.Draw(base); font=ImageFont.truetype(FONT,30); small=ImageFont.truetype(FONT,22)
        d.text((70,45),f'{family}: tracers in a steady Newtonian velocity field',font=font,fill='#202020')
        d.text((70,790),'Colour: local shear stress (Pa); orange ring: 30–60 Pa. h₁ = 50 µm.',font=small,fill='#333333')
        d.rectangle((left,top,left+pw,top+ph),outline='#555555',width=2)
        d.text((left,top+ph+12), f'x* = {gx[0]:.1f}', font=small, fill='#333333')
        d.text((left+pw-110,top+ph+12), f'{gx[-1]:.1f}', font=small, fill='#333333')
        d.text((left,top-30), f'y* = {gy[-1]:.1f}', font=small, fill='#333333')
        bar = 50e-6 / 50e-6 * scale
        d.line((left+20,top+ph-15,left+20+bar,top+ph-15),fill='#333333',width=4)
        d.text((left+20,top+ph+40),'50 µm',font=small,fill='#333333')
        for j in range(300):
            color=tuple((colormaps['viridis'](j/299)[:3]*np.array([255,255,255])).astype(int))
            d.line((1100+j,790,1100+j,810),fill=color)
        d.text((1100,820),'0.1',font=small,fill='#333333'); d.text((1360,820),'100 Pa',font=small,fill='#333333')
        inlet=np.flatnonzero(mask[:,2]); ymin,ymax=gy[inlet[0]],gy[inlet[-1]]
        x=np.full(600,gx[2]); y=rng.uniform(ymin,ymax,600); history=[]
        dt=(gx[-1]-gx[0])/150
        scale_pa=r['params']['Re']*.00345**2/(1060*(50e-6)**2)
        name=f'anim01_particles_{family}'; mp4=OUT/(name+'.mp4')
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','23',str(mp4)],stdin=subprocess.PIPE)
        for frame in range(FRAMES):
            u=sample(ux,x,y,gx,gy); v=sample(uy,x,y,gx,gy)
            xm=x+.5*dt*u; ym=y+.5*dt*v
            un=sample(ux,xm,ym,gx,gy); vn=sample(uy,xm,ym,gx,gy)
            xn=x+dt*un; yn=y+dt*vn
            good=np.isfinite(un)&np.isfinite(vn)&(sample(mask,xn,yn,gx,gy,0)>.5)&(sample(mask,xm,ym,gx,gy,0)>.5)
            x=np.where(good,xn,gx[2]); y=np.where(good,yn,rng.uniform(ymin,ymax,len(y)))
            history.append((x.copy(),y.copy())); history=history[-4:]
            img=base.copy(); draw=ImageDraw.Draw(img)
            for age,(hx,hy) in enumerate(history[:-1]):
                for px,py in zip(left+(hx-gx[0])*scale,top+ph-(hy-gy[0])*scale):
                    draw.ellipse((px-1,py-1,px+1,py+1),fill=(190-age*20,190-age*20,190-age*20))
            t=sample(tau,x,y,gx,gy)*scale_pa
            colors=(colormaps['viridis'](np.clip((np.log10(np.maximum(t,.1))+1)/3,0,1))*255).astype('uint8')
            for px,py,c,stress in zip(left+(x-gx[0])*scale,top+ph-(y-gy[0])*scale,colors,t):
                rad=4 if 30<=stress<=60 else 3
                draw.ellipse((px-rad,py-rad,px+rad,py+rad),fill=tuple(c[:3]),outline='#D55E00' if 30<=stress<=60 else None)
            draw.text((70,90),f't* = {frame*dt:.2f}    Re = {r["params"]["Re"]:.2f}',font=small,fill='#333333')
            proc.stdin.write(img.tobytes())
        proc.stdin.close(); assert proc.wait()==0
        subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(mp4),'-vf','fps=8,scale=800:-2:flags=lanczos','-loop','0',str(OUT/(name+'.gif'))],check=True)
        print(name,r['run_id'],flush=True)
if __name__ == '__main__': main()
