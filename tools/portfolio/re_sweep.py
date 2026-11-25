"""Render a descriptive G2 sequence; geometry varies across legacy runs."""
import subprocess
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from legacy_figures import rows, load, ROOT
import style
plt.rcParams["savefig.bbox"] = None

def main():
    runs=sorted((r for r in rows() if r['status']=='ok' and r['family']=='G2' and r['params']['rheology']=='Newtonian'),key=lambda r:r['params']['Re'])
    idx=np.linspace(0,len(runs)-1,min(12,len(runs))).round().astype(int)
    selected=[runs[i] for i in idx]
    out=ROOT/'docs/animations'; out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i,r in enumerate(selected):
            z=load(r); x,y=z['grid_x'],z['grid_y']; mask=z['grid_mask']; u=np.ma.masked_where(~mask,z['grid_ux']); v=np.ma.masked_where(~mask,z['grid_uy'])
            fig,ax=plt.subplots(figsize=(16,9),dpi=100); ax.set_facecolor('#e5e5e5')
            ax.contourf(x,y,mask,levels=[.5,1.5],colors=['#f7f7f7']); ax.contourf(x,y,np.ma.masked_where(~mask,z['grid_ux']<0),levels=[.5,1.5],colors=['#e69f00'],alpha=.6)
            ax.streamplot(x,y,u,v,color='#0072b2',density=1,linewidth=.8,arrowsize=.7)
            ax.contour(x,y,mask,levels=[.5],colors=['#444444'],linewidths=.8); ax.set_aspect('equal'); ax.set(xlabel='x* = x / h₁',ylabel='y* = y / h₁',title=f'G2 Newtonian flow: Re = {r["params"]["Re"]:.2f}; expansion ratio = {r["params"]["geometry"]["E"]:.2f}')
            fig.text(.5,.15,'Orange: locally negative axial velocity. Geometry also varies across frames; this is not a controlled Reynolds-number sweep.',ha='center',fontsize=10)
            fig.savefig(Path(tmp)/f'{i:03d}.png',dpi=100);plt.close(fig)
        subprocess.run(['ffmpeg','-loglevel','error','-y','-framerate','1','-i',str(Path(tmp)/'%03d.png'),'-vf','fps=24','-c:v','libx264','-pix_fmt','yuv420p',str(out/'anim02_re_sweep_G2.mp4')],check=True)
        subprocess.run(['ffmpeg','-loglevel','error','-y','-i',str(out/'anim02_re_sweep_G2.mp4'),'-vf','fps=4,scale=800:-2','-loop','0',str(out/'anim02_re_sweep_G2.gif')],check=True)
    print(' '.join(r['run_id'] for r in selected))
if __name__=='__main__': main()
