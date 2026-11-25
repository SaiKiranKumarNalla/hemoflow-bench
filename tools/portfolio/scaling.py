"""Plot legacy wall times and the task-specified illustrative 3D multiplier."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from legacy_figures import rows, ROOT, FAMILIES
import style
plt.rcParams['savefig.bbox']=None

def main():
    ok=[r for r in rows() if r['status']=='ok']
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),layout='constrained')
    for fam,color in zip(FAMILIES,style.PALETTE):
        rr=[r for r in ok if r['family']==fam and r['params'].get('wall_seconds') and r['params'].get('n_cells')]
        axes[0].scatter([r['params']['n_cells'] for r in rr],[r['params']['wall_seconds'] for r in rr],s=16,alpha=.65,label=fam,color=color)
    axes[0].set(xscale='log',yscale='log',xlabel='Mesh cells',ylabel='Wall time per run (s)',title='Measured legacy 2D runs');axes[0].legend(ncol=2)
    times=np.array([r['params']['wall_seconds'] for r in ok if r['params'].get('wall_seconds')]); median=np.median(times)
    axes[1].bar(['2D median','3D estimate'],[median,75*median],color=['#0072B2','#e1e1e1']); axes[1].errorbar([1],[75*median],yerr=[[25*median],[25*median]],fmt='none',color='#D55E00',capsize=6)
    axes[1].set(yscale='log',ylabel='Wall time per run (s)',title='Illustrative 3D estimate: 50–100 × 2D median');axes[1].grid(axis='y',alpha=.2)
    for ext in ('png','svg'):fig.savefig(ROOT/f'docs/figures/fig10_data_scaling.{ext}',dpi=200)
    print('2D median seconds',median,'count',len(times))
if __name__=='__main__':main()
