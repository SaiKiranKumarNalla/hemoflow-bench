"""Plot reported fine-grid GCI values from the legacy summary."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from legacy_figures import ROOT
import style
plt.rcParams['savefig.bbox']=None

def main():
    records=[]
    for line in (ROOT/'cfd/openfoam/gci_summary.md').read_text().splitlines():
        if line.startswith('| GCI_'):
            p=[x.strip() for x in line.strip('|').split('|')]
            if p[3] in ('Δp*','x_r/h₁') and p[9]!='-': records.append((p[0],p[3],float(p[9]),p[10]))
    cases=list(dict.fromkeys(r[0] for r in records));fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for metric,color,offset in [('Δp*','#0072B2',-.12),('x_r/h₁','#D55E00',.12)]:
        rr=[r for r in records if r[1]==metric];ax.scatter([r[2] for r in rr],[cases.index(r[0])+offset for r in rr],label=metric,color=color,s=45)
    ax.axvline(5,color='#555555',linestyle='--',label='5% reference'); ax.axvline(2,color='#999999',linestyle=':',label='2% target for these quantities')
    ax.set_yticks(range(len(cases)),[c.replace('GCI_','') for c in cases]);ax.invert_yaxis();ax.set(xlabel='GCI L1→L2 (%)',title='Reported fine-grid convergence estimates');ax.grid(axis='x',alpha=.2);ax.legend()
    for ext in ('png','svg'):fig.savefig(ROOT/f'docs/figures/fig06_gci.{ext}',dpi=200)
    print('reported points:',len(records),'max:',max(r[2] for r in records))
if __name__=='__main__':main()
