"""Reproduce selected portfolio plots from the legacy manifest and GCI table."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/corpus_v1_legacy'
OUT = ROOT / 'docs/figures'
FAMILIES = [f'G{i}' for i in range(2, 8)]
plt.rcParams.update({'font.size': 9, 'figure.dpi': 110, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight'})

def rows():
    latest = {}
    for line in (DATA / 'manifest.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['run_id'].startswith('M1_'):
            latest[r['run_id']] = r
    return list(latest.values())

def load(r):
    return np.load(DATA / 'm1' / r['family'] / (r['run_id'] + '.npz'))

def save(fig, name):
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = rows()
    ok = [r for r in all_rows if r['status'] == 'ok']
    reps = [sorted((r for r in ok if r['family'] == f and r['params']['rheology'] == 'Newtonian'), key=lambda r:r['run_id'])[0] for f in FAMILIES]
    fig, axes = plt.subplots(4, 2, figsize=(11, 8), constrained_layout=True)
    for ax, r in zip(axes.flat, reps):
        z = load(r); ax.contourf(z['grid_x'], z['grid_y'], z['grid_mask'], levels=[.5, 1.5], colors=['#56B4E9'])
        ax.set(title=r['run_id'], xlabel='x* = x / h₁', ylabel='y* = y / h₁')
    ax = axes.flat[6]
    # Wu 2015 KG geometry: radii 75.5 and 252 μm, inlet diameter 151 μm.
    ratio = 252 / 75.5
    ax.plot([-6.6225, 0, 0, 33.1126], [.5, .5, ratio/2, ratio/2], color='#0072B2')
    ax.plot([-6.6225, 0, 0, 33.1126], [-.5, -.5, -ratio/2, -ratio/2], color='#0072B2')
    ax.set(title='KG tube, axial outline (axisymmetric)', xlabel='x* = x / d₁', ylabel='r* = r / d₁')
    axes.flat[7].axis('off');fig.suptitle('Representative legacy geometry masks and validation tube'); save(fig,'fig01_geometries.png')
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    for ax, r in zip(axes.flat, reps):
        z = load(r); x, y = z['grid_x'], z['grid_y']
        speed = np.ma.masked_where(~z['grid_mask'].astype(bool), np.hypot(z['grid_ux'], z['grid_uy']))
        im = ax.pcolormesh(x, y, speed, shading='auto', cmap='viridis')
        ax.set_title(r['run_id'] + f"  Re={r['params']['Re']:.2f}")
        ax.set(xlabel='x* = x / h₁', ylabel='y* = y / h₁'); fig.colorbar(im, ax=ax, label='|u*| = |u| / U')
    fig.suptitle('Legacy Newtonian velocity fields'); save(fig, 'fig02_velocity_fields.png')
    # Reference dimensionalization for Newtonian runs: h₁ = 50 μm, ρ = 1060 kg/m³,
    # μ∞ = 0.00345 Pa s (knowledge/11); U follows from Re.
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    from matplotlib.colors import LogNorm
    for ax, r in zip(axes.flat, reps):
        z = load(r); x, y = z['grid_x'], z['grid_y']
        tau = np.ma.masked_where(~z['grid_mask'].astype(bool) | (z['grid_tau'] <= 0), z['grid_tau'])
        im = ax.pcolormesh(x, y, tau, shading='auto', cmap='viridis', norm=LogNorm(vmin=max(float(tau.min()), 1e-3), vmax=float(tau.max())))
        scale = r['params']['Re'] * .00345**2 / (1060 * (50e-6)**2)
        for threshold, color in [(30, '#D55E00'), (60, '#CC79A7')]:
            level = threshold / scale
            if tau.min() < level < tau.max():
                ax.contour(x, y, tau, levels=[level], colors=[color], linewidths=1)
        ax.set_title(r['run_id']); ax.set(xlabel='x* = x / h₁', ylabel='y* = y / h₁')
        fig.colorbar(im, ax=ax, label='τ* = τ h₁ / (μ∞ U)')
    fig.suptitle('Shear stress; contours: 30 Pa orange, 60 Pa pink\nNewtonian reference h₁ = 50 μm, ρ = 1060 kg/m³, μ∞ = 3.45 mPa s')
    save(fig, 'fig03_shear_fields.png')
    values = {f: [] for f in FAMILIES}
    for r in ok:
        p = r['params']; z = load(r)
        scale = (.00345 * p['Cu'] / 3.313 if p['rheology'] != 'Newtonian'
                 else p['Re'] * .00345**2 / (1060 * (50e-6)**2))
        flux = np.maximum(z['outlet_u_n'], 0) * z['outlet_weights']
        valid = np.isfinite(flux) & np.isfinite(z['outlet_tau'])
        denominator = flux[valid].sum()
        if denominator > 0:
            inside = valid & (z['outlet_tau'] * scale >= 30) & (z['outlet_tau'] * scale <= 60)
            values[r['family']].append(float(flux[inside].sum() / denominator))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.boxplot([values[f] for f in FAMILIES], tick_labels=FAMILIES, showfliers=True)
    ax.set(xlabel='Geometry family', ylabel='Outlet flux fraction', ylim=(-.04, 1.04),
           title='30–60 Pa outlet screening fraction; not exposure or loading')
    fig.text(.5, -.04, 'Reference mapping: μ∞ = 3.45 mPa s; λ = 3.313 s (Carreau, TODO source verification);\nNewtonian h₁ = 50 μm, ρ = 1060 kg/m³. Uniform RBC fraction; some Carreau scales are mm-sized.', ha='center', fontsize=8)
    save(fig, 'fig08_window_fraction.png')
    fig, axes = plt.subplots(4, 4, figsize=(11, 10), constrained_layout=True)
    names = ['Re', 'Cu', 'beta', 'n']; colors = ['#0072B2', '#D55E00']
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            ax=axes[i,j]
            for k, status in enumerate(['ok','failed']):
                subset=[r for r in all_rows if r['status']==status and r['params'].get(a) is not None and r['params'].get(b) is not None]
                xx=np.array([r['params'][b] for r in subset]); yy=np.array([r['params'][a] for r in subset])
                if i == j:
                    if a == 'Cu': xx=xx[xx>0]; xx=np.log10(xx)
                    ax.hist(xx, bins=20, alpha=.5, color=colors[k], label=status)
                else:
                    for fam in FAMILIES:
                        q=[r for r in subset if r['family']==fam]
                        ax.scatter([r['params'][b] for r in q], [r['params'][a] for r in q],s=8,alpha=.55,color=colors[k],marker=['o','s','^','v','D','P'][FAMILIES.index(fam)])
                    if b=='Cu': ax.set_xscale('log')
                    if a=='Cu': ax.set_yscale('log')
            if i==3: ax.set_xlabel('log₁₀ Cu' if i==j and b=='Cu' else b)
            if j==0: ax.set_ylabel(a)
    fig.suptitle('Legacy design coverage: blue ok, orange failed; markers distinguish G2–G7\nNewtonian Cu, β, n are undefined and omitted'); save(fig,'fig05_parameter_coverage.png')
    lines=(ROOT/'cfd/openfoam/gci_summary.md').read_text().splitlines()
    records=[]
    for line in lines:
        if line.startswith('| GCI_'):
            p=[v.strip() for v in line.strip('|').split('|')]
            if p[3] in ('Δp*','x_r/h₁'):
                records.append((p[0],p[3],*[float(v) for v in p[4:7]]))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,metric in zip(axes,['Δp*','x_r/h₁']):
        rr=[r for r in records if r[1]==metric]
        for level,c in enumerate(['#0072B2','#E69F00','#009E73']):
            ax.scatter([r[2+level] for r in rr], range(len(rr)), label=f'L{level}', color=c, marker=['o','s','^'][level])
        ax.set_yticks(range(len(rr)),[r[0].replace('GCI_','') for r in rr]);ax.invert_yaxis();ax.set_xlabel(metric);ax.legend()
    fig.suptitle('GCI mesh levels: G4–G6 have no reported reattachment scalar');save(fig,'fig06_gci.png')
    fig,ax=plt.subplots(figsize=(6,4)); x=np.array([12.2,37.8]);ax.plot(x,[1.5232,4.7351],'o-',label='Wu 2015 target (verified)');ax.plot(x,[1.5386,4.6615],'s--',label='Legacy OpenFOAM');ax.set(xlabel='Re (diameter based)',ylabel='xᵣ / d₁',title='Karino–Goldsmith tube: d₁ = 151 μm');ax.legend();save(fig,'fig07_validation_kg.png')
    fig,ax=plt.subplots(figsize=(7,5))
    for fam in FAMILIES:
        rr=[r for r in ok if r['family']==fam and r['params'].get('wall_seconds') and r['params'].get('n_cells')]
        ax.scatter([r['params']['n_cells'] for r in rr],[r['params']['wall_seconds'] for r in rr],s=15,alpha=.65,label=fam)
    ax.set(xscale='log',yscale='log',xlabel='Mesh cells',ylabel='Wall time (s)',title='Legacy 2D run time; no 3D timing measured');ax.legend(ncol=2);save(fig,'fig10_data_scaling.png')
    print('representatives:', ', '.join(r['run_id'] for r in reps))
    print('M1 latest rows:',len(all_rows),'ok:',len(ok))

if __name__ == '__main__':
    main()
