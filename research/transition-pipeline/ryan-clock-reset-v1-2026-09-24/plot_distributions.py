"""Plot empirical discrete score distributions; no fitted normal curve."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/packers-falcons-matplotlib')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parent/'matchup'
r=json.loads((ROOT/'results.json').read_text())['imputed']
scores=np.load(ROOT/'imputed-scores.npy');scores=scores[(scores>=0).all(axis=1)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,7.4))
fig.patch.set_facecolor('#f7f8fa')
fig.suptitle('Green Bay (home) vs Atlanta (away)',fontsize=23,fontweight='bold',y=.965,color='#172b3a')
fig.text(.5,.895,'Elapsed-clock reset • 49,591 completed of 50,000 attempts',ha='center',fontsize=13,color='#435767')
for ax,name,values,title,color,xlabel in [
    (axes[0],'total',scores.sum(axis=1),'Total points','#176b59','Green Bay score + Atlanta score'),
    (axes[1],'home_minus_away',scores[:,0]-scores[:,1],'Home minus away','#a33c48','Green Bay score − Atlanta score (positive favors Green Bay)')]:
    stat=r['distributions'][name]
    x,c=np.unique(values,return_counts=True)
    ax.bar(x,c/len(values),width=.9,color=color,alpha=.85)
    ax.axvline(stat['mu'],color='#172b3a',ls='--',lw=1.6)
    if name=='home_minus_away':ax.axvline(0,color='#777',lw=1,alpha=.6)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=16,pad=14)
    ax.text(.97,.95,f"μ = {stat['mu']:.2f} points\nσ = {stat['sigma']:.2f} points",ha='right',va='top',transform=ax.transAxes,
        fontsize=13,bbox={'boxstyle':'round,pad=.5','facecolor':'white','edgecolor':'#d7dee4','alpha':.97})
    ax.set_xlabel(xlabel,fontsize=10,labelpad=12);ax.set_ylabel('Simulated probability')
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.set_ylim(0,max(c/len(values))*1.28)
fig.subplots_adjust(left=.075,right=.975,bottom=.29,top=.81,wspace=.24)
fig.text(.075,.18,'Artificial terminal stops removed; remaining limitations recorded',fontsize=13,fontweight='bold',color='#9a412e')
fig.text(.075,.105,'409 runs (0.82%) failed on missing state/time support; plotted probabilities condition on completion.\nFinal-segment evidence is censored from transition training. These are not validated forecasts.',fontsize=11,color='#435767',linespacing=1.6)
fig.text(.075,.025,'Two 2026 games per team; same-clock-bucket historical imputation. No home-field or opponent-defense adjustment.\nDashed line: mean (μ). σ: score-distribution standard deviation, not uncertainty in the mean.',fontsize=9,color='#576675',linespacing=1.5)
fig.savefig(ROOT/'packers-falcons-distributions.png',dpi=180,facecolor=fig.get_facecolor())
fig.savefig(ROOT/'packers-falcons-distributions.pdf',facecolor=fig.get_facecolor())
