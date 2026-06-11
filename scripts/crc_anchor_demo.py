import pandas as pd, numpy as np, json, math
B='/sessions/optimistic-peaceful-bardeen/mnt/detr_Q3'
cal=pd.read_csv(B+'/data/dfire_local/eval_calib_labels.csv'); tst=pd.read_csv(B+'/data/dfire_local/eval_test_labels.csv')
def scores(pred,labels):
    df=pd.read_csv(pred); s=df.groupby('image')['conf'].max()
    out=labels.copy(); out['score']=out.image.map(s).fillna(0.0); return out
try:
    from scipy.stats import binom; bcdf=lambda k,n,p: binom.cdf(k,n,p)
except Exception:
    from math import lgamma
    def bcdf(k,n,p):
        k=int(k); 
        return sum(math.exp(lgamma(n+1)-lgamma(i+1)-lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log(1-p)) for i in range(0,k+1))
def pr_pick(pos,target):
    s=np.sort(pos)[::-1]; k=math.ceil(target*len(pos)); return s[k-1]
def crc_pick(pos,alpha):
    N=len(pos); mstar=math.floor(alpha*(N+1)-1)
    if mstar<0: return -np.inf
    s=np.sort(pos); return s[mstar] if mstar< N else np.inf  # largest tau with #(s<tau)<=mstar
def ltt_pick(pos,alpha,delta=0.05):
    N=len(pos); s=np.unique(np.sort(pos))[::-1]
    best=-np.inf
    # monotone: find max tau with p<=delta (FST ascending equivalent)
    lo,hi=0,len(s)-1; cand=None
    miss=lambda tau: int((pos<tau).sum())
    # binary search over sorted desc candidates: p increases with tau
    arr=s
    L,R=0,len(arr)-1
    while L<=R:
        M=(L+R)//2; tau=arr[M]; p=bcdf(miss(tau),N,alpha)
        if p<=delta: cand=tau; R=M-1  # tau ok, try larger (smaller index = larger tau)
        else: L=M+1
    return cand if cand is not None else -np.inf
def evaln(d,tau):
    pos=d[d.label!='none']; neg=d[d.label=='none']
    return (pos.score>=tau).mean(), (neg.score>=tau).mean()
runs=[('yolo',s,a,B+f'/formal_results/stage5/yolo_seed{s}/{a}.csv') for s in (11,22,33) for a in ('baseline','hardneg')]
runs+=[('rtdetr',s,a,B+f'/formal_results/stage5_pv_v2/dfire_rtdetr_seed{s}/{a}.csv') for s in (11,22) for a in ('baseline','hardneg')]
rows=[]
cache={}
for fam,seed,arm,p in runs:
    dc=scores(p.replace('stage5/','stage5/').replace('.csv','.csv'),cal) if False else None
    dcal=scores(p,cal); dtst=scores(p,tst); cache[(fam,seed,arm)]=(dcal,dtst)
    posc=dcal[dcal.label!='none'].score.values
    for R0 in (0.85,0.90,0.95):
        a=1-R0
        for name,tau in (('PR',pr_pick(posc,R0)),('CRC',crc_pick(posc,a)),('LTT',ltt_pick(posc,a))):
            r,f=evaln(dtst,tau); rows.append(dict(fam=fam,seed=seed,arm=arm,R0=R0,method=name,tau=round(float(tau),4),test_recall=round(r,4),test_FPR=round(f,4)))
res=pd.DataFrame(rows)
print('=== full-calib results (test realized recall / FPR) ===')
piv=res.pivot_table(index=['fam','arm','R0','method'],values=['test_recall','test_FPR'],aggfunc=['mean','std'])
print(piv.round(4).to_string())
# sanity vs gonogo
m=pd.read_csv(B+'/formal_results/stage5/stage5_repeat_metrics.csv')
chk=m[(m.seed==11)&(m.arm=='baseline')&(m.target_recall==0.90)][['threshold','actual_recall','FPR']].iloc[0]
ours=res[(res.fam=='yolo')&(res.seed==11)&(res.arm=='baseline')&(res.R0==0.90)&(res.method=='PR')].iloc[0]
print('\nsanity gonogo seed11 baseline R0.90:', dict(chk), '| ours:', dict(tau=ours.tau,recall=ours.test_recall,FPR=ours.test_FPR))
# group heterogeneity at global PR tau (yolo hardneg seed11..33 mean)
print('\n=== per-class realized recall at global PR tau (R0.90) ===')
for fam,seeds in (('yolo',(11,22,33)),('rtdetr',(11,22))):
    for arm in ('baseline','hardneg'):
        accs={'smoke':[],'fire':[],'FPR':[]}
        for s in seeds:
            dcal,dtst=cache[(fam,s,arm)]; tau=pr_pick(dcal[dcal.label!='none'].score.values,0.90)
            for cls in ('smoke','fire'):
                accs[cls].append((dtst[dtst.label==cls].score>=tau).mean())
            accs['FPR'].append((dtst[dtst.label=='none'].score>=tau).mean())
        print(fam,arm,'smoke %.4f fire %.4f (gap %.3f) FPR %.4f'%(np.mean(accs['smoke']),np.mean(accs['fire']),np.mean(accs['fire'])-np.mean(accs['smoke']),np.mean(accs['FPR'])))
