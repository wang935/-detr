import pandas as pd, numpy as np, math
B='/sessions/optimistic-peaceful-bardeen/mnt/detr_Q3'
from scipy.stats import binom
cal=pd.read_csv(B+'/data/dfire_local/eval_calib_labels.csv'); tst=pd.read_csv(B+'/data/dfire_local/eval_test_labels.csv')
def sc(pred,lab):
    df=pd.read_csv(pred); s=df.groupby('image')['conf'].max(); o=lab.copy(); o['score']=o.image.map(s).fillna(0.0); return o
def pr(pos,R0): s=np.sort(pos)[::-1]; return s[math.ceil(R0*len(pos))-1]
def crc(pos,a):
    N=len(pos); m=math.floor(a*(N+1)-1)
    return np.sort(pos)[m] if m>=0 else -np.inf
def ltt(pos,a,d=0.05):
    N=len(pos); arr=np.unique(pos)[::-1]; cand=-np.inf; L,R=0,len(arr)-1
    while L<=R:
        M=(L+R)//2; t=arr[M]; p=binom.cdf(int((pos<t).sum()),N,a)
        if p<=d: cand=t; R=M-1
        else: L=M+1
    return cand
rng=np.random.default_rng(0)
print('=== bootstrap小标定集: 500 reps, 违约率(test recall<R0) & mean FPR ===')
for fam,seed,arm,path in [('yolo',11,'hardneg',B+'/formal_results/stage5/yolo_seed11/hardneg.csv'),('yolo',11,'baseline',B+'/formal_results/stage5/yolo_seed11/baseline.csv'),('rtdetr',11,'hardneg',B+'/formal_results/stage5_pv_v2/dfire_rtdetr_seed11/hardneg.csv')]:
    dc=sc(path,cal); dt=sc(path,tst)
    posc=dc[dc.label!='none'].score.values
    post=dt[dt.label!='none'].score.values; negt=dt[dt.label=='none'].score.values
    for R0 in (0.90,):
        a=1-R0
        for N in (2006,500,200,100):
            V={'PR':[],'CRC':[],'LTT':[]}; F={'PR':[],'CRC':[],'LTT':[]}
            for _ in range(500):
                sub=rng.choice(posc,size=N,replace=False) if N<len(posc) else posc
                for nm,t in (('PR',pr(sub,R0)),('CRC',crc(sub,a)),('LTT',ltt(sub,a))):
                    V[nm].append((post>=t).mean()<R0); F[nm].append((negt>=t).mean())
            print(f'{fam} {arm} R{R0} Ncal={N}: ' + ' | '.join(f'{nm} viol {np.mean(V[nm])*100:5.1f}% FPR {np.mean(F[nm]):.4f}' for nm in ('PR','CRC','LTT')))
print()
print('=== smoke锚定CRC (担保最难组): 全标定集, R0.90 ===')
for fam,seeds,pat in [('yolo',(11,22,33),B+'/formal_results/stage5/yolo_seed%d/%s.csv'),('rtdetr',(11,22),B+'/formal_results/stage5_pv_v2/dfire_rtdetr_seed%d/%s.csv')]:
    for arm in ('baseline','hardneg'):
        rs={'smoke':[],'fire':[],'FPR':[],'tau':[]}
        for s in seeds:
            dc=sc(pat%(s,arm),cal); dt=sc(pat%(s,arm),tst)
            smk=dc[dc.label=='smoke'].score.values
            t=crc(smk,0.10)
            rs['tau'].append(t); rs['smoke'].append((dt[dt.label=='smoke'].score>=t).mean()); rs['fire'].append((dt[dt.label=='fire'].score>=t).mean()); rs['FPR'].append((dt[dt.label=='none'].score>=t).mean())
        print(f"{fam} {arm}: tau {np.mean(rs['tau']):.4f} | smoke recall {np.mean(rs['smoke']):.4f}±{np.std(rs['smoke']):.4f} | fire {np.mean(rs['fire']):.4f} | FPR {np.mean(rs['FPR']):.4f}±{np.std(rs['FPR']):.4f}")
