import requests
from pathlib import Path
url='https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/20160604_FIRE_smer-tcs3-mobo-c.tgz'
out=Path('data/figlib/20160604_FIRE_smer-tcs3-mobo-c.tgz.req')
out.parent.mkdir(parents=True,exist_ok=True)
headers={'User-Agent':'Mozilla/5.0'}
with requests.get(url,headers=headers,stream=True,timeout=30,verify=True) as r:
    r.raise_for_status()
    with out.open('wb') as f:
        for chunk in r.iter_content(chunk_size=1<<20):
            if chunk:
                f.write(chunk)
print(out.stat().st_size)