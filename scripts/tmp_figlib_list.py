import urllib.request
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin

class A(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)

url = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html"
html = urllib.request.urlopen(
    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30
).read().decode("utf-8", errors="replace")
parser = A()
parser.feed(html)
links = [u for u in parser.links if u.lower().endswith('.tgz')]
base = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/"
rows = []
for href in links:
    name = href.rsplit('/', 1)[-1]
    if not name.lower().endswith('.tgz'):
        continue
    target = urljoin(base, href)
    req = urllib.request.Request(target, method='HEAD', headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        size = int(resp.headers.get('Content-Length', '0') or 0)
        status = str(resp.status)
    except Exception as exc:
        size = 0
        status = str(exc)
    rows.append((name, status, size))

out = Path('data/figlib_tgzs.tsv')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', encoding='utf-8') as f:
    f.write('name\tstatus\tsize_bytes\n')
    for name, status, size in rows:
        f.write(f"{name}\t{status}\t{size}\n")

print('count=', len(rows))
print('total_gb=', round(sum(size for _, _, size in rows) / 1024 / 1024 / 1024, 3))