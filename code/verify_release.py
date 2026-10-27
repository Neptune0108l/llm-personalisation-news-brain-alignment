"""Offline integrity checks only; no model call, statistical rerun or plotting."""
from pathlib import Path
import ast, csv, hashlib, json, re

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=ROOT/'docs/file_manifest.csv'
    with manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    errors=[]
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file():errors.append('Missing: '+row['path']);continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:errors.append('Hash mismatch: '+row['path'])
        if p.stat().st_size!=int(row['bytes']):errors.append('Size mismatch: '+row['path'])
        if p.suffix=='.py':
            try:ast.parse(p.read_text(encoding='utf-8-sig'),filename=row['path'])
            except SyntaxError as e:errors.append(f'Syntax: {row["path"]}: {e}')
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as handle:
                data=list(csv.reader(handle))
            if data and any(len(x)!=len(data[0]) for x in data[1:]):errors.append('Ragged CSV: '+row['path'])
    for p in [ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^\s)]+)\)',p.read_text(encoding='utf-8')):
            if re.match(r'https?://|mailto:|#',target):continue
            if not (p.parent/target).exists():errors.append(f'Broken link: {p.name}: {target}')
    with (ROOT/'docs/figure_table_map.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            for key in ['figures','source_data','code']:
                for target in row[key].split(';'):
                    if target.strip() and not (ROOT/target.strip()).exists():errors.append('Missing map target: '+target)
    report={'status':'passed' if not errors else 'failed','manifest_files':len(rows),'errors':errors,'scope':'hashes, Python syntax, CSV structure and links; no scientific computations'}
    print(json.dumps(report,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
