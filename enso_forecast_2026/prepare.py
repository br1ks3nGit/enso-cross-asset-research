"""Normalize current-vintage NOAA inputs, preserving units and missing values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; OUT=ROOT/'calculated'; OUT.mkdir(exist_ok=True)
STATES=['Alabama','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming','Hawaii','Alaska']
SEASONS=['DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ']

def oni(version=6):
    soup=BeautifulSoup((RAW/f'oni_v{version}.html').read_text(),'html.parser')
    rows={}
    for tr in soup.find_all('tr'):
        vals=[x.get_text(' ',strip=True) for x in tr.find_all(['td','th'],recursive=False)]
        if vals and len(vals[0])==4 and vals[0].isdigit() and 1950<=int(vals[0])<=2026:
            for month,v in enumerate(vals[1:],1):
                try:value=float(v)
                except ValueError:continue
                if abs(value)>10:continue
                rows[int(vals[0]),month]={'year':int(vals[0]),'month':month,'season':SEASONS[month-1],'oni_c':value,'version':f'ERSSTv{version}'}
    return pd.DataFrame(rows.values()).sort_values(['year','month'])

def temperatures(divisions=False):
    rows=[]
    path=RAW/('division_temperature.txt' if divisions else 'state_temperature.txt')
    for line in path.read_text().splitlines():
        code=int(line[:2] if divisions else line[:3])
        division=int(line[2:4]) if divisions else None
        year=int(line[6:10])
        if not 1<=code<=50:continue
        for month,v in enumerate(line[10:].split(),1):
            f=float(v)
            if f<=-99:continue
            if (year,month)>(2026,8):raise ValueError('Future observation found')
            rows.append({'state_code':code,'state':STATES[code-1],'division':division,'year':year,'month':month,'temperature_c':(f-32)*5/9})
    return pd.DataFrame(rows)

def main():
    summary={}
    for v in [5,6]:
        d=oni(v);d.to_json(OUT/f'oni_v{v}.jsonl',orient='records',lines=True)
        summary[f'oni_v{v}']={'rows':len(d),'latest':d.iloc[-1].to_dict()}
    for divisions in [False,True]:
        d=temperatures(divisions)
        name='division' if divisions else 'state'
        d.to_json(OUT/f'{name}_temperature.jsonl.gz',orient='records',lines=True,compression='gzip')
        summary[name]={'rows':len(d),'states':d.state.nunique(),'regions':len(d[['state','division']].drop_duplicates()),'first_year':int(d.year.min()),'last_year':int(d.year.max())}
    (OUT/'coverage.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
