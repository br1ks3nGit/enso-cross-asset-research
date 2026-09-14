"""Download cached historical inputs for the ENSO study; never logs credentials."""
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone
import requests

OUT=Path(__file__).resolve().parent
WORKSPACE=OUT.parent
RAW=OUT/'raw'
RAW.mkdir(parents=True,exist_ok=True)
CONFIG=json.loads((WORKSPACE/'config.json').read_text())
BASE='https://api.massive.com'
START=CONFIG['start']
END=CONFIG['end']

class Download:
    def __init__(self):
        self.session=requests.Session()
        self.last=0
        self.audit=[]

    def get(self,label,path,params):
        file=RAW/(label+'.json')
        if file.exists(): return json.loads(file.read_text())
        all_rows=[]
        url=BASE+path
        original=params.copy()
        for page in range(1000):
            time.sleep(max(0,12.1-(time.monotonic()-self.last)))
            self.last=time.monotonic()
            for attempt in range(4):
                response=self.session.get(url,params=params,headers={'Authorization':'Bearer '+CONFIG['api_key']},timeout=40)
                if response.status_code==429 or response.status_code>=500:
                    time.sleep(15*(attempt+1));continue
                break
            if response.status_code!=200:
                result={'status':'error','http_status':response.status_code,'path':path,'params':original,'results':[]}
                break
            body=response.json()
            rows=body.get('results',[])
            if isinstance(rows,dict): rows=[rows]
            all_rows.extend(rows)
            nxt=body.get('next_url')
            if not nxt:
                result={'status':body.get('status'),'path':path,'params':original,'results':all_rows,'retrieved_at':datetime.now(timezone.utc).isoformat()}
                break
            from urllib.parse import urlsplit,parse_qsl,urlencode,urlunsplit
            parts=urlsplit(nxt)
            if parts.scheme!='https' or parts.netloc!='api.massive.com': raise ValueError('Invalid pagination host')
            url=urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode([(k,v) for k,v in parse_qsl(parts.query) if k.lower()!='apikey']),''))
            params=None
        else: raise RuntimeError('Pagination exceeded bound')
        file.write_text(json.dumps(result))
        print(label,result.get('status'),'HTTP',result.get('http_status',200),'rows',len(result['results']),flush=True)
        return result

    def public(self,name,url):
        file=RAW/name
        if file.exists():return
        r=self.session.get(url,timeout=45)
        r.raise_for_status()
        file.write_bytes(r.content)
        print(name,'bytes',len(r.content),flush=True)

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['probe','stocks','futures','public_futures','proxies','actions'],default='probe')
    args=parser.parse_args()
    d=Download()
    if args.stage=='probe':
        d.get('rnr_full','/v2/aggs/ticker/RNR/range/1/day/'+START+'/'+END,{'adjusted':'true','sort':'asc','limit':50000})
        d.get('products_current','/futures/v1/products',{'limit':1000})
        d.public('weekly_nino34.txt','https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for')
        d.public('oni.txt','https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt')
    elif args.stage=='stocks':
        for ticker in ['RNR','RE','EG','ACGL','SPY']:
            d.get('equity_'+ticker,'/v2/aggs/ticker/'+ticker+'/range/1/day/'+START+'/'+END,{'adjusted':'true','sort':'asc','limit':50000})
    elif args.stage=='futures':
        d.get('corn_contracts','/futures/v1/contracts',{'product_code':'ZC','last_trade_date.gte':'2017-04-03','first_trade_date.lte':END,'limit':1000})
        d.get('corn_bar_access','/futures/v1/aggs/ZCZ6',{'resolution':'1session','window_start.gte':'2026-08-01','window_start.lt':'2026-09-01','limit':1000})
    elif args.stage=='public_futures':
        from urllib.parse import quote
        from datetime import timedelta
        p1=int(datetime.fromisoformat(START).replace(tzinfo=timezone.utc).timestamp())
        p2=int((datetime.fromisoformat(END).replace(tzinfo=timezone.utc)+timedelta(days=1)).timestamp())
        for symbol in ['ZC=F','ZS=F','ZW=F','ZM=F','ZL=F','ZO=F','ZR=F','KE=F','LE=F','GF=F','HE=F','CC=F','KC=F','CT=F','SB=F','OJ=F']:
            dest=RAW/('yahoo_'+symbol.replace('=F','')+'.json')
            if dest.exists():continue
            url='https://query1.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe='')
            response=d.session.get(url,params={'period1':p1,'period2':p2,'interval':'1d'},headers={'User-Agent':'Mozilla/5.0'},timeout=40)
            if response.status_code==200:
                payload=response.json(); dest.write_text(json.dumps(payload))
                chart=payload.get('chart',{}).get('result') or []
                print(symbol,'rows',len(chart[0].get('timestamp',[])) if chart else 0,flush=True)
            else:
                dest.write_text(json.dumps({'http_status':response.status_code}))
                print(symbol,'HTTP',response.status_code,flush=True)
            time.sleep(1)
    elif args.stage=='proxies':
        for ticker in ['DBA','CORN','SOYB','WEAT']:
            d.get('equity_'+ticker,'/v2/aggs/ticker/'+ticker+'/range/1/day/'+START+'/'+END,{'adjusted':'true','sort':'asc','limit':50000})
    elif args.stage=='actions':
        for ticker in ['RNR','RE','EG','ACGL','SPY','DBA','CORN','SOYB','WEAT']:
            d.get('dividends_'+ticker,'/v3/reference/dividends',{'ticker':ticker,'ex_dividend_date.gte':START,'ex_dividend_date.lte':END,'limit':1000})
            d.get('splits_'+ticker,'/v3/reference/splits',{'ticker':ticker,'execution_date.gte':START,'limit':1000})

if __name__=='__main__':main()
