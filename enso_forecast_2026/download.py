"""Audited climate downloads. Credentials come from CDSAPI_RC, never output files."""
from pathlib import Path
import argparse, hashlib, json, os, re, time
from datetime import datetime, timezone
import requests

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'
RAW.mkdir(exist_ok=True)

def audit(record):
    record['retrieved_at']=datetime.now(timezone.utc).isoformat()
    with (ROOT/'download_manifest.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')

def public(name,url):
    p=RAW/name
    if p.exists(): return
    r=requests.get(url,timeout=60);r.raise_for_status();p.write_bytes(r.content)
    audit({'file':name,'source':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(r.content).hexdigest()})
    print(name,p.stat().st_size,flush=True)

def cds(name,request,dataset='seasonal-monthly-single-levels'):
    import cdsapi
    if Path('/private/tmp/enso2026-cdsapirc').exists():
        os.environ.setdefault('CDSAPI_RC','/private/tmp/enso2026-cdsapirc')
    p=RAW/name
    if p.exists():
        print('Cached',name,flush=True);return
    (ROOT/'requests').mkdir(exist_ok=True)
    (ROOT/'requests'/(name+'.json')).write_text(json.dumps({'dataset':dataset,'request':request},indent=2))
    c=cdsapi.Client(timeout=60,retry_max=2,sleep_max=30,quiet=True,progress=False,debug=False)
    print('Request',name,flush=True)
    c.retrieve(dataset,request,str(p)+'.part')
    Path(str(p)+'.part').rename(p)
    audit({'file':name,'dataset':dataset,'request':request,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    print('Downloaded',name,p.stat().st_size,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['public','forecast','hindcast','era5'])
    parser.add_argument('--centre',default='ecmwf')
    parser.add_argument('--system',default='51')
    args=parser.parse_args()
    if args.stage=='public':
        urls={
          'state_temperature.txt':'https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcst-v1.0.0-20260904',
          'division_temperature.txt':'https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcdv-v1.0.0-20260904',
          'divisional-readme.txt':'https://www.ncei.noaa.gov/pub/data/cirs/climdiv/divisional-readme.txt',
          'oni_v6.html':'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/',
          'oni_v5.html':'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v5/',
          'monthly_sstoi.txt':'https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices',
          'enso_discussion.html':'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml',
        }
        for name,url in urls.items(): public(name,url)
        return
    if args.stage=='era5':
        cds('era5_pacific_2026.nc',{'product_type':['monthly_averaged_reanalysis'],'variable':['sea_surface_temperature','10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure'],'year':['2026'],'month':[f'{m:02}' for m in range(1,9)],'time':['00:00'],'area':[10,120,-10,280],'data_format':'netcdf','download_format':'unarchived'},'reanalysis-era5-single-levels-monthly-means')
        return
    req={'originating_centre':args.centre,'system':args.system,'variable':['sea_surface_temperature'],'product_type':['monthly_mean'],'year':['2026'],'month':['09'],'leadtime_month':['1','2','3','4','5'],'area':[5,-170,-5,-120],'data_format':'netcdf'}
    if args.stage=='hindcast':req['year']=[str(y) for y in range(1993,2026)]
    cds(f'{args.centre}_{args.system}_{args.stage}_sst.nc',req)

if __name__=='__main__':main()
