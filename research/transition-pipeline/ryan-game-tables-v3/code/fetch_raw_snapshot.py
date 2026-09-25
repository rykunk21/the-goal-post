import concurrent.futures,hashlib,json,pathlib,time,urllib.request
import pandas as pd
import subprocess
from datetime import datetime, timezone

def main():
    parser=__import__('argparse').ArgumentParser()
    parser.add_argument('--previous-college-table',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(args.previous_college_table,columns=['game_id','season','matrix_status'])
    ids=sorted(d.loc[(d.season==2026)&(d.matrix_status!='parsed_regulation'),'game_id'].tolist())
    def fetch(gid):
     url=f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={gid}'
     path=root/f'{gid}.json';row={'game_id':gid,'url':url}
     try:
      if not path.exists():
       data=subprocess.run(['curl','-fsSL','--max-time','40',url],capture_output=True,check=True).stdout
       json.loads(data);path.write_bytes(data)
       time.sleep(.15)
      data=path.read_bytes();row.update(status='downloaded',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data), checked_at=datetime.now(timezone.utc).isoformat())
     except Exception as e:row.update(status='failed',error=str(e))
     return row
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
     for row in pool.map(fetch,ids):
      results.append(row)
      if len(results)%25==0:print(len(results),'/',len(ids),flush=True)
    (root/'receipts.json').write_text(json.dumps(results,indent=2)+'\n')
    print('completed',len(results),'failed',sum(r['status']=='failed' for r in results),flush=True)

if __name__=='__main__':main()
