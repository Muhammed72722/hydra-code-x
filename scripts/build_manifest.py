#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse,json
from hydra.data.provenance import CodeRecord,stable_record_id,write_jsonl
from hydra.data.quality import score_code
EXT_TO_LANG={'.py':'python','.pyi':'python','.js':'javascript','.jsx':'javascript','.ts':'typescript','.tsx':'typescript','.rs':'rust','.go':'go','.java':'java','.c':'c','.h':'c','.cc':'cpp','.cpp':'cpp','.cxx':'cpp','.hpp':'cpp','.cs':'csharp','.rb':'ruby','.php':'php','.swift':'swift','.kt':'kotlin','.kts':'kotlin','.scala':'scala','.sh':'shell','.bash':'shell','.lua':'lua','.dart':'dart','.ex':'elixir','.exs':'elixir','.hs':'haskell','.sql':'sql','.proto':'protobuf','.sol':'solidity'}
def build(root:Path,out:Path,min_score:float)->int:
    rows=[]
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in EXT_TO_LANG: continue
        try:text=p.read_text(encoding='utf-8')
        except UnicodeDecodeError:continue
        q=score_code(text)
        if q.quality_score<min_score:continue
        rows.append(CodeRecord(stable_record_id(str(root),str(p.relative_to(root)),text),text,EXT_TO_LANG[p.suffix.lower()],'UNKNOWN',repo_id=root.name,path=str(p.relative_to(root)),source_uri=str(root),quality_score=q.quality_score))
    write_jsonl(rows,out); print(json.dumps({'records':len(rows),'output':str(out)},indent=2)); return 0
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--output',required=True); ap.add_argument('--min-score',type=float,default=.55); a=ap.parse_args(); return build(Path(a.root),Path(a.output),a.min_score)
if __name__=='__main__': raise SystemExit(main())
