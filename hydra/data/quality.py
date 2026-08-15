from __future__ import annotations
import re
from dataclasses import dataclass
COMMENT_RE = re.compile(r'(#.*$|//.*$|/\*.*?\*/)', re.MULTILINE | re.DOTALL)
SECRET_RE = re.compile(r'''(?i)(api[_-]?key|secret|password|private[_-]?key)\s*[:=]\s*["'][^"']{8,}["']''')
@dataclass(frozen=True)
class QualitySignals:
    chars: int; lines: int; comment_ratio: float; repeated_line_ratio: float
    secret_like: bool; balanced_delimiters: bool; quality_score: float

def _balanced(text: str) -> bool:
    pairs = {')':'(', ']':'[', '}':'{'}; stack=[]
    for ch in text:
        if ch in '([{': stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]: return False
    return not stack

def score_code(text: str) -> QualitySignals:
    lines = text.splitlines() or ['']; nonempty=[ln.strip() for ln in lines if ln.strip()]
    unique=len(set(nonempty)) if nonempty else 0
    repeated_ratio=1.0-unique/max(1,len(nonempty))
    comments=COMMENT_RE.findall(text); comment_ratio=min(1.0,sum(map(len,comments))/max(1,len(text)))
    secret_like=bool(SECRET_RE.search(text)); balanced=_balanced(text)
    score=1.0
    if len(text)<80: score-=0.25
    if len(text)>1_000_000: score-=0.35
    if repeated_ratio>0.60: score-=0.25
    if secret_like: score-=0.60
    if not balanced: score-=0.20
    if comment_ratio>0.90 and len(text)>500: score-=0.10
    return QualitySignals(len(text),len(lines),comment_ratio,repeated_ratio,secret_like,balanced,max(0.0,min(1.0,score)))
