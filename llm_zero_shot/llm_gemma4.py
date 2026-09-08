import json
import re
import os
import time
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE = '<PLACEHOLDER>'
KEY = '<PLACEHOLDER>'
URL = '<PLACEHOLDER>'
DATA = '<PLACEHOLDER>'
LOG = os.path.join(BASE, 'run_atc_gemma4_calls.jsonl')
RENDER = 'struct'
SIZE_GUIDE = 'none'
OUTPUT_MODE = 'atc'
CAP = 320
WORKERS = 4  # Fixed concurrency, at all times.
MODELS = ['gemma-4']

os.makedirs(BASE, exist_ok=True)
d = pd.read_pickle(DATA)


def vlen(r):
    n = 1
    for i in range(1, 6):
        if isinstance(r.get('icd10_%d' % i), (list, tuple)):
            n += 1
        else:
            break
    return n


d['vlen'] = d.apply(vlen, axis=1)
print('FULL TEST SET n=%d | patients=%d' %
      (len(d), d['hash_hn'].nunique()), flush=True)

SYS = (
    'You are a clinical pharmacology assistant supporting outpatient prescribing at a '
    'tertiary-care hospital in Thailand. Given a patient longitudinal clinical narrative, recommend '
    'the medications the physician is most likely to prescribe AT THIS VISIT.\n'
    'Output ONLY a JSON array of WHO ATC codes truncated to the 4th level (5 characters, '
    'e.g. C10AA, A10BA), ranked MOST to LEAST likely. No prose, no explanation.'
)


def render(r):
    L = [
        f"CURRENT VISIT: {r['sex_']}, age {r['age']}, clinic {r['clinic']} ({r['dep_name']})",
        f"  Diagnoses (ICD-10-TM): {', '.join(r['icd10'])}",
        f"  Clinical note: {' '.join(r['note_token'])}",
        'PRIOR VISITS (most recent first):'
    ]
    for i in range(1, 6):
        if not isinstance(r.get('icd10_%d' % i), (list, tuple)):
            break
        L.append(
            f"  - age {r['age_%d' % i]}, clinic {r['clinic_%d' % i]}, "
            f"dx {', '.join(r['icd10_%d' % i])}, prescribed {', '.join(r['y_%d' % i])}"
        )
    return '\n'.join(L)


def parse(t):
    t = re.sub(r'^```(?:json)?|```$', '', str(t).strip(), flags=re.M).strip()
    m = re.search(r'\[.*\]', t, re.S)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(0))]
        except Exception:
            pass
    return [x.strip(' "\'-*.') for x in re.split(r'[,\n]', t)
            if x.strip(' "\'-*.')]


done = set()
if os.path.exists(LOG):
    with open(LOG, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('error'):
                continue
            ct = (r.get('usage') or {}).get('completion_tokens', 0)
            # Preserve the original resume rule: rerun responses that reached CAP.
            if ct and ct < CAP:
                done.add((r['model'], int(r['row_idx'])))

print('reusable: %s' % dict(Counter(m for m, _ in done)), flush=True)

lock = threading.Lock()
logf = open(LOG, 'a', encoding='utf-8')
stat = {m: {'n': 0, 'cost': 0.0, 'err': 0} for m in MODELS}
todo = {m: 0 for m in MODELS}
T0 = time.time()


def call(model, idx):
    r = d.loc[idx]
    usr = ('CLINICAL NARRATIVE:\n' + render(r) +
           '\n\nRANKED MEDICATIONS (JSON array of ATC-4 codes):')
    p = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYS},
            {'role': 'user', 'content': usr}
        ],
        'temperature': 0,
        'max_tokens': CAP,
        'seed': 42
    }
    raw = ''
    err = None
    usage = {}
    cost = None
    fin = None
    t0 = time.time()

    for a in range(4):
        try:
            q = urllib.request.Request(
                URL,
                data=json.dumps(p, ensure_ascii=False).encode(),
                headers={
                    'Authorization': 'Bearer ' + KEY,
                    'Content-Type': 'application/json'
                }
            )
            with urllib.request.urlopen(q, timeout=300) as resp:
                cost = resp.headers.get('x-litellm-response-cost')
                j = json.loads(resp.read().decode())
            raw = j['choices'][0]['message']['content']
            usage = j.get('usage', {})
            fin = j['choices'][0].get('finish_reason')
            err = None
            break
        except Exception as e:
            err = repr(e)
            time.sleep(3 * (a + 1))

    rec = {
        'ts': time.strftime('%m-%d %H:%M:%S'),
        'model': model,
        'render': RENDER,
        'size_guide': SIZE_GUIDE,
        'output_mode': OUTPUT_MODE,
        'variant': f'{RENDER}|{SIZE_GUIDE}|atc',
        'row_idx': int(idx),
        'hash_hn': str(r['hash_hn']),
        'vlen': int(r['vlen']),
        'dep': str(r['dep_name']),
        'latency_s': round(time.time() - t0, 2),
        'error': err,
        'system': SYS,
        'user': usr,
        'response_raw': raw,
        'parsed_names': parse(raw) if raw else [],
        'usage': usage,
        'cost': cost,
        'finish_reason': fin,
        'truth': list(r['y'])
    }
    with lock:
        logf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        logf.flush()
        s = stat[model]
        s['n'] += 1
        if cost:
            s['cost'] += float(cost)
        if err:
            s['err'] += 1
        if s['n'] % 1000 == 0:
            el = (time.time() - T0) / 60
            print('  [%-14s] %6d/%6d  %.1f min  %.2f req/s  $%.2f  err %d' % (
                model, s['n'], todo[model], el, s['n'] / (el * 60),
                s['cost'], s['err']), flush=True)


def arm(model):
    idxs = [i for i in d.index if (model, int(i)) not in done]
    todo[model] = len(idxs)
    print('  %-14s -> %d calls' % (model, len(idxs)), flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda i: call(model, i), idxs))
    print('  %-14s ARM COMPLETE  $%.2f  err %d' % (
        model, stat[model]['cost'], stat[model]['err']), flush=True)


try:
    arm(MODELS[0])
finally:
    logf.close()
print('WALLCLOCK %.1f min | total $%.2f' % (
    (time.time() - T0) / 60, sum(s['cost'] for s in stat.values())), flush=True)
print('DONE', flush=True)
