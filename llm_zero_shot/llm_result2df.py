import json
import re
from pathlib import Path

import pandas as pd

D = Path('')
LOG_DIRS = [
    D / 'logs',
    Path(''
         ''),
]
OUTPUT_DIR = D / 'output_llm'
MODEL_IDS = {
    'gemma4': 'gemma-4',
    'medgemma': 'medgemma',
    'typhoon_si': 'typhoon-si-med',
}
ATC4_PATTERN = re.compile(r'^[A-Z]\d\d[A-Z]{2}$')
COLUMNS = [
    'model', 'mode', 'row_idx', 'call_id', 'rank',
    'raw', 'atc4', 'valid', 'is_hit',
]


def export_predictions(log_dirs=LOG_DIRS, output_dir=OUTPUT_DIR):
    files = sorted({
        file.resolve()
        for directory in log_dirs
        for pattern in ('*.jsonl', '*.bak')
        for file in Path(directory).glob(pattern)
        if file.is_file()
    })
    if not files:
        raise FileNotFoundError('No JSONL or backup logs found. Check LOG_DIRS.')

    best = {}
    malformed_lines = 0
    for file in files:
        with file.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get('model') not in MODEL_IDS.values():
                    continue
                # These exports use only the direct ATC-output arm.
                if record.get('output_mode') != 'atc':
                    continue
                if (record.get('error') or record.get('render') != 'struct'
                        or record.get('size_guide') != 'none'):
                    continue
                count = (record.get('usage') or {}).get('completion_tokens', 0)
                if not count:
                    continue
                key = (record['model'], int(record['row_idx']))
                if key not in best or count > best[key][0]:
                    best[key] = (count, record)

    if not best:
        raise ValueError('No eligible ATC-output calls found. Check LOG_DIRS and MODEL_IDS.')
    if malformed_lines:
        print(f'Skipped {malformed_lines:,} malformed JSON lines.')
    print(f'Read {len(files):,} files; selected {len(best):,} unique model/visit calls.')

    rows = {label: [] for label in MODEL_IDS}
    calls = {label: 0 for label in MODEL_IDS}
    empty_calls = {label: 0 for label in MODEL_IDS}
    labels = {model: label for label, model in MODEL_IDS.items()}
    for (model, row_idx), (_, record) in sorted(best.items()):
        label = labels[model]
        calls[label] += 1
        parsed = record.get('parsed_names') or []
        if not isinstance(parsed, list):
            raise ValueError(f'parsed_names must be a list: {model}, row_idx={row_idx}')
        if not parsed:
            empty_calls[label] += 1
        # Ground truth is already present in each LLM log record.
        truth = set(record['truth'])
        call_id = f'{model}|atc|{row_idx}'
        for rank, item in enumerate(parsed, start=1):
            raw = str(item)
            code = raw.strip().upper()
            valid = bool(ATC4_PATTERN.fullmatch(code))
            # This checks code format only, as in the database builder.
            atc4 = code if valid else None
            rows[label].append({
                'model': model,
                'mode': 'atc',
                'row_idx': row_idx,
                'call_id': call_id,
                'rank': rank,
                'raw': raw,
                'atc4': atc4,
                'valid': int(valid),
                'is_hit': int(valid and atc4 in truth),
            })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for label in MODEL_IDS:
        df = pd.DataFrame(rows[label], columns=COLUMNS)
        for column in ('row_idx', 'rank', 'valid', 'is_hit'):
            df[column] = df[column].astype('int64')
        frames[label] = df
        df.to_pickle(output_dir / f'predictions_{label}.pkl')
        df.to_csv(output_dir / f'predictions_{label}.csv', index=False, encoding='utf-8-sig')
        print(f'{label}: {calls[label]:,} calls, {len(df):,} prediction rows; '
              f'{empty_calls[label]:,} empty calls have no item rows.')
        if not calls[label]:
            print(f'WARNING: No eligible calls found for {MODEL_IDS[label]}.')
    print(f'Saved to: {output_dir}')
    return frames


if __name__ == '__main__':
    predictions = export_predictions()
    df_gemma4 = predictions['gemma4']
    df_medgemma = predictions['medgemma']
    df_typhoon_si = predictions['typhoon_si']
