#!/usr/bin/env python3
"""
Run profiling for target dashboard scripts across phases and summarize results.
Saves outputs into profiling/ directory.
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path

TARGETS = [
    ("dashboard.py", "dashboard"),
    ("_legacy_dashboard_ref.py", "legacy_dashboard"),
]
PHASES = ['import', 'init', 'dataset']
OUTDIR = 'profiling'


def run_worker(target_path, name, phase, outdir):
    cmd = [sys.executable, os.path.join(outdir, 'worker.py'), '--target', target_path, '--phase', phase, '--outdir', outdir, '--name', name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'Worker failed for {name} {phase}:', proc.stderr, file=sys.stderr)
    # worker prints json path on success
    out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''
    return proc.returncode, out, proc.stdout, proc.stderr


def main():
    p = Path(OUTDIR)
    p.mkdir(exist_ok=True)

    results = []
    for target, name in TARGETS:
        tpath = Path(target)
        if not tpath.exists():
            print(f'Target {target} not found, skipping.', file=sys.stderr)
            continue
        for phase in PHASES:
            print(f'Profiling {name} phase={phase}...')
            rc, out, stdout, stderr = run_worker(str(tpath.resolve()), name, phase, OUTDIR)
            if out:
                try:
                    with open(out, 'r', encoding='utf-8') as f:
                        j = json.load(f)
                except Exception:
                    j = {'error': 'failed to read result', 'stdout': stdout, 'stderr': stderr}
            else:
                j = {'error': 'worker failed', 'returncode': rc, 'stdout': stdout, 'stderr': stderr}
            results.append(j)
            time.sleep(0.5)

    # write summary
    summary_json = Path(OUTDIR) / 'summary.json'
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # human readable summary
    summary_txt = Path(OUTDIR) / 'summary.txt'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"Target: {r.get('name')} phase: {r.get('phase')}\n")
            f.write(f"  elapsed: {r.get('elapsed_seconds')}s rss: {r.get('rss_bytes')} bytes\n")
            if r.get('error'):
                f.write(f"  error: {r.get('error')}\n")
            f.write('\n')

    print('Profiling complete. Summary written to', summary_json)


if __name__ == '__main__':
    main()
