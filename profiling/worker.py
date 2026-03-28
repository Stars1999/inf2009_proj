#!/usr/bin/env python3
"""
Worker script that profiles a single target file and phase using tracemalloc, psutil, and PyInstrument.
Writes outputs to specified outdir: JSON metrics, tracemalloc text, pyinstrument HTML, and stderr logs.
"""
import argparse
import json
import os
import sys
import time
import traceback

try:
    import tracemalloc
    import psutil
    from pyinstrument import Profiler
except Exception:
    sys.stderr.write('Missing required profiling packages. Please install psutil and pyinstrument.\n')
    raise

import importlib.util


def import_module_from_path(path):
    spec = importlib.util.spec_from_file_location("_target_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INIT_CANDIDATES = ['init', 'initialize', 'setup', 'start', 'main', 'run']
DATA_CANDIDATES = ['load_dataset', 'load_data', 'prepare_dataset', 'get_dataset', 'load']


def find_and_call(mod, phase):
    candidates = INIT_CANDIDATES if phase == 'init' else DATA_CANDIDATES
    for name in candidates:
        if hasattr(mod, name) and callable(getattr(mod, name)):
            func = getattr(mod, name)
            try:
                func()
                return True, None
            except Exception as e:
                return False, traceback.format_exc()
    return False, 'No candidate function found for phase: {}'.format(phase)


def snapshot_top_traces(n=20):
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')[:n]
    out = []
    for stat in stats:
        tb = '\n'.join(stat.traceback.format())
        out.append({'size_bytes': stat.size, 'count': stat.count, 'traceback': tb})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, help='Path to target python file')
    parser.add_argument('--phase', required=True, choices=['import', 'init', 'dataset'])
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--name', required=True, help='Short name for target')
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    result = {
        'target': args.target,
        'name': args.name,
        'phase': args.phase,
        'timestamp': time.time(),
        'elapsed_seconds': None,
        'rss_bytes': None,
        'error': None,
        'tracemalloc_top': None,
        'pyinstrument_html': None,
    }

    tracemalloc.start()
    proc = psutil.Process(os.getpid())

    profiler = Profiler()

    start = time.perf_counter()
    try:
        profiler.start()
        # Always import the module for import phase; for other phases we'll try to call a candidate function
        mod = import_module_from_path(os.path.abspath(args.target))
        if args.phase in ('init', 'dataset'):
            ok, err = find_and_call(mod, args.phase)
            if not ok:
                # record that we didn't find a function but not fatal
                result['error'] = err
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        try:
            profiler.stop()
        except Exception:
            pass
        end = time.perf_counter()

    result['elapsed_seconds'] = end - start
    try:
        result['rss_bytes'] = proc.memory_info().rss
    except Exception:
        result['rss_bytes'] = None

    # tracemalloc top
    try:
        result['tracemalloc_top'] = snapshot_top_traces(20)
    except Exception:
        result['tracemalloc_top'] = None

    # write pyinstrument html
    try:
        html = profiler.output_html()
        html_fname = os.path.join(outdir, f"{args.name}_{args.phase}_pyinstrument.html")
        with open(html_fname, 'w', encoding='utf-8') as f:
            f.write(html)
        result['pyinstrument_html'] = html_fname
    except Exception:
        result['pyinstrument_html'] = None

    # write tracemalloc text
    try:
        tm_fname = os.path.join(outdir, f"{args.name}_{args.phase}_tracemalloc.json")
        with open(tm_fname, 'w', encoding='utf-8') as f:
            json.dump(result.get('tracemalloc_top'), f, indent=2)
    except Exception:
        pass

    # write result json
    try:
        res_fname = os.path.join(outdir, f"{args.name}_{args.phase}_result.json")
        with open(res_fname, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        # also print path for caller
        print(res_fname)
    except Exception:
        print('Failed to write result', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
