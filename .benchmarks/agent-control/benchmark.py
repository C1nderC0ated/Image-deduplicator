#!/usr/bin/env python3
"""Prepare, isolate, and verify a human + frontier-agent comparison run.

This deliberately does not score visual judgements. It standardizes the
parts that should not depend on operator style: a Linux-isolated writable
workspace, content fingerprint, fixed agent/model, prompt/output contract,
session transcript, source-tree verification, and manifest shape.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_CORPUS = Path('/home/solteris/Apps/Asset Pack V2')
RUNS_ROOT = Path(os.environ.get(
    'IMAGE_AGENT_BENCH_RUNS',
    str(Path.home() / 'Benchmarks' / 'image-agent-control'))).resolve()
VISIBLE_CORPUS = Path('/corpus')
VISIBLE_WORKSPACE = Path('/work')
RUN_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')
REQUIRED = {'group', 'keeper', 'candidate', 'action', 'relation',
            'confidence', 'evidence'}
ACTIONS = {'delete', 'review'}
RELATIONS = {'exact', 'reencoded', 'resized', 'cropped', 'rotated',
             'mirrored', 'animation', 'other'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.jfif', '.jpe', '.png', '.webp', '.tif',
              '.tiff', '.bmp', '.heic', '.heif', '.hif', '.avif', '.gif',
              '.apng', '.tga', '.qoi'}


def run_paths(name):
    """Return the public workspace and private harness state for one run."""
    if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
        raise SystemExit(
            'Run name must be one safe component (letters, digits, ._-).')
    root = RUNS_ROOT.resolve()
    return root / name, root / '.state' / name


def _file_content(path, original):
    """Hash a regular file and reject replacement/mutation during the read."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        identity = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns')
        if any(getattr(opened, key) != getattr(original, key)
               for key in identity):
            raise OSError('entry changed before content read')
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        if any(getattr(after, key) != getattr(opened, key)
               for key in identity):
            raise OSError('entry changed during content read')
    return digest.hexdigest()


def corpus_fingerprint(root):
    """Hash metadata, content, and symlink identity for every source entry."""
    root = Path(root).resolve()
    entries = []
    walk_errors = []
    for base, dirs, names in os.walk(
            root, followlinks=False, onerror=walk_errors.append):
        dirs.sort()
        names.sort()
        for name in dirs + names:
            path = Path(base, name)
            entries.append((path.relative_to(root).as_posix(), path))
    entries.sort(key=lambda pair: pair[0])

    digest = hashlib.sha256()
    files = total = 0
    errors = len(walk_errors)
    for rel, path in entries:
        try:
            info = path.lstat()
            mode = info.st_mode
            if stat.S_ISREG(mode):
                kind = 'file'
                detail = _file_content(path, info)
                files += 1
                total += info.st_size
            elif stat.S_ISDIR(mode):
                kind, detail = 'dir', ''
            elif stat.S_ISLNK(mode):
                kind, detail = 'symlink', os.readlink(path)
            else:
                kind, detail = 'special', ''
            row = (rel, kind, info.st_size, info.st_mtime_ns,
                   stat.S_IMODE(mode), detail)
        except OSError as exc:
            errors += 1
            row = (rel, 'error', getattr(exc, 'errno', None), str(exc))
        digest.update(json.dumps(row, ensure_ascii=False,
                                 separators=(',', ':')).encode('utf-8'))
        digest.update(b'\n')
    for exc in sorted(walk_errors, key=lambda error: str(error.filename)):
        row = ('walk-error', str(exc.filename), getattr(exc, 'errno', None),
               str(exc))
        digest.update(json.dumps(row, ensure_ascii=False,
                                 separators=(',', ':')).encode('utf-8'))
        digest.update(b'\n')
    return {'kind': 'sha256-content-v1', 'sha256': digest.hexdigest(),
            'entries': len(entries), 'files': files, 'bytes': total,
            'errors': errors}


def _public_state(state):
    public = dict(state)
    public['corpus'] = str(VISIBLE_CORPUS)
    public['workspace'] = str(VISIBLE_WORKSPACE)
    return public


def write_state(run, private, state):
    payload = json.dumps(state, indent=2, ensure_ascii=False) + '\n'
    (private / 'run.json').write_text(payload, encoding='utf-8')
    (run / 'run.json').write_text(
        json.dumps(_public_state(state), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


def prepare(args):
    corpus = args.corpus.resolve()
    run, private = run_paths(args.run)
    if not corpus.is_dir():
        raise SystemExit('Corpus is not a directory: ' + str(corpus))
    if run.exists() or private.exists():
        raise SystemExit('Run already exists: ' + str(run))
    # A workspace inside either tree would alter its own fingerprint or leak
    # the implementation into the comparison.
    if (corpus == PROJECT_ROOT or corpus in PROJECT_ROOT.parents or
            PROJECT_ROOT in corpus.parents):
        raise SystemExit('Corpus must not contain the project repository.')
    if run == corpus or corpus in run.parents or run in corpus.parents:
        raise SystemExit('Run workspace and corpus must be disjoint.')

    print('Hashing source contents before creating the workspace ...',
          flush=True)
    before = corpus_fingerprint(corpus)
    if before['errors']:
        raise SystemExit('Cannot prepare: %d corpus entries could not be hashed.'
                         % before['errors'])

    run.mkdir(parents=True)
    private.mkdir(parents=True)
    for name in ('prompt.md', 'operator-log.md', 'scorecard.md'):
        text = (HERE / name).read_text(encoding='utf-8')
        if name == 'prompt.md':
            text = text.replace(str(DEFAULT_CORPUS), str(VISIBLE_CORPUS))
        elif name == 'operator-log.md':
            text = text.replace('- Run:', '- Run: ' + args.run, 1)
            text = text.replace('- Model and reasoning:',
                                '- Model and reasoning: gpt-5.6-sol / high', 1)
        (run / name).write_text(text, encoding='utf-8')
    state = {
        'schema': 'agent-control/2',
        'run': args.run,
        'corpus': str(corpus),
        'workspace': str(run),
        'prepared_ms': int(time.time() * 1000),
        'before': before,
    }
    write_state(run, private, state)
    print('Prepared:', run)
    print('Agent sees corpus/workspace as: /corpus  /work')
    print('Before:  ', before['sha256'],
          '(%d files, %.3f GiB)' % (before['files'],
                                    before['bytes'] / 1024 ** 3))
    print('Next: python benchmark.py launch ' + args.run)


def load_state(run_name):
    run, private = run_paths(run_name)
    try:
        state = json.loads(
            (private / 'run.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise SystemExit('No prepared run: ' + str(run))
    if state.get('schema') != 'agent-control/2' or state.get('run') != run_name:
        raise SystemExit('Invalid private run state: ' + str(private))
    return run, private, state


def launch(args):
    """Launch the fixed interactive Codex session inside a minimal mount."""
    run, private, state = load_state(args.run)
    bwrap = shutil.which('bwrap')
    codex = shutil.which('codex')
    script = shutil.which('script')
    auth = Path.home() / '.codex' / 'auth.json'
    if not bwrap:
        raise SystemExit('Bubblewrap (bwrap) is required for benchmark isolation.')
    if not codex or not script:
        raise SystemExit('Codex CLI and util-linux script are required.')
    if not auth.is_file():
        raise SystemExit('ChatGPT Codex login not found; run `codex login`.')

    codex_state = private / 'codex-home'
    codex_state.mkdir(mode=0o700, exist_ok=True)
    auth_target = codex_state / 'auth.json'
    auth_target.touch(mode=0o600, exist_ok=True)

    command = [bwrap, '--die-with-parent', '--new-session', '--clearenv',
               '--unshare-pid', '--unshare-ipc', '--unshare-uts',
               '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
               '--tmpfs', '/home', '--dir', '/home/agent']
    for host in ('/usr', '/etc', '/opt', '/var', '/sys'):
        if Path(host).exists():
            command += ['--ro-bind', host, host]
    for link, target in (('/bin', 'usr/bin'), ('/sbin', 'usr/bin'),
                         ('/lib', 'usr/lib'), ('/lib64', 'usr/lib')):
        command += ['--symlink', target, link]
    command += [
        '--ro-bind', str(Path(codex).resolve()), '/codex',
        '--bind', str(run), str(VISIBLE_WORKSPACE),
        '--ro-bind', state['corpus'], str(VISIBLE_CORPUS),
        '--bind', str(codex_state), '/codex-state',
        '--ro-bind', str(auth), '/codex-state/auth.json',
        '--setenv', 'HOME', '/home/agent',
        '--setenv', 'CODEX_HOME', '/codex-state',
        '--setenv', 'PATH', '/usr/local/bin:/usr/bin:/bin',
        '--setenv', 'PWD', str(VISIBLE_WORKSPACE),
        '--setenv', 'USER', 'agent',
        '--setenv', 'LOGNAME', 'agent',
        '--setenv', 'SHELL', '/bin/bash',
        '--setenv', 'TERM', os.environ.get('TERM') or 'xterm-256color',
        '--setenv', 'LANG', os.environ.get('LANG') or 'C.UTF-8',
        '--chdir', str(VISIBLE_WORKSPACE),
    ]
    common = ['--model', 'gpt-5.6-sol',
              '-c', 'model_reasoning_effort="high"',
              '--sandbox', 'workspace-write',
              '--ask-for-approval', 'on-request',
              '-C', str(VISIBLE_WORKSPACE), '--no-alt-screen']
    if args.resume:
        command += ['/codex', 'resume'] + common + [args.resume]
    else:
        prompt = (run / 'prompt.md').read_text(encoding='utf-8')
        command += ['/codex'] + common + [prompt]

    transcript = run / 'terminal.typescript'
    wrapped = [script, '-q', '-f', '-a', '-e', '-c', shlex.join(command),
               str(transcript)]
    if args.dry_run:
        print(shlex.join(wrapped))
        return
    version = subprocess.run([codex, '--version'], check=False,
                             capture_output=True, text=True).stdout.strip()
    event = {'started_ms': int(time.time() * 1000),
             'model': 'gpt-5.6-sol', 'reasoning': 'high',
             'codex_version': version, 'resume': args.resume}
    state.setdefault('launches', []).append(event)
    write_state(run, private, state)
    print('Launching isolated ChatGPT-subscription Codex Sol (high).')
    print('Only /work and read-only /corpus are exposed; transcript:',
          transcript)
    exit_code = None
    try:
        exit_code = subprocess.call(wrapped)
        return exit_code
    finally:
        event['ended_ms'] = int(time.time() * 1000)
        event['exit_code'] = exit_code
        write_state(run, private, state)


def verify(args):
    run, private, state = load_state(args.run)
    print('Hashing source contents after the run ...', flush=True)
    after = corpus_fingerprint(Path(state['corpus']))
    same = after == state['before'] and not after['errors']
    state['after'] = after
    state['verified_ms'] = int(time.time() * 1000)
    state['corpus_unchanged'] = same
    write_state(run, private, state)
    print('Before:', state['before']['sha256'])
    print('After: ', after['sha256'])
    print('Corpus unchanged:', 'YES' if same else 'NO')
    return 0 if same else 1


def safe_relative(value):
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and '..' not in path.parts and
            '\\' not in value)


def corpus_file(corpus, value):
    """A manifest path must resolve to a file inside the source corpus."""
    if not safe_relative(value):
        return False, 'unsafe/non-relative'
    candidate = corpus.joinpath(*PurePosixPath(value).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(corpus)
    except (OSError, ValueError):
        return False, 'missing or escapes corpus'
    return (True, '') if candidate.is_file() else (False, 'not a file')


def validate(args):
    run, _private, state = load_state(args.run)
    corpus = Path(state['corpus']).resolve()
    manifest = run / 'manifest.jsonl'
    errors = []
    identities = set()
    candidates = {}
    group_keepers = {}
    keepers = set()
    delete_candidates = set()
    counts = {action: 0 for action in sorted(ACTIONS)}
    try:
        lines = manifest.read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        raise SystemExit('Missing manifest: ' + str(manifest))

    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except Exception as exc:
            errors.append('%d: invalid JSON (%s)' % (number, exc))
            continue
        if not isinstance(row, dict):
            errors.append('%d: row must be a JSON object' % number)
            continue
        missing = REQUIRED - set(row)
        if missing:
            errors.append('%d: missing %s' %
                          (number, ', '.join(sorted(missing))))
            continue
        for key in ('group', 'evidence'):
            if not isinstance(row[key], str) or not row[key].strip():
                errors.append('%d: %s must be a non-empty string' %
                              (number, key))
        if (not isinstance(row['action'], str) or
                row['action'] not in ACTIONS):
            errors.append('%d: invalid action %r' % (number, row['action']))
        else:
            counts[row['action']] += 1
        if (not isinstance(row['relation'], str) or
                row['relation'] not in RELATIONS):
            errors.append('%d: invalid relation %r' %
                          (number, row['relation']))
        for key in ('keeper', 'candidate'):
            valid, reason = corpus_file(corpus, row[key])
            if not valid:
                errors.append('%d: %s %s path %r' %
                              (number, reason, key, row[key]))
            elif PurePosixPath(row[key]).suffix.lower() not in IMAGE_EXTS:
                errors.append('%d: %s is not a recognized image path %r' %
                              (number, key, row[key]))
        if row['keeper'] == row['candidate']:
            errors.append('%d: keeper equals candidate' % number)
        confidence = row['confidence']
        if (isinstance(confidence, bool) or
                not isinstance(confidence, (int, float)) or
                not 0 <= confidence <= 1):
            errors.append('%d: confidence must be in [0, 1]' % number)

        group, keeper, candidate = (row['group'], row['keeper'],
                                    row['candidate'])
        if all(isinstance(value, str) for value in (group, keeper, candidate)):
            identity = (group, keeper, candidate)
            if identity in identities:
                errors.append('%d: duplicate group/keeper/candidate row' % number)
            identities.add(identity)
            if candidate in candidates:
                errors.append('%d: candidate already has an action on row %d' %
                              (number, candidates[candidate]))
            else:
                candidates[candidate] = number
            previous = group_keepers.setdefault(group, keeper)
            if previous != keeper:
                errors.append('%d: group %r has multiple keepers' %
                              (number, group))
            keepers.add(keeper)
            if row['action'] == 'delete':
                delete_candidates.add(candidate)

    for path in sorted(keepers & delete_candidates):
        errors.append('delete candidate is also used as a keeper: %r' % path)

    print('Rows: %d  delete: %d  review: %d' %
          (len(lines), counts['delete'], counts['review']))
    if errors:
        for error in errors[:50]:
            print('ERROR:', error)
        if len(errors) > 50:
            print('... %d more error(s)' % (len(errors) - 50))
        return 1
    print('Manifest contract: PASS')
    return 0


def score_exact(args):
    """Score byte-identical discovery without using the project pipeline."""
    if validate(args):
        print('Exact score aborted: manifest contract failed.')
        return 1
    run, _private, state = load_state(args.run)
    corpus = Path(state['corpus']).resolve()
    print('Hashing recognized image files for exact-copy ground truth ...',
          flush=True)
    by_path = {}
    by_hash = {}
    unscored = []
    walk_errors = []
    for base, dirs, names in os.walk(
            corpus, followlinks=False, onerror=walk_errors.append):
        dirs.sort()
        for name in sorted(names):
            path = Path(base, name)
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = path.relative_to(corpus).as_posix()
            valid, reason = corpus_file(corpus, rel)
            if not valid:
                unscored.append('%s (%s)' % (rel, reason))
                continue
            try:
                info = path.stat()
                digest = _file_content(path.resolve(), info)
            except OSError as exc:
                unscored.append('%s (%s)' % (rel, exc))
                continue
            by_path[rel] = (digest, info.st_size)
            by_hash.setdefault(digest, []).append(rel)

    unscored.extend('walk error: %s' % error for error in walk_errors)

    if unscored:
        raise SystemExit('Exact ground truth incomplete; %d image(s) could '
                         'not be hashed (first: %s)' %
                         (len(unscored), unscored[0]))

    duplicate_groups = {digest: paths for digest, paths in by_hash.items()
                        if len(paths) > 1}
    denominator = sum(len(paths) - 1 for paths in duplicate_groups.values())
    found = set()
    deleted = set()
    exact_delete_bytes = 0
    delete_rows = nonexact_delete_rows = claimed_exact_wrong = 0
    for line in (run / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        keeper = by_path.get(row['keeper'])
        candidate = by_path.get(row['candidate'])
        same = bool(keeper and candidate and keeper[0] == candidate[0])
        if same:
            found.add(row['candidate'])
            if row['action'] == 'delete':
                deleted.add(row['candidate'])
                exact_delete_bytes += candidate[1]
        if row['action'] == 'delete':
            delete_rows += 1
            if not same:
                nonexact_delete_rows += 1
        if row['relation'] == 'exact' and not same:
            claimed_exact_wrong += 1

    # Review-only cycles can name every path in an N-file exact group as a
    # candidate. Discovery credit is capped at the N-1 redundant files that
    # actually exist, keeping recall at or below 100% without prescribing a
    # particular keeper.
    found_count = sum(min(len(paths) - 1,
                          len(found.intersection(paths)))
                      for paths in duplicate_groups.values())

    result = {
        'schema': 'agent-control-exact-score/1',
        'recognized_images': len(by_path),
        'exact_groups': len(duplicate_groups),
        'exact_duplicate_files': denominator,
        'exact_found': found_count,
        'exact_recall': (found_count / denominator if denominator else 1.0),
        'exact_delete_candidates': len(deleted),
        'exact_delete_bytes': exact_delete_bytes,
        'all_delete_rows': delete_rows,
        'delete_rows_requiring_visual_adjudication': nonexact_delete_rows,
        'incorrect_exact_relation_rows': claimed_exact_wrong,
    }
    (run / 'exact-score.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Recognized images: %d  exact groups: %d  duplicate files: %d' %
          (result['recognized_images'], result['exact_groups'], denominator))
    print('Exact found: %d/%d (%.2f%%)' %
          (found_count, denominator, result['exact_recall'] * 100))
    print('Exact delete candidates: %d (%.3f GiB)' %
          (len(deleted), exact_delete_bytes / 1024 ** 3))
    print('Delete rows requiring visual adjudication:', nonexact_delete_rows)
    print('Incorrect `relation: exact` rows:', claimed_exact_wrong)
    print('Wrote:', run / 'exact-score.json')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('prepare')
    command.add_argument('run')
    command.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    command.set_defaults(func=prepare)
    command = sub.add_parser('launch')
    command.add_argument('run')
    command.add_argument('--resume', metavar='SESSION_ID')
    command.add_argument('--dry-run', action='store_true')
    command.set_defaults(func=launch)
    command = sub.add_parser('verify')
    command.add_argument('run')
    command.set_defaults(func=verify)
    command = sub.add_parser('validate')
    command.add_argument('run')
    command.set_defaults(func=validate)
    command = sub.add_parser('score-exact')
    command.add_argument('run')
    command.set_defaults(func=score_exact)
    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
