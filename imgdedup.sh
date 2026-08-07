#!/bin/sh
# ----------------------------------------------------------------------
#  imgdedup.sh  -  one launcher for all four stages, on Linux and macOS.
#
#  The Windows side has a .bat per stage because they are double-clicked
#  and dragged onto. A POSIX shell is a different habit: one entry point
#  with a subcommand reads better than four scripts, and it keeps the
#  interpreter choice in a single place.
#
#    ./imgdedup.sh setup                    install what is missing, GPU first
#    ./imgdedup.sh collect  [folder]        scan a folder into an inventory
#    ./imgdedup.sh embed    [inventory|dir] optional CLIP vectors
#    ./imgdedup.sh analyze  [inventory|dir] find duplicates, write the report
#    ./imgdedup.sh doctor                   what can this machine actually do
#
#  setup detects your GPU (NVIDIA / AMD / Intel), asks which PyTorch build
#  you want, shows the exact pip command and waits for a yes. It never
#  installs anything silently. When a stage cannot run, the launcher offers
#  to run it for you.
#
#  Anything after the subcommand is passed straight through, so
#    ./imgdedup.sh analyze ~/Pictures --no-orient
#  works as expected.
#
#  Interpreter order, most trusted first:
#     1. $IMGDEDUP_PYTHON     explicit override, always wins
#     2. .venv beside this script - what setup creates on distros whose
#        Python is externally managed (Arch, Debian 12+, Fedora 38+), where
#        pip refuses to install into the system interpreter at all
#     3. python3 / python on PATH, newest usable first
#  Every candidate is probed FUNCTIONALLY (it must import and call into the
#  package, not merely name it) - a package whose files were deleted but
#  whose directory survived still imports as an empty namespace package,
#  and that has fooled this toolkit before.
# ----------------------------------------------------------------------
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

usage() {
    # print the header comment up to (not including) its closing dashed
    # rule, so the range cannot drift when the header grows
    sed -n '3,/^# ----/p' "$0" | sed '$d' | sed 's/^# \{0,2\}//'
    exit "${1:-1}"
}

[ $# -ge 1 ] || usage 1
CMD=$1
shift

case "$CMD" in
    collect)  SCRIPT=collect-image-inventory.py
              PROBE='from PIL import Image, ImageOps; Image.new("RGB",(2,2)).convert("L")' ;;
    embed)    SCRIPT=embed-images.py
              PROBE='from PIL import Image; import torch, transformers; assert torch.__file__ and transformers.__file__; torch.zeros(1)' ;;
    analyze)  SCRIPT=analyze-inventory.py
              PROBE='import numpy, PIL; from PIL import Image; Image.new("RGB",(2,2))' ;;
    doctor)   SCRIPT=check-image-tools.py
              PROBE='import sys' ;;
    setup)    SCRIPT=_setup.py
              PROBE='import sys' ;;
    -h|--help|help) usage 0 ;;
    *)        printf 'Unknown command: %s\n\n' "$CMD" >&2; usage 1 ;;
esac

if [ ! -f "$HERE/$SCRIPT" ]; then
    printf '[FAIL] %s is missing from %s - keep the toolkit folder together.\n' \
           "$SCRIPT" "$HERE" >&2
    exit 1
fi

PYCMD=""
try_py() {
    command -v "$1" >/dev/null 2>&1 || return 1
    "$1" -c "$PROBE" >/dev/null 2>&1 || return 1
    PYCMD=$1
    return 0
}

if [ -n "${IMGDEDUP_PYTHON:-}" ]; then
    if "$IMGDEDUP_PYTHON" -c "$PROBE" >/dev/null 2>&1; then
        PYCMD=$IMGDEDUP_PYTHON
    else
        printf '[WARN] IMGDEDUP_PYTHON=%s cannot run this stage; trying others.\n' \
               "$IMGDEDUP_PYTHON" >&2
    fi
fi
# A .venv beside this script is how setup gets out of a PEP 668 distro,
# where pip will not touch the system Python. Prefer it over anything on
# PATH: it is the environment we populated on purpose.
#
# For `setup` alone the probe is tightened to require pip. setup's whole
# job is installing, a venv without pip cannot install anything, and the
# ordinary probe ('import sys') would happily accept one - trapping every
# later run inside the very environment that needs repairing.
VPROBE=$PROBE
[ "$CMD" = setup ] && VPROBE='import sys, pip'
if [ -z "$PYCMD" ]; then
    for v in "$HERE/.venv/bin/python" "$HERE/.venv/Scripts/python.exe"; do
        [ -x "$v" ] || continue
        if "$v" -c "$VPROBE" >/dev/null 2>&1; then
            PYCMD=$v
            break
        fi
    done
fi
if [ -z "$PYCMD" ]; then
    for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 \
             python3 python; do
        try_py "$c" && break
    done
fi

if [ -z "$PYCMD" ]; then
    printf '\n[FAIL] no Python here can run "%s".\n\n' "$CMD" >&2
    printf 'Each candidate was asked to actually run:\n    %s\n\n' "$PROBE" >&2
    BASEPY=""
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then
            printf '  %s (%s) said:\n' "$c" "$(command -v "$c")" >&2
            "$c" -c "$PROBE" 2>&1 | sed 's/^/      /' >&2
            [ -z "$BASEPY" ] && BASEPY=$c
        else
            printf '  %s: not found\n' "$c" >&2
        fi
    done
    if [ -n "$BASEPY" ] && [ -f "$HERE/_setup.py" ]; then
        # There IS a Python, it just lacks packages - the one case we can
        # actually fix. setup detects the GPU, asks which build, and shows
        # every command before running it.
        printf '\n' >&2
        printf 'A Python is present; it is the packages that are missing.\n' >&2
        printf 'Run setup now? It will show each command and ask first.\n' >&2
        printf '  [y/N]: ' >&2
        read -r ans </dev/tty 2>/dev/null || ans=""
        case "$ans" in
            [Yy]*) "$BASEPY" "$HERE/_setup.py" || exit 1
                   printf '\nSetup finished - re-run: ./imgdedup.sh %s\n\n' "$CMD" >&2
                   exit 0 ;;
        esac
        printf '\nSkipped. Run it yourself any time:  ./imgdedup.sh setup\n\n' >&2
    else
        printf '\nInstall Python 3 from your package manager, then run:\n' >&2
        printf '  ./imgdedup.sh setup\n\n' >&2
    fi
    exit 1
fi

exec "$PYCMD" "$HERE/$SCRIPT" "$@"
