#!/usr/bin/env python3
"""
_trash.py  -  move a file to the OS trash, on Windows, Linux or macOS.

This is the only place in the toolkit that can make a file disappear, so it
is deliberately small, dependency-free, and fails CLOSED: anything it is not
certain it can do reversibly, it refuses to do at all. Every function returns
(ok, detail) rather than raising, and `detail` is meant to be shown verbatim.

It is imported by the generated Recycle-Duplicates script and vendored INTO
that script so the script keeps working if this folder moves.

Backends
  Windows  SHFileOperationW with FOF_ALLOWUNDO - the same call Explorer makes.
           Refuses paths at/over MAX_PATH, where the API silently deletes
           permanently while reporting success (measured; see CHANGES v3.10).
  Linux    the freedesktop.org Trash specification: $XDG_DATA_HOME/Trash for
           files on the home volume, $topdir/.Trash/$uid or $topdir/.Trash-$uid
           for other volumes, with a matching .trashinfo record so the desktop
           can restore them. Refuses to cross a filesystem boundary rather
           than silently turning a rename into a whole-file copy.
  macOS    ~/.Trash. Finder's "Put Back" needs private metadata we cannot
           write, so the file is restorable by hand but not by Put Back -
           this is stated rather than pretended away.
"""
import errno
import os
import sys

__all__ = ['send_to_trash', 'trash_backend_name', 'precheck']


# --------------------------------------------------------------- Windows --
def _win_send(path):
    import ctypes
    import struct
    from ctypes import wintypes

    _IS64 = struct.calcsize('P') == 8

    class SHFILEOPSTRUCTW(ctypes.Structure):
        # shellapi.h wraps the body in <pshpack1.h> ONLY when !_WIN64, so the
        # struct is 1-byte packed on 32-bit and naturally aligned on x64.
        # FILEOP_FLAGS is a WORD (2 bytes), and hNameMappings is a pointer.
        # The widely-copied declaration gets both wrong; it survives on x64
        # only because the fields it misplaces are zero.
        if not _IS64:
            _pack_ = 1
        _fields_ = [('hwnd', wintypes.HWND),
                    ('wFunc', wintypes.UINT),
                    ('pFrom', wintypes.LPCWSTR),
                    ('pTo', wintypes.LPCWSTR),
                    ('fFlags', ctypes.c_ushort),
                    ('fAnyOperationsAborted', wintypes.BOOL),
                    ('hNameMappings', ctypes.c_void_p),
                    ('lpszProgressTitle', wintypes.LPCWSTR)]

    FO_DELETE = 3
    FOF_SILENT = 0x0004
    FOF_NOCONFIRMATION = 0x0010
    FOF_ALLOWUNDO = 0x0040          # this is what means "Recycle Bin"
    FOF_NOERRORUI = 0x0400

    full = os.path.abspath(path)
    ok, why = _win_precheck(full)
    if not ok:
        return False, why

    op = SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    # pFrom is a double-NUL-terminated list; ctypes adds one NUL, we add the
    # second. The buffer is counted in UTF-16 code units, so an astral
    # character (an emoji in a filename) must not be measured with len().
    op.pFrom = full + '\0'
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    op.fAnyOperationsAborted = 0
    op.hNameMappings = None
    op.lpszProgressTitle = None

    shell32 = ctypes.WinDLL('shell32', use_last_error=True)
    shell32.SHFileOperationW.argtypes = [ctypes.POINTER(SHFILEOPSTRUCTW)]
    shell32.SHFileOperationW.restype = ctypes.c_int
    rc = shell32.SHFileOperationW(ctypes.byref(op))
    if rc != 0:
        # Non-zero codes are pre-Win32 values that only partly overlap
        # winerror.h; report them opaquely rather than mistranslating.
        return False, 'shell delete failed (code 0x%X)' % rc
    if op.fAnyOperationsAborted:
        return False, 'the shell reported the operation was aborted'
    if os.path.exists(full):
        return False, 'the shell reported success but the file is still there'
    return True, 'Recycle Bin'


def _win_long_path_limit():
    return 259          # MAX_PATH (260) minus the terminating NUL


def _win_expand(path):
    """Resolve 8.3 short components - they make a path longer than it looks,
    and the shell measures the expanded form."""
    try:
        import ctypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetLongPathNameW.restype = ctypes.c_uint
        buf = ctypes.create_unicode_buffer(32768)
        n = k32.GetLongPathNameW(os.path.abspath(path), buf, 32768)
        if n and n < 32768:
            return buf.value
    except Exception:
        pass
    return os.path.abspath(path)


_BIN_CACHE = {}


def _win_volume_has_bin(drive):
    """True if the volume actually has a usable Recycle Bin. Removable and
    FAT-formatted drives often do not, and the shell delete call does NOT
    fail there - it silently degrades to PERMANENT deletion and still
    reports success. Same failure family as the long-path case."""
    got = _BIN_CACHE.get(drive)
    if got is not None:
        return got
    ok = False
    try:
        import ctypes

        class SHQUERYRBINFO(ctypes.Structure):
            _fields_ = [('cbSize', ctypes.c_ulong),
                        ('i64Size', ctypes.c_longlong),
                        ('i64NumItems', ctypes.c_longlong)]

        q = SHQUERYRBINFO()
        q.cbSize = ctypes.sizeof(SHQUERYRBINFO)
        shell32 = ctypes.WinDLL('shell32')
        ok = shell32.SHQueryRecycleBinW(drive + '\\', ctypes.byref(q)) == 0
    except Exception:
        ok = False                    # cannot prove a bin exists: refuse
    _BIN_CACHE[drive] = ok
    return ok


def _win_precheck(path):
    full = _win_expand(path)
    if full.startswith('\\\\'):
        return False, ('it is on a network path; Windows cannot send network '
                       'files to the Recycle Bin - deleting would be '
                       'PERMANENT, so it is refused. Delete it on the machine '
                       'that owns the share')
    lim = _win_long_path_limit()
    if len(full) > lim:
        return False, ('path is %d characters; Windows cannot recycle past %d - '
                       'it would delete the file PERMANENTLY and report success'
                       % (len(full), lim))
    drive = os.path.splitdrive(full)[0]
    if drive and not _win_volume_has_bin(drive):
        return False, ('volume %s has no usable Recycle Bin (removable and '
                       'FAT-formatted drives often do not) - the delete call '
                       'would destroy the file PERMANENTLY while reporting '
                       'success, so it is refused. Move the file to a drive '
                       'with a Recycle Bin, or delete it yourself knowingly'
                       % drive)
    return True, ''


# ----------------------------------------------------------------- Linux --
def _fd_quote(path_bytes):
    """RFC 2396 escaping, byte-wise, exactly as glib and trash-cli do.
    Bytes are passed (not str) so undecodable filenames survive."""
    from urllib.parse import quote
    return quote(path_bytes, safe='/')


def _fd_trash_dirs(path):
    """Candidate trash directories for `path`, in spec order. Returns a list
    of (trash_dir, relative_to) where relative_to is None for the home
    trash (which records absolute paths)."""
    home = os.environ.get('XDG_DATA_HOME') or os.path.join(
        os.path.expanduser('~'), '.local', 'share')
    home_trash = os.path.join(home, 'Trash')

    parent = os.path.realpath(os.path.dirname(os.path.normpath(path)))
    try:
        dev = os.lstat(parent).st_dev
    except OSError:
        dev = None
    try:
        home_dev = os.stat(os.path.dirname(home_trash)).st_dev
    except OSError:
        try:
            os.makedirs(os.path.dirname(home_trash), exist_ok=True)
            home_dev = os.stat(os.path.dirname(home_trash)).st_dev
        except OSError:
            home_dev = None

    if dev is not None and home_dev is not None and dev == home_dev:
        return [(home_trash, None)]

    # Different volume: walk up to the mount point that holds the file.
    top = parent
    while True:
        nxt = os.path.dirname(top)
        if nxt == top:
            break
        try:
            if os.lstat(nxt).st_dev != dev:
                break
        except OSError:
            break
        top = nxt

    out = []
    uid = os.getuid()
    admin = os.path.join(top, '.Trash')
    try:
        st = os.lstat(admin)
        import stat as _stat
        # MUST be a real directory, MUST have the sticky bit, MUST NOT be a
        # symlink. If any check fails the spec says fall through to method 2.
        if _stat.S_ISDIR(st.st_mode) and (st.st_mode & _stat.S_ISVTX) \
                and not _stat.S_ISLNK(st.st_mode):
            out.append((os.path.join(admin, str(uid)), top))
    except OSError:
        pass
    out.append((os.path.join(top, '.Trash-%d' % uid), top))
    return out


def _fd_unique(trash_dir, name):
    """Claim files/<name> and info/<name>.trashinfo together, atomically.
    Uniqueness is won by O_EXCL on the info file - a look-then-create is a
    race the spec explicitly forbids."""
    files_dir = os.path.join(trash_dir, 'files')
    info_dir = os.path.join(trash_dir, 'info')
    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    stem, ext = os.path.splitext(name)
    for i in range(0, 10000):
        cand = name if i == 0 else '%s_%d%s' % (stem, i, ext)
        info_path = os.path.join(info_dir, cand + '.trashinfo')
        if os.path.lexists(os.path.join(files_dir, cand)):
            # a stray file with no info record: never rename on top of it
            continue
        try:
            fd = os.open(info_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except OSError as e:
            if e.errno == errno.EEXIST:
                continue
            if e.errno == errno.ENAMETOOLONG and len(stem) > 16:
                stem = stem[:len(stem) // 2]
                continue
            raise
        return cand, fd, os.path.join(files_dir, cand), info_path
    raise OSError('could not find a free name in %s' % trash_dir)


def _linux_send(path):
    import datetime
    norm = os.path.normpath(path)
    # Never resolve the final component: if the user marked a symlink, the
    # symlink is what goes to the trash and its target is untouched.
    base = os.path.basename(norm)
    parent = os.path.realpath(os.path.dirname(norm))
    orig = os.path.join(parent, base)

    last = 'no usable trash directory'
    for trash_dir, rel_to in _fd_trash_dirs(orig):
        try:
            os.makedirs(trash_dir, exist_ok=True)
        except OSError as e:
            last = 'cannot create %s (%s)' % (trash_dir, e.strerror)
            continue
        try:
            name, fd, dest, info_path = _fd_unique(trash_dir, base)
        except OSError as e:
            last = str(e)
            continue

        if rel_to is None:
            recorded = orig
        else:
            pre = rel_to.rstrip('/') + '/'
            recorded = orig[len(pre):] if orig.startswith(pre) else orig
        stamp = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        body = ('[Trash Info]\nPath=%s\nDeletionDate=%s\n'
                % (_fd_quote(os.fsencode(recorded)), stamp))
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(body.encode('utf-8'))
        except OSError as e:
            last = 'cannot write %s (%s)' % (info_path, e.strerror)
            continue
        try:
            os.rename(orig, dest)
        except OSError as e:
            try:
                os.unlink(info_path)        # do not leave a dangling record
            except OSError:
                pass
            if e.errno == errno.EXDEV:
                # Refuse rather than copy: a copy would silently turn an
                # instant metadata move into a whole-file duplication, can
                # fill the home partition, and is not what "move to trash"
                # promises. glib refuses here too.
                last = ('it is on a different filesystem from %s; refusing to '
                        'copy it there' % trash_dir)
                continue
            last = 'rename failed (%s)' % e.strerror
            continue
        return True, trash_dir
    return False, last


# ----------------------------------------------------------------- macOS --
def _macos_send(path):
    norm = os.path.normpath(path)
    base = os.path.basename(norm)
    parent = os.path.realpath(os.path.dirname(norm))
    orig = os.path.join(parent, base)
    trash = os.path.join(os.path.expanduser('~'), '.Trash')
    try:
        os.makedirs(trash, exist_ok=True)
    except OSError as e:
        return False, 'cannot create %s (%s)' % (trash, e.strerror)
    stem, ext = os.path.splitext(base)
    for i in range(0, 10000):
        cand = base if i == 0 else '%s %d%s' % (stem, i, ext)
        dest = os.path.join(trash, cand)
        if os.path.lexists(dest):
            continue
        try:
            os.rename(orig, dest)
        except OSError as e:
            if e.errno == errno.EEXIST:
                continue
            if e.errno == errno.EXDEV:
                return False, ('it is on a different volume from ~/.Trash; '
                               'refusing to copy it there')
            return False, 'rename failed (%s)' % e.strerror
        return True, trash
    return False, 'could not find a free name in %s' % trash


# ------------------------------------------------------------------ api --
def trash_backend_name():
    if sys.platform == 'win32':
        return 'Recycle Bin'
    if sys.platform == 'darwin':
        return '~/.Trash'
    return 'Trash (freedesktop.org)'


def precheck(path):
    """Cheap "would this be refused?" test, so a run can report problems in
    its preview instead of half way through deleting."""
    if sys.platform == 'win32':
        return _win_precheck(path)
    return True, ''


def send_to_trash(path):
    """Returns (ok, detail). Never raises, never deletes permanently."""
    try:
        if not os.path.lexists(path):
            return False, 'file is gone'
        if sys.platform == 'win32':
            return _win_send(path)
        if sys.platform == 'darwin':
            return _macos_send(path)
        return _linux_send(path)
    except Exception as e:
        return False, '%s: %s' % (type(e).__name__, str(e)[:160])


if __name__ == '__main__':
    print('backend: ' + trash_backend_name())
    for p in sys.argv[1:]:
        ok, detail = send_to_trash(p)
        print(('  trashed  %s  -> %s' if ok else '  REFUSED  %s  (%s)')
              % (p, detail))
