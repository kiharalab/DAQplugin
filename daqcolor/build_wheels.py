#!/usr/bin/env python3
"""
Build the ChimeraX-DAQplugin wheel using ChimeraX's bundled Python.

Since the bundle is pure Python and all backend selection is handled by
PEP 508 environment markers in pyproject.toml, a single py3-none-any wheel
covers Linux, Windows, and macOS. pip picks the correct ONNX Runtime
variant (and MLX on Apple Silicon) at install time.

Usage:
    python build_wheels.py [--chimerax PATH] [--out DIR]

Output:
    wheels/chimerax_daqplugin-VERSION-py3-none-any.whl
"""

import os
import re
import sys
import glob
import shutil
import subprocess
import argparse
from pathlib import Path


def find_chimerax():
    """Find a ChimeraX executable across platforms and install layouts.

    Checks PATH first, then well-known locations, globbing so that
    version-stamped installs (``ChimeraX-1.11.1.app``, ``/apps/chimerax/1.11``,
    ``C:\\Program Files\\ChimeraX 1.11``) are matched — newest first. No
    version or site path is hard-coded. Returns the path, or None.
    """
    # 1. On PATH (user launcher / `chimerax` symlink).
    for name in ('ChimeraX', 'chimerax'):
        found = shutil.which(name)
        if found:
            return found

    # 2. Platform install locations (literal paths + globs).
    if sys.platform == 'darwin':
        patterns = ['/Applications/ChimeraX*.app/Contents/bin/ChimeraX']
    elif sys.platform.startswith('linux'):
        patterns = [
            '/usr/local/bin/chimerax', '/usr/bin/chimerax',
            '/opt/UCSF/ChimeraX*/bin/chimerax',
            '/apps/chimerax/*/usr/bin/chimerax',
        ]
    elif sys.platform.startswith('win'):
        patterns = [
            r'C:\Program Files\ChimeraX*\bin\ChimeraX*.exe',
            r'C:\Program Files\ChimeraX*\bin\chimerax*.exe',
        ]
    else:
        patterns = []

    for pat in patterns:
        if glob.escape(pat) != pat:  # contains a wildcard
            for m in sorted(glob.glob(pat), key=_cx_version_key, reverse=True):
                if os.path.exists(m):
                    return m
        elif os.path.exists(pat):
            return pat
    return None


def _looks_like_python(path):
    """True if `path` is an executable named python / python3 / python3.X."""
    base = os.path.basename(path)
    ok_name = bool(re.match(r'^python(3(\.\d+)?)?$', base)) or base.lower() == 'python.exe'
    return ok_name and os.path.isfile(path) and os.access(path, os.X_OK)


def _cx_version_key(path):
    """Sort key = the ChimeraX version embedded in a path (newest = largest).

    Matches the version right after `chimerax/` or `ChimeraX-` so unrelated
    numbers (python3.11, manylinux_2_28) don't skew ordering. A lexical sort
    would wrongly rank 1.9 above 1.11; this returns (1,9,0) < (1,11,1).
    """
    m = re.search(r'chimerax[/\-](\d+)\.(\d+)(?:\.(\d+))?', path, re.IGNORECASE)
    if not m:
        return (0, 0, 0)
    return tuple(int(x) if x else 0 for x in m.groups())


def find_chimerax_python(chimerax_path):
    """Find ChimeraX's bundled Python, derived from the executable location.

    Handles the macOS .app framework layout, the Linux ucsf-chimerax tree, and
    Windows — without hard-coding a version or site path.
    """
    # 1. Derive from the executable's real location. Works for a real .app /
    #    Linux install tree; fails for a thin launcher wrapper (e.g.
    #    /usr/local/bin/ChimeraX -> /scratch/.../chimerax) that hides the tree.
    if chimerax_path and os.path.exists(chimerax_path):
        real = os.path.realpath(chimerax_path)
        bindir = os.path.dirname(real)
        patterns = []
        if sys.platform == 'darwin':
            # .../ChimeraX.app/Contents/bin/ChimeraX ->
            #   .../Contents/Library/Frameworks/Python.framework/Versions/<v>/bin/python3*
            contents = os.path.dirname(bindir)
            patterns.append(os.path.join(
                contents, 'Library', 'Frameworks', 'Python.framework',
                'Versions', '*', 'bin', 'python3*'))
        patterns += [
            os.path.join(bindir, 'python3*'),
            os.path.join(bindir, 'python.exe'),
            os.path.join(os.path.dirname(bindir), 'lib', 'python3*', 'bin', 'python3*'),
        ]
        for pat in patterns:
            for cand in sorted(glob.glob(pat), key=_cx_version_key, reverse=True):
                if _looks_like_python(cand):
                    return cand

    # 2. Fallback: scan standard ChimeraX install roots directly (newest
    #    first), so a launcher wrapper that hides the tree still resolves.
    if sys.platform == 'darwin':
        roots = ['/Applications/ChimeraX*.app/Contents/Library/Frameworks/'
                 'Python.framework/Versions/*/bin/python3*']
    elif sys.platform.startswith('linux'):
        roots = [
            '/apps/chimerax/*/usr/lib/ucsf-chimerax/bin/python3*',
            '/opt/UCSF/ChimeraX*/lib/python3*/bin/python3*',
            '/usr/lib/ucsf-chimerax/bin/python3*',
        ]
    elif sys.platform.startswith('win'):
        roots = [r'C:\Program Files\ChimeraX*\bin\python.exe']
    else:
        roots = []
    for pat in roots:
        for cand in sorted(glob.glob(pat), key=_cx_version_key, reverse=True):
            if _looks_like_python(cand):
                return cand
    return None


def build_wheel(python_exe, bundle_dir, dist_dir):
    """Build the wheel using ChimeraX's bundled Python (which has BundleBuilder)."""
    cmd = [python_exe, '-m', 'build', '--wheel', '--no-isolation',
           '-v', '--outdir', str(dist_dir), str(bundle_dir)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=bundle_dir)

    if result.returncode != 0:
        print("Trying fallback: pip wheel --no-build-isolation")
        cmd = [python_exe, '-m', 'pip', 'wheel', '--no-deps',
               '--no-build-isolation', '-v', '--wheel-dir', str(dist_dir),
               str(bundle_dir)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=bundle_dir)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Build the ChimeraX-DAQplugin wheel (single py3-none-any).')
    parser.add_argument('--chimerax', '-x', default=None,
                        help='Path to ChimeraX executable (auto-detected if omitted)')
    parser.add_argument('--out', '-o', default='wheels',
                        help='Output directory for the built wheel (default: wheels)')
    args = parser.parse_args()

    chimerax = args.chimerax or find_chimerax()
    if not chimerax or not os.path.exists(chimerax):
        print("Error: ChimeraX not found. Use --chimerax to specify path.")
        return 1

    python_exe = find_chimerax_python(chimerax)
    if not python_exe:
        print(f"Error: Could not find ChimeraX's Python executable.")
        print(f"ChimeraX path: {chimerax}")
        return 1

    bundle_dir = Path(__file__).parent.absolute()
    dist_dir = bundle_dir / 'dist'
    out_dir = Path(args.out) if os.path.isabs(args.out) else bundle_dir / args.out

    print("=" * 60)
    print("Building ChimeraX-DAQplugin wheel")
    print("=" * 60)
    print(f"ChimeraX: {chimerax}")
    print(f"Python:   {python_exe}")
    print(f"Output:   {out_dir}")
    print("=" * 60)

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    build_dir = bundle_dir / 'build'
    if build_dir.exists():
        shutil.rmtree(build_dir)

    if not build_wheel(python_exe, str(bundle_dir), dist_dir):
        print("Error: wheel build failed.")
        return 1

    wheels = list(dist_dir.glob('*.whl'))
    if not wheels:
        print("Error: no wheel produced.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / wheels[0].name
    shutil.move(str(wheels[0]), str(final_path))

    print("\n" + "=" * 60)
    print(f"Built: {final_path}")
    print("Install with: chimerax --cmd 'toolshed install <path-to-wheel>'")
    return 0


if __name__ == '__main__':
    sys.exit(main())
