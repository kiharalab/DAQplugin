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
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def find_chimerax():
    """Find ChimeraX executable."""
    if sys.platform.startswith('linux'):
        candidates = ['/usr/local/bin/chimerax', '/usr/bin/chimerax', shutil.which('chimerax')]
    elif sys.platform == 'darwin':
        candidates = ['/Applications/ChimeraX.app/Contents/bin/ChimeraX']
    else:
        candidates = []

    for c in candidates:
        if c and os.path.exists(c):
            return c
    return shutil.which('chimerax')


def find_chimerax_python(chimerax_path):
    """Find ChimeraX's bundled Python executable."""
    if not chimerax_path or not os.path.exists(chimerax_path):
        return None

    real_path = os.path.realpath(chimerax_path)
    chimerax_dir = os.path.dirname(real_path)

    candidates = [
        os.path.join(chimerax_dir, 'python3.11'),
        os.path.join(chimerax_dir, 'python3'),
        os.path.join(chimerax_dir, 'python'),
        os.path.join(os.path.dirname(chimerax_dir), 'lib', 'python3.11', 'bin', 'python3.11'),
        '/apps/chimerax/1.10-noble/usr/lib/ucsf-chimerax/bin/python3.11',
    ]

    if 'chimerax' in real_path.lower():
        parts = real_path.split(os.sep)
        for i, part in enumerate(parts):
            if 'chimerax' in part.lower():
                base = os.sep.join(parts[:i+2])
                candidates.extend([
                    os.path.join(base, 'bin', 'python3.11'),
                    os.path.join(base, 'bin', 'python3'),
                    os.path.join(base, 'bin', 'python'),
                ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

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
