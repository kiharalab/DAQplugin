#!/usr/bin/env python3
"""
Build separate wheels for each platform configuration using ChimeraX's bundled Python.

This script creates platform-specific wheels with the appropriate dependencies
baked in, rather than using optional dependencies. It uses ChimeraX's bundled
Python which includes ChimeraX-BundleBuilder.

Usage:
    python build_wheels.py [--platforms PLATFORMS] [--chimerax PATH]

Options:
    --platforms     Comma-separated list: linux,linux-cuda,linux-cpu,win,mac,all
    --chimerax      Path to ChimeraX executable (auto-detected if not specified)

Output wheels (PEP 425 platform-tagged, all in wheels/):
    chimerax_daqplugin-VERSION-py3-none-linux_x86_64.whl            (linux-bundled-cuda)
    chimerax_daqplugin-VERSION-py3-none-win_amd64.whl               (win/DirectML)
    chimerax_daqplugin-VERSION-py3-none-macosx_11_0_universal2.whl  (mac: arm64 + Intel; mlx gated by PEP 508 marker)
    chimerax_daqplugin-VERSION-py3-none-any.whl                     (generic CPU fallback)

Two Linux variants (bundled-cuda vs system-cuda) collide on linux_x86_64.
Default build target excludes linux-system-cuda. Pass --platforms explicitly
to build it; it lands in dist/wheels/system_cuda/ to avoid clobbering.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# Platform-specific dependencies
# NOTE: Only ONE onnxruntime variant can be installed at a time!
# onnxruntime-gpu (Linux/NVIDIA, includes CPU fallback), onnxruntime-directml (Windows), onnxruntime (macOS)
PLATFORM_DEPS = {
    # Linux with bundled CUDA stack: TensorRT + cuDNN + CUDA via pip. TRT is
    # ~356x faster than CUDA EP on the 3D-Conv DAQ model (benchmark on
    # RTX 6000 Ada: TRT 825K p/s, CUDA EP 2.3K p/s). Bundling adds ~1.5 GB
    # to the install, but no system CUDA toolkit required.
    'linux-bundled-cuda': [
        "onnxruntime-gpu >=1.19.0",
        "tensorrt-cu12 >=10.0.0",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12 >=9.0.0",
        "nvidia-cublas-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
    ],
    # Linux with system CUDA: user provides CUDA toolkit + cuDNN + (optional)
    # TensorRT. ORT-GPU picks TRT EP if libs are in LD_LIBRARY_PATH, falls
    # back to CUDA EP otherwise.
    'linux-system-cuda': [
        "onnxruntime-gpu >=1.19.0",
    ],
    # Windows: DirectML covers all GPU vendors (NVIDIA/AMD/Intel). TensorRT
    # pip wheels for Windows exist only as cu13 variants, incompatible with
    # cu12-based ORT 1.19+. NVIDIA NVIDIA users who want TRT must install
    # it via the official NVIDIA installer (zip/MSI) and add it to PATH;
    # ORT will then pick it up automatically. Until ORT ships cu13 builds,
    # we cannot pip-bundle TRT on Windows.
    'win': [
        "onnxruntime-directml >=1.16.0",
    ],
    # Single universal2 mac wheel covers Apple Silicon AND Intel Mac. MLX
    # is gated by a PEP 508 marker -- arm64 hosts pull it and use the Metal
    # backend; x86_64 hosts skip it (no x86_64 mlx wheel exists) and fall
    # back to ORT CPU. Matches the ISOLDE distribution pattern.
    'mac': [
        "onnxruntime >=1.16.0",
        "mlx >=0.20.0; platform_machine == 'arm64'",
    ],
    'cpu': [
        "onnxruntime >=1.16.0",
    ],
}

# Base dependencies (always included, no onnxruntime - it's platform-specific)
BASE_DEPS = [
    "ChimeraX-Core ~=1.1",
    "ChimeraX-UI ~=1.0",
    "ChimeraX-Map ~=1.0",
    "ChimeraX-Markers ~=1.0",
    "numpy >=1.20.0",
    "scipy >=1.7.0",
    "mrcfile >=1.4.0",
    "numba >=0.52.0",
]


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

    # Resolve symlinks to get the actual installation directory
    real_path = os.path.realpath(chimerax_path)
    chimerax_dir = os.path.dirname(real_path)

    # Common locations for ChimeraX's Python
    candidates = [
        # Linux: /usr/lib/ucsf-chimerax/bin/python3.11
        os.path.join(chimerax_dir, 'python3.11'),
        os.path.join(chimerax_dir, 'python3'),
        os.path.join(chimerax_dir, 'python'),
        # Look in parent dirs
        os.path.join(os.path.dirname(chimerax_dir), 'lib', 'python3.11', 'bin', 'python3.11'),
        # For installed ChimeraX
        '/apps/chimerax/1.10-noble/usr/lib/ucsf-chimerax/bin/python3.11',
    ]

    # Also check standard installation patterns
    if 'chimerax' in real_path.lower():
        # Extract base installation directory
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


def read_pyproject():
    """Read and return pyproject.toml content."""
    with open('pyproject.toml', 'r') as f:
        return f.read()


def write_pyproject(content):
    """Write pyproject.toml content."""
    with open('pyproject.toml', 'w') as f:
        f.write(content)


def generate_pyproject_for_platform(platform_name, original_content):
    """
    Generate pyproject.toml content with platform-specific dependencies.
    """
    # Get platform-specific deps
    platform_deps = PLATFORM_DEPS.get(platform_name, [])

    # Combine base + platform deps
    all_deps = BASE_DEPS.copy()
    all_deps.extend(platform_deps)

    # Format dependencies as TOML
    deps_str = ',\n    '.join(f'"{d}"' for d in all_deps)

    # Create new pyproject.toml content
    # We'll use a simpler approach: find and replace the dependencies section
    import re

    # Pattern to match the dependencies array
    pattern = r'(dependencies\s*=\s*\[)[^\]]*(\])'
    replacement = f'\\1\n    {deps_str},\n\\2'

    new_content = re.sub(pattern, replacement, original_content, count=1)

    # Also remove optional-dependencies section for cleaner wheel
    # Match from [project.optional-dependencies] to next section or end
    opt_pattern = r'\[project\.optional-dependencies\].*?(?=\n\[|\Z)'
    new_content = re.sub(opt_pattern, '', new_content, flags=re.DOTALL)

    return new_content


def build_wheel(python_exe, bundle_dir):
    """Build wheel using Python build tools with ChimeraX's Python."""
    # Use python -m build with ChimeraX's bundled Python
    # --no-isolation: use system packages instead of isolated build environment
    # -v: verbose output for debugging

    # Try using python -m build (recommended)
    cmd = [python_exe, '-m', 'build', '--wheel', '--no-isolation', '-v', '--outdir', 'dist', str(bundle_dir)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=bundle_dir)

    if result.returncode != 0:
        # Fallback: try pip wheel with --no-build-isolation
        print("Trying fallback: pip wheel --no-build-isolation")
        cmd = [python_exe, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '-v', '--wheel-dir', 'dist', str(bundle_dir)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=bundle_dir)

    return result.returncode == 0


## PEP 425 tag tuples per platform (python-tag, abi-tag, platform-tag).
## Bundle is pure Python (no compiled extensions) -> use py3/none for the
## Python+ABI slots. Platform tag is what matters for OS routing. This
## survives ChimeraX Python version bumps (cp311 -> cp312) since our deps
## have multi-cp binary wheels that pip picks at install time.
WHEEL_TAGS = {
    'linux-bundled-cuda': ('py3', 'none', 'linux_x86_64'),
    'linux-system-cuda':  ('py3', 'none', 'linux_x86_64'),
    'win':                ('py3', 'none', 'win_amd64'),
    'mac':                ('py3', 'none', 'macosx_11_0_universal2'),
    'cpu':                ('py3', 'none', 'any'),
}


def retag_wheel(python_exe, wheel_path, tags):
    """
    Retag a py3-none-any wheel to the platform-specific tag triple via
    `python -m wheel tags`. Returns the new wheel path or None on failure.
    """
    py_tag, abi_tag, plat_tag = tags
    cmd = [
        python_exe, '-m', 'wheel', 'tags',
        '--python-tag', py_tag,
        '--abi-tag', abi_tag,
        '--platform-tag', plat_tag,
        '--remove',
        str(wheel_path),
    ]
    print(f"Retagging: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  retag failed: {result.stderr.strip()}")
        return None
    new_name = result.stdout.strip().splitlines()[-1].strip()
    # wheel tags prints just the basename; resolve against source dir.
    return (Path(wheel_path).parent / new_name) if new_name else None


def organize_wheel(python_exe, dist_dir, platform_name, out_dir):
    """
    Retag the built wheel with platform-specific PEP 425 tags and move it
    to a single flat output dir. linux-system-cuda goes to a subdir to
    avoid collision with linux-bundled-cuda on the same platform tag.
    """
    wheels = list(Path(dist_dir).glob('*.whl'))
    if not wheels:
        return None

    wheel = wheels[0]
    if platform_name not in WHEEL_TAGS:
        raise ValueError(
            f"No PEP 425 tag mapping for platform '{platform_name}'. "
            f"Known: {sorted(WHEEL_TAGS)}. Add an entry to WHEEL_TAGS."
        )
    tags = WHEEL_TAGS[platform_name]
    retagged = retag_wheel(python_exe, wheel, tags) or wheel

    # linux-system-cuda shares the linux_x86_64 tag with bundled-cuda; park
    # it in a subdir so the bundled wheel stays the canonical Linux artifact.
    target_dir = Path(out_dir)
    if platform_name == 'linux-system-cuda':
        target_dir = target_dir / 'system_cuda'
    target_dir.mkdir(parents=True, exist_ok=True)

    final_path = target_dir / retagged.name
    shutil.move(str(retagged), str(final_path))
    return final_path


def main():
    parser = argparse.ArgumentParser(description='Build platform-specific wheels using ChimeraX Python')
    parser.add_argument('--platforms', '-p', default='default',
                        help="Platforms to build. 'default' = linux-bundled-cuda,"
                             "win,mac,cpu (one wheel per Toolshed-target platform). "
                             "'all' adds linux-system-cuda (clashes with bundled "
                             "on linux_x86_64; lands in subdir). Or comma-list: "
                             "linux-bundled-cuda,linux-system-cuda,win,mac,cpu")
    parser.add_argument('--chimerax', '-x', default=None,
                        help='Path to ChimeraX executable (auto-detected if not specified)')
    parser.add_argument('--out', '-o', default='wheels',
                        help='Output directory for built wheels (default: wheels). '
                             "Must be outside 'dist/' which is wiped between builds.")
    args = parser.parse_args()

    # Default = Toolshed-ready set (excludes linux-system-cuda; same plat tag
    # as bundled-cuda would clobber it). 'all' includes everything.
    if args.platforms == 'default':
        platforms = ['linux-bundled-cuda', 'win', 'mac', 'cpu']
    elif args.platforms == 'all':
        platforms = list(PLATFORM_DEPS.keys())
    else:
        platforms = [p.strip() for p in args.platforms.split(',')]

    # Find ChimeraX
    chimerax = args.chimerax or find_chimerax()
    if not chimerax or not os.path.exists(chimerax):
        print("Error: ChimeraX not found. Use --chimerax to specify path.")
        return 1

    # Find ChimeraX's bundled Python
    python_exe = find_chimerax_python(chimerax)
    if not python_exe:
        print(f"Error: Could not find ChimeraX's Python executable.")
        print(f"ChimeraX path: {chimerax}")
        print("Please ensure ChimeraX is properly installed.")
        return 1

    bundle_dir = Path(__file__).parent.absolute()
    dist_dir = bundle_dir / 'dist'

    # Backup original pyproject.toml
    original_content = read_pyproject()

    print("=" * 60)
    print("Building platform-specific wheels")
    print("=" * 60)
    print(f"ChimeraX: {chimerax}")
    print(f"Python: {python_exe}")
    print(f"Platforms: {', '.join(platforms)}")
    print("=" * 60)

    built_wheels = []

    # MLX weights (.npz) are bundled only with the mac wheel. Stash them aside
    # for non-mac builds so the [chimerax.package-data] glob *.npz doesn't pull
    # them into wheels that won't use them (the file is ~37 MB).
    data_dir = bundle_dir / 'src' / 'data'
    npz_stash = bundle_dir / '.npz_stash'
    npz_files = list(data_dir.glob('*.npz')) if data_dir.exists() else []

    def _stash_npz():
        if not npz_files:
            return
        npz_stash.mkdir(exist_ok=True)
        for f in npz_files:
            target = npz_stash / f.name
            if f.exists():
                shutil.move(str(f), str(target))

    def _restore_npz():
        if not npz_stash.exists():
            return
        for f in npz_stash.glob('*.npz'):
            shutil.move(str(f), str(data_dir / f.name))
        try:
            npz_stash.rmdir()
        except OSError:
            pass

    try:
        for platform_name in platforms:
            print(f"\n{'='*60}")
            print(f"Building wheel for: {platform_name}")
            print('='*60)

            # Clean previous build
            if dist_dir.exists():
                shutil.rmtree(dist_dir)
            build_dir = bundle_dir / 'build'
            if build_dir.exists():
                shutil.rmtree(build_dir)

            # Per-iteration try/finally: restore pyproject.toml immediately
            # after this platform's build so a Ctrl-C or crash between
            # iterations never leaves the source tree mutated.
            platform_content = generate_pyproject_for_platform(platform_name, original_content)
            write_pyproject(platform_content)
            try:
                # Bundle MLX weights only with the mac wheel
                if platform_name == 'mac':
                    _restore_npz()
                else:
                    _stash_npz()

                # Build wheel
                if not build_wheel(python_exe, str(bundle_dir)):
                    print(f"Error: Failed to build wheel for {platform_name}")
                    continue

                # Retag + move into the single flat output directory.
                out_dir = bundle_dir / args.out if not os.path.isabs(args.out) else Path(args.out)
                wheel_path = organize_wheel(python_exe, dist_dir, platform_name, out_dir)
                if wheel_path:
                    built_wheels.append(wheel_path)
                    print(f"Built: {wheel_path.name}")
            finally:
                # Restore pyproject before the next iteration so the
                # source tree is always clean between platforms.
                write_pyproject(original_content)

    finally:
        # Outer finally is now only for npz restoration; pyproject is
        # already clean per inner finally, but write once more as belt-
        # and-suspenders in case the outer try never entered the loop.
        write_pyproject(original_content)
        _restore_npz()
        print("\nRestored original pyproject.toml")

    # Summary
    print("\n" + "=" * 60)
    print("Build Summary")
    print("=" * 60)
    if built_wheels:
        print(f"Successfully built {len(built_wheels)} wheels:")
        for w in built_wheels:
            # Show relative path when possible, else fall back to absolute.
            try:
                shown = w.relative_to(bundle_dir)
            except ValueError:
                shown = w
            print(f"  - {shown}")
        out_dir = bundle_dir / args.out if not os.path.isabs(args.out) else Path(args.out)
        print(f"\nWheels in: {out_dir}")
        print("Install with: chimerax --cmd 'toolshed install <path-to-wheel>'")
    else:
        print("No wheels were built.")

    return 0 if built_wheels else 1


if __name__ == '__main__':
    sys.exit(main())
