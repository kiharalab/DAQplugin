# DAQplugin Manual for ChimeraX

**DAQplugin** computes, visualizes, and inspects DAQ scores for protein atomic models fitted into cryo-EM density maps.

DAQ, Deep-learning-based Amino-acid-wise model Quality, is a residue-wise local quality score designed to detect amino-acid misassignment and local modeling errors in cryo-EM-derived protein structures.
https://www.nature.com/articles/s41592-022-01574-4

## Quick Usage

### GUI workflow

1. Open your cryo-EM map and atomic model in ChimeraX.
2. Start the tool: **Tools > Validation > DAQplugin**.
3. In the **Main** tab, select:
   - **Structure**: the atomic model, for example `#2`
   - **Map**: the cryo-EM map, for example `#1`
   - **Output / Overwrite NPY**: where computed DAQ scores will be saved
   - **Metric**: usually `DAQ(AA)`
4. In the **Parameters** tab, check:
   - **Contour**: density threshold for grid sampling
   - **Stride**: grid sampling interval, default `2`
   - **Batch size**: inference batch size, default `Auto` (picks the tuned default per backend)
   - **Backend**: inference path, default `Auto` (per-platform fallback chain — TensorRT/CUDA on Linux/NVIDIA, DirectML on Windows, MLX on Apple Silicon)
5. Return to **Main** and click **Calculate DAQ Scores**.
6. Click **Color Structure** to color the selected model.
7. Optional: click **Start Live Update** to monitor score changes while the model moves.

### Command-line quick start

Compute grid-based DAQ scores from a loaded map and color a loaded model:

```chimerax
daqscore compute_grid #1 0.007 structure #2 metric aa_score output ./daq_scores.npy
```

Color a model from an existing DAQ `.npy` file:

```chimerax
daqcolor apply ./daq_scores.npy #2 metric aa_score half_window 9
```

Start and stop live coloring:

```chimerax
daqcolor monitor #2 npy_path ./daq_scores.npy metric aa_score interval 0.5
daqcolor monitor #2 on false
```

Show sequence-shift suggestion arrows:

```chimerax
daq arrowwin #2 ./daq_scores.npy nwin 5 kshift 5 min_improvement 0.5
```

Save the colored/scored model:

```chimerax
save scored_model.cif #2
```

The scores used for coloring are written to the model B-factor field.

## What DAQplugin Provides

- `daqscore compute_grid`: compute DAQ probability grids from a cryo-EM map.
- `daqscore compute_pdb`: compute DAQ scores at model atom positions, similar to the original DAQ workflow.
- `daqcolor apply`: color a model once from a DAQ `.npy` file.
- `daqcolor monitor`: repeatedly recolor a model as coordinates change.
- `daqcolor points`: display DAQ grid points as markers.
- `daqcolor clear`: close marker models created by `daqcolor points`.
- `daq arrowwin`: draw sequence-shift suggestion arrows.
- `daq clearrestraints`: clear ISOLDE restraints created by `daq arrowwin`.
- GUI table: inspect per-residue DAQ scores and click rows to focus residues.

Supported coloring metrics:

- `aa_score`: DAQ(AA), amino-acid assignment quality.
- `atom_score`: DAQ(CA), C-alpha atom likelihood.
- `ss_score`: DAQ(SS), secondary-structure agreement.
- `aa_conf:<AA>`: confidence for a specific amino-acid type, for example `aa_conf:ALA`.

DAQplugin runs inference through one of several GPU/CPU backends, auto-selected for your platform. See [GPU Acceleration and Backends](#gpu-acceleration-and-backends) below. For very large maps that exceed local GPU memory, use the Google Colab notebook to generate the `.npy` file, then visualize it in ChimeraX:

- [DAQ_Score_Grid.ipynb](https://colab.research.google.com/github/gterashi/DAQplugin/blob/main/DAQ_Score_Grid.ipynb)

<img src="https://github.com/gterashi/DAQplugin/blob/main/img/demo.gif?raw=true" width="600">

## GUI Reference

Start the GUI from **Tools > Validation > DAQplugin**.




### Main tab

<img src="https://github.com/gterashi/DAQplugin/blob/main/img/gui1.png?raw=true" width="200">

**Inputs**

- **Structure**: loaded atomic model.
- **Map**: loaded cryo-EM density map.
- **Output / Overwrite NPY**: output path for newly computed DAQ scores. If `.npy` is omitted, it is added automatically.
- **Load Existing NPY**: existing DAQ `.npy` file for coloring, monitoring, residue table, and arrows.

**Actions**

- **Calculate DAQ Scores**: run grid-based DAQ computation from the selected map.
- **Color Structure**: apply coloring once using the current `.npy` and metric.
- **Start Live Update**: repeatedly recolor the model using current coordinates.
- **Stop Update**: stop live recoloring.
- **Show Shift Arrows**: draw sequence-shift suggestion arrows.
- **Clear Shift Arrows**: delete the DAQ arrow group.
- **Add Arrow Constraints**: draw arrows and create ISOLDE position restraints.
- **Clear Arrow Constraints**: clear DAQ-created ISOLDE restraints.
- **Calculate Atom-Based DAQ**: run structure/atom-position-based DAQ computation.

### Parameters tab

<img src="https://github.com/gterashi/DAQplugin/blob/main/img/gui2.png?raw=true" width="200">

**Compute Settings**

- **Batch size**: samples per inference batch, default `Auto`. Auto picks the tuned default per backend (TensorRT 2048, CUDA 1024, DirectML/CPU 256). Set a numeric value to override (e.g. to avoid OOM on a small GPU).
- **Backend**: inference path, default `Auto`. Auto follows the per-platform fallback chain. Force a specific backend (`TensorRT`, `CUDA`, `DirectML`, `MLX (Metal)`, `MLX (CPU)`, `CPU`) to skip the chain.
- **GPU device** (Linux only): NVIDIA device picker for multi-GPU hosts. Active for `tensorrt`/`cuda` backends.

**Grid Settings**

- **Contour**: density threshold for sampling grid points. This usually matches the map display contour.
- **Stride**: sampling interval in voxels. `1` is denser and slower; `2` is the default.
- **Max Points**: maximum number of sampled grid points, default `500000`.

**Scoring Settings**

- **k**: number of nearest DAQ grid points queried by cKDTree, default `1`.
- **Half window**: residue-number half-window for final window averaging, default `9`. A value of `9` averages scores over residues with numbers `n-9` through `n+9` in the same chain. Missing residue numbers are simply skipped.

**Coloring Settings**

- **Clamp min / max**: color scale bounds, default `-1.0` to `1.0`.
- **Interval (sec)**: live update interval, default `0.5`.
- **kNN workers**: SciPy cKDTree query workers. `1` keeps default behavior; `-1` uses all available cores when supported by SciPy.
- **Log timing**: write detailed timing to the ChimeraX log.

**Sequence Shift Suggestion Parameters**

- **Half window**: scoring window used by arrow suggestions, default `5`.
- **Max shift**: candidate sequence shifts in `[-kshift..-1, +1..+kshift]`.
- **Minimum distance**: minimum arrow length in angstroms.
- **Minimum improvement**: minimum window-mean DAQ improvement required to draw an arrow.
- **Base radius**, **Max color**, **Max radius score**, **Radius scale min/max**: arrow appearance controls.
- **Constraint spring**: ISOLDE position-restraint spring constant, default `1500`.

### Per-Residue DAQ Scores tab
<img src="https://github.com/gterashi/DAQplugin/blob/main/img/gui3.png?raw=true" width="200">

This tab shows chain ID, residue ID, amino-acid name, and window-averaged DAQ score for the current structure and `.npy` file. Click **Refresh** to recompute the table. Click a residue row to select, center, and zoom to that residue in ChimeraX.

## Command Reference

### `daqscore compute_grid`

Compute DAQ scores from a cryo-EM map by sampling grid points above a contour threshold.

```chimerax
daqscore compute_grid mapInput contour [structure #model] [output npyPath] [stride N] [batch_size N] [max_points N] [ckpt ckptPath] [metric metricName] [k N] [colormap cmap] [half_window N] [monitor true|false] [backend name] [gpu_id N]
```

Parameters:

- `mapInput`: MRC/MAP file path or loaded ChimeraX volume model, for example `#1`.
- `contour`: density threshold for grid sampling.
- `structure`: optional model to color after computation.
- `output`: output `.npy` path. If omitted, an output name is generated.
- `stride`: grid sampling interval, default `2`.
- `batch_size`: inference batch size, default `0` (Auto — picks the tuned default per backend).
- `max_points`: maximum sampled grid points, default `500000`.
- `ckpt`: optional ONNX model path. The bundled model is used by default.
- `metric`: coloring metric used if `structure` is supplied, default `aa_score`.
- `k`: nearest-neighbor count for coloring, default `1`.
- `colormap`: optional ChimeraX colormap.
- `half_window`: residue-number half-window for smoothing, default `9`.
- `monitor`: if `true` and `structure` is supplied, start `daqcolor monitor`.
- `backend`: inference backend, default `auto`. Choices: `auto`, `tensorrt`, `cuda`, `directml`, `mlx`, `mlx-cpu`, `cpu`. See [GPU Acceleration and Backends](#gpu-acceleration-and-backends).
- `gpu_id`: NVIDIA device ID for `tensorrt`/`cuda` backends on multi-GPU Linux hosts, default `0`. Ignored by other backends.

Examples:

```chimerax
daqscore compute_grid #1 0.007 output ./daq_scores.npy
daqscore compute_grid #1 0.007 structure #2 metric aa_score
daqscore compute_grid #1 0.007 structure #2 monitor true metric aa_score half_window 9
daqscore compute_grid /path/to/map.mrc 0.5 output /path/to/output.npy
```

### `daqscore compute_pdb`

Compute DAQ scores using heavy atom positions from a structure as query points instead of grid points from the map. This is suitable for atom-position-based evaluation and comparison with the original DAQ workflow. Monitoring is not started by this command.

```chimerax
daqscore compute_pdb mapInput structure #model [output npyPath] [batch_size N] [ckpt ckptPath] [metric metricName] [k N] [colormap cmap] [half_window N] [apply_color true|false] [save_model modelPath] [backend name] [gpu_id N]
```

Parameters:

- `mapInput`: MRC/MAP file path or loaded ChimeraX volume model, for example `#1`.
- `structure`: required atomic model.
- `output`: output `.npy` path. If omitted, an output name is generated.
- `batch_size`: inference batch size, default `0` (Auto — picks the tuned default per backend).
- `ckpt`: optional ONNX model path. The bundled model is used by default.
- `metric`: coloring metric, default `aa_score`.
- `k`: nearest-neighbor count for coloring, default `1`.
- `colormap`: optional ChimeraX colormap.
- `half_window`: residue-number half-window for smoothing, default `9`.
- `apply_color`: apply coloring after computation, default `true`.
- `save_model`: optional PDB/mmCIF path. Scores are saved in B-factors.
- `backend`: inference backend, default `auto`. See [GPU Acceleration and Backends](#gpu-acceleration-and-backends).
- `gpu_id`: NVIDIA device ID for `tensorrt`/`cuda` backends on multi-GPU Linux hosts, default `0`.

Examples:

```chimerax
daqscore compute_pdb #1 structure #2 metric aa_score
daqscore compute_pdb #1 structure #2 apply_color false
daqscore compute_pdb #1 structure #2 metric aa_score save_model scored_model.pdb
daqscore compute_pdb #1 structure #2 metric atom_score k 1 half_window 9 save_model output.cif
```

### `daqcolor apply`

Color a model once from an existing DAQ `.npy` file.

```chimerax
daqcolor apply npyPath model [k N] [metric metricName] [atom_name atomName] [half_window N] [colormap cmap] [clamp_min value] [clamp_max value] [log_timing true|false] [knn_workers N]
```

Examples:

```chimerax
daqcolor apply ./daq_scores.npy #2 metric aa_score
daqcolor apply ./daq_scores.npy #2 metric atom_score atom_name CA clamp_min -1 clamp_max 1
daqcolor apply ./daq_scores.npy #2 metric ss_score half_window 9 log_timing true
```

Notes:

- `atom_name` defaults to `CA`.
- `half_window` defaults to `9`.
- Window averaging is computed within the same chain by residue number, not by array position.
- Window plans are cached for the current residue layout and `half_window`, so repeated coloring of the same model avoids rebuilding the residue-number window structure.

### `daqcolor monitor`

Start or stop live recoloring. When monitoring is on, the model is recolored at the requested interval using current atom coordinates.

```chimerax
daqcolor monitor model [npy_path npyPath] [k N] [metric metricName] [atom_name atomName] [half_window N] [colormap cmap] [clamp_min value] [clamp_max value] [on true|false] [interval seconds] [log_timing true|false] [knn_workers N]
```

Examples:

```chimerax
daqcolor monitor #2 npy_path ./daq_scores.npy metric aa_score
daqcolor monitor #2 npy_path ./daq_scores.npy metric aa_score interval 1.0
daqcolor monitor #2 on false
```

Notes:

- `npy_path` is required when starting monitoring.
- `npy_path` is not required when stopping with `on false`.
- Starting a new monitor for the same model replaces the previous monitor.

### `daqcolor points`

Show DAQ `.npy` point coordinates as marker models.

```chimerax
daqcolor points npyPath [radius value] [metric metricName] [colormap cmap] [clamp_min value] [clamp_max value]
```

Examples:

```chimerax
daqcolor points ./daq_scores.npy radius 0.4
daqcolor points ./daq_scores.npy metric aa_conf radius 0.3
daqcolor clear
```

For `daqcolor points`, supported `metric` values are `aa_conf` and `aa_top:<AA>`, for example `aa_top:ALA`.

### `daq arrowwin`

Draw sequence-shift suggestion arrows using DAQ AA scores from a `.npy` file. If residues are selected, only selected residues are processed; otherwise the whole model is processed.

```chimerax
daq arrowwin structure npy_path [chain chainId] [nwin N] [kshift N] [minmove value] [radius value] [min_improvement value] [vmax_color value] [vmax_radius value] [max_radius_scale value] [min_radius_scale value] [group_name name] [apply_isolde_restraints true|false] [spring_constant value]
```

Examples:

```chimerax
daq arrowwin #2 ./daq_scores.npy nwin 5 kshift 5 min_improvement 0.5
daq arrowwin #2 ./daq_scores.npy chain A apply_isolde_restraints true spring_constant 1500
daq clearrestraints #2
```

<img src="https://github.com/gterashi/DAQplugin/blob/main/img/with_arrow.png?raw=true" width="300">
<img src="https://github.com/gterashi/DAQplugin/blob/main/img/arrow.png?raw=true" width="300">

## GPU Acceleration and Backends

DAQplugin runs inference through one of several backends. The plugin auto-selects the best one for your platform on first use, with a fallback chain when the preferred backend fails to initialize. The active backend is printed to the ChimeraX log as `DAQ: backend='...'`.

| Platform | Fallback chain | Notes |
|----------|----------------|-------|
| Linux (NVIDIA)        | TensorRT → CUDA → CPU             | TRT engines cached at `~/.chimerax/daq_model/trt_cache/` (first build ~10 s, subsequent ~0.5 s). |
| Windows               | TensorRT → DirectML → CPU         | DirectML covers any GPU vendor (NVIDIA/AMD/Intel). |
| macOS (Apple Silicon) | MLX-Metal → MLX-CPU → ORT-CPU     | MLX uses the unified GPU and the Accelerate/AMX coprocessor. |
| macOS (Intel)         | ORT-CPU                           | No MLX wheel exists for x86_64 Mac. |

**GUI**: pick from the **Backend** dropdown in the Parameters tab. The **GPU device** dropdown is shown on Linux only and selects the NVIDIA device for TensorRT/CUDA.

**Command line**: pass `backend NAME` and (Linux NVIDIA) `gpu_id N`.

```chimerax
daqscore compute_grid #1 0.5 backend tensorrt gpu_id 1
daqscore compute_grid #1 0.5 backend cpu
daqcolor monitor #2 npy_path ./daq_scores.npy backend cuda
```

Available backend names: `auto`, `tensorrt`, `cuda`, `directml`, `mlx`, `mlx-cpu`, `cpu`.

**Batch size**: auto-selected per backend (TensorRT 2048, CUDA 1024, DirectML/CPU 256). Override via the GUI **Batch size** field or `batch_size N` on the command line. Lower batch sizes reduce memory but may underutilize the GPU. Set the env var `DAQ_BATCH_OVERRIDE=<n>` for one-shot benchmarking.

**TensorRT engine cache**: shared across launches. Delete `~/.chimerax/daq_model/trt_cache/` to force a rebuild (e.g., after a driver upgrade).

## Window Averaging

`half_window` controls the final residue-level smoothing used by `daqcolor apply`, `daqcolor monitor`, `daqscore compute_grid` coloring, and `daqscore compute_pdb` coloring.

For residue number `n` and `half_window=9`, DAQplugin averages valid finite scores from the same chain whose residue numbers are in:

```text
n-9 through n+9
```

This is residue-number based, so missing residue numbers are skipped naturally. For example, if a chain contains residue numbers `1, 2, 3, 7, 8, 9`, the window around residue `7` with `half_window=2` includes existing residues `7, 8, 9`.

## Typical Workflows

### Full DAQ computation and visualization

1. Open map and model.
2. Set the map contour.
3. Run `daqscore compute_grid`, or click **Calculate DAQ Scores** in the GUI.
4. Color with `aa_score`.
5. Inspect the per-residue table and suspicious regions.

### Coloring only from a precomputed `.npy`

1. Open the model.
2. Set **Load Existing NPY** in the GUI, or run `daqcolor apply`.
3. Use `aa_score`, `atom_score`, `ss_score`, or `aa_conf:<AA>` as needed.

### Live DAQ monitoring with ISOLDE or manual model movement

1. Compute or load an existing DAQ `.npy`.
2. Start live update with `daqcolor monitor` or **Start Live Update**.
3. Move/refine the model.
4. Stop monitoring with `daqcolor monitor #model on false` or **Stop Update**.

### Sequence-shift suggestions and restraints

1. Compute or load a DAQ `.npy`.
2. Draw arrows with `daq arrowwin` or **Show Shift Arrows**.
3. Optionally add ISOLDE restraints using `apply_isolde_restraints true` or **Add Arrow Constraints**.
4. Clear restraints with `daq clearrestraints #model` or **Clear Arrow Constraints**.

## Interpreting DAQ Scores

- Positive values: local density supports the modeled amino-acid type better than the average distribution.
- Negative values: possible amino-acid misassignment or local modeling inconsistency.
- Near zero: ambiguous density or locally low-resolution signal.
- Green coloring is used for residues without valid nearby DAQ support or invalid score values.

## Citation

If you use DAQplugin, please cite:

- Terashi G., Wang X., Maddhuri Venkata Subramaniya S. R., Tesmer J. J. G., Kihara D. Residue-wise local quality estimation for protein models from cryo-EM maps. Nature Methods 19, 1116-1125 (2022). [Link](https://www.nature.com/articles/s41592-022-01574-4)
- Nakamura, T., Wang, X., Terashi, G., & Kihara, D. DAQ-Score Database: assessment of map-model compatibility for protein structure models from cryo-EM maps. Nature Methods 20, 775-776 (2023).

## Licensing and Commercial Use

DAQplugin is provided free of charge for academic and non-profit research purposes.

For commercial licensing inquiries, please contact:

- contact@intellicule.com
- dkihara@purdue.edu

For technical questions, please contact:

- gterashi@purdue.edu
