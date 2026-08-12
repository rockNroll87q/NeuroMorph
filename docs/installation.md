# Installation instructions

NeuroMorph can be installed or used via the following options. We recommend installing via pip, but if you have docker that may be the easiest and fastest method.

### Option 1: pip (recommended)
```bash
pip install git+https://github.com/rocknroll87q/neuromorph.git
```

Alternatively: 
```bash
git clone https://github.com/rocknroll87q/neuromorph.git
cd NeuroMorph
pip install .
```

> NeuroMorph requires Python 3.8.10 and TensorFlow 2.13. **Python 3.12+ is not supported**. If you are on 3.12 or newer, please create a dedicated environment with Python 3.11 using Miniforge (see [Option 2](#option-2-miniforge)) or use [Docker](#option-3-docker-zero-friction-recommended-if-you-hit-dependency-issues). See [requirements.txt](../requirements.txt) for full dependencies.

### Option 2: Miniforge
```bash
mamba create -n NeuroMorph python=3.8.10
mamba activate NeuroMorph
pip install git+https://github.com/rocknroll87q/neuromorph.git
```

### Option 3: Docker *(zero-friction, recommended if you hit dependency issues)*

#### Inference docker
```bash
docker pull rocknroll87q/neuromorph:latest
docker run --rm -v /path/to/data:/input -v /path/to/output_dir:/output rocknroll87q/neuromorph \
    --data.vol_in=/input/scan.nii.gz --data.output_dir=/output/
```

#### Notebook docker

```bash
docker pull rocknroll87q/neuromorph:v1.0_notebook
docker run --rm -v /path/to/data:/input -v /path/to/output_dir:/output -p 8888:8888 rocknroll87q/neuromorph:v1.0_notebook
```

Open browser to `http://localhost:8888`.

### Option 4: Singularity *(for HPC/cluster environments)*
```bash
cd /local/path/to/NeuroMorph/
singularity build ./neuromorph_v1.0.simg docker://rocknroll87q/neuromorph:latest

singularity run --env LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
 --bind /path/to/input:/input --bind /path/to/output_dir:/output neuromorph_v1.0.simg \
    --data.vol_in=/input/scan.nii.gz --data.output_dir=/output/
```

---

*For more information, see the [main README](../README.md) or open an issue on [GitHub](https://github.com/rocknroll87q/neuromorph/issues).*