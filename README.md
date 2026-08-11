# NeuroMorph: A Deep Learning Framework for Individualized Cortical Feature Extraction

![Schema](<media/NeuroMorph_Schema.png>)


📄 [Paper (medRxiv)](placeholder) &nbsp;|&nbsp; 
🖥️ [Website](https://rocknroll87q.github.io/neuromorph/) &nbsp;|&nbsp; 
📓 [Inference Guide](./Inference_Guide.ipynb) &nbsp;|&nbsp;
⚙️ [Installation Guide](./docs/installation.md) &nbsp;|&nbsp;
🐳 [Docker](https://hub.docker.com/r/rocknroll87q/neuromorph) &nbsp;|&nbsp; 
📦 [Weights](https://huggingface.co/NeuroAI-UofG/NeuroMorph) &nbsp;|&nbsp;
📈 [Morphological Profiling](https://github.com/rockNroll87q/morphological_profiling) &nbsp;|&nbsp; 

> **NeuroMorph is a deep learning framework for individualized cortical feature extraction from T1w structural MRI scans. The framework consists of two deep learning models *`LODBrain+`* for cortical segmentation and *`DeepThickness`* for cortical surface reconstruction and thickness estimation.** 

*Morphological profiles* used in the paper were derived by applying normative modelling to the cortical features extracted by NeuroMorph. See *[Morphological Profiling](https://github.com/rockNroll87q/morphological_profiling)* for additional code relating to normative modelling and classification.



## What it does

NeuroMorph takes a T1w MRI scan and produces:

>| Output | Description | Format |
>|--------|-------------|--------|
>| `subjects_morphology` | Predicted cortical morphology (13 quantitative features, see breakdown below) | `.csv` |
>| `cortical_segmentation` *(optional)* | Predicted cortical segmentation mask | `.nii.gz` |
>| `cortical_surface` *(optional)* | Predicted cortical surface mesh | `.ply` or `.surf` |


**`subjects_morphology` feature breakdown:**

**Volumetric Features (8):** GM, WM, ICV, CSF, Ventricles, Basal Ganglia, Brainstem, Cerebellum

**Surface Features (5):** Cortical Thickness, Pial Curvature, WM Curvature, Pial Surface Area, WM Surface Area.

To understand the differences in outputs and how they might be used, see [outputs](./Inference_Guide.ipynb) "`Saving Options`".


## Installation

### Git
```bash
pip install git+https://github.com/rockNroll87q/NeuroMorph.git
```
Alternatively: 
```bash
git clone https://github.com/rockNroll87q/NeuroMorph.git
cd NeuroMorph
pip install .
```
**Requires Python 3.8.10 and TensorFlow 2.13**. 


### Container

The quickest way to get started is to use the provided Docker container. This will ensure that all dependencies are installed correctly and avoid any potential issues with your local environment.
```bash
docker pull rocknroll87q/neuromorph:latest
docker run --rm -v /path/to/data:/input -v /path/to/output_dir:/output rocknroll87q/neuromorph \
    --data.vol_in=/input/scan.nii.gz --data.output_dir=/output/
```
For more installation options see the [`Installation Guide`](./docs/installation.md).

## Usage

### Single file
```bash
python3 /NeuroMorph/scripts/run_inference.py \
         --data.vol_in='/NeuroMorph/neuromorph_input/sub-0001_T1w.nii.gz' \
         --data.output_dir='/NeuroMorph/neuromorph_output' 
```

For working examples and more usage cases, see [Inference Examples](./Inference_Guide.ipynb#Inference-Examples).

## Useful Notebooks

>| Notebook | Description |
>|----------|-------------|
>| [Installation.ipynb](./docs/installation.md) | A guide on how to install NeuroMorph |
>| [Inference_Guide.ipynb](./Inference_Guide.ipynb) | A guide on how to run inference with NeuroMorph, including working examples and troubleshooting |



## Input requirements

| Input Parameter | Requirements |
|---|---|
|Modality	| T1-weighted MRI
|Format	NIfTI | (.nii, .nii.gz)
|Resolution	| 1mm isotropic
|Orientation |	LIA recommended; the pipeline will attempt to reorient automatically
|Preprocessing	| None (All handled internally)
|Space	| Any (No registration needed) | 

---


## Citation

If you use NeuroMorph in your research, please cite:

TODO: Add citation information here.



## License

NeuroMorph code and model weights are released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

For further detail, see [LICENSE](./LICENSE).


## Contributing

This repository is in an early-release state accompanying the manuscript. Bug reports and questions are welcome via [GitHub Issues](https://github.com/rockNroll87q/NeuroMorph/issues). Please open an issue before submitting a pull request.

