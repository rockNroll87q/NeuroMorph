---
layout: page
title: <a href="https://rocknroll87q.github.io/neuromorph/">NeuroMorph</a>
---

# Abstract


Structural MRI is routinely acquired in clinical practice, yet quantitative morphometry has had limited impact on clinical decision-making. 
Overlapping symptoms, trajectories and comorbidities remain difficult to interpret within disease-specific frameworks, leaving it unclear how individual patients relate to the broader organization of brain disease.Here, we construct a cross-disease morphological reference space from 110,591 T1w MRI scans of 78,794 participants, spanning four disease families, 19 diagnoses, and seven subtypes. To construct this space, we developed NeuroMorph, an AI framework deriving thirteen interpretable morphological descriptors and individual normative deviation profiles. The reference space reveals shared and distinct morphological signatures that distinguish conditions within a hierarchy of disease families, diagnoses, subtypes and individual profiles. It identifies overlapping and comorbid morphological profiles and captures longitudinal deviations that precede clinical diagnosis and track progression. Together, these findings establish a unified framework for mapping brain disease organization and positioning individual patients within its morphological landscape.
Code, model, and demo are available on the [GitHub repository](https://github.com/rockNroll87q/NeuroMorph).

<p align="center">
<img src="./misc/Overview_Figure.png" width="75%" />
<figcaption>Study overview | a, The NeuroMorph framework extracts 13 morphological features from native-space T1-weighted sMRI. NeuroMorph integrates two deep learning models, LOD Brain+ and DeepThickness, which generate segmentation masks and cortical surface reconstructions to derive seven volumetric and five surface-based features respectively. b, The disease hierarchy comprises 110,591 sMRI scans from 78,794 participants, spanning four levels of resolution from population, through disease family and diagnosis, to subtype. c, The analysis workflow applied at each level of the hierarchy: defining the shared morphological space, classifying conditions and quantifying feature importance. d, Downstream applications of the shared morphological space at the group and individual levels, comprising, from left to right, differential, comorbid and longitudinal diagnosis.</figcaption>
</p>


<hr>
# Citation


If you find this work useful, please consider citing our paper:
```bibtex
@article {DalbyNeuroMorph2026,
	author = {Dalby, Connor and Dibble, Austin and Benini, Sergio and Ferrari, Damiano and Lyall, Donald M and Harvey, Monika and Quinn, Terry and Muckli, Lars and Fracasso, Alessio and Svanera, Michele and {Alzheimer's Disease Neuroimaging Initiative} and {Frontotemporal Lobar Degeneration Neuroimaging Initiative}},
	title = {NeuroMorph: A Unified Morphological Reference Space for Cross-Disease Brain Profiling},
	elocation-id = {2026.08.13.26359403},
	year = {2026},
	doi = {10.64898/2026.08.13.26359403},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Structural MRI is routinely acquired in clinical practice, yet quantitative morphometry has had limited impact on clinical decision-making. Overlapping symptoms, trajectories and comorbidities remain difficult to interpret within disease-specific frameworks, leaving it unclear how individual patients relate to the broader organization of brain disease. Here, we construct a cross-disease morphological reference space from 110,591 T1w MRI scans of 78,794 participants, spanning four disease families, 19 diagnoses, and seven subtypes. To construct this space, we developed NeuroMorph, an AI framework deriving thirteen interpretable morphological descriptors and individual normative deviation profiles. The reference space reveals shared and distinct morphological signatures that distinguish conditions within a hierarchy of disease families, diagnoses, subtypes and individual profiles. It identifies overlapping and comorbid morphological profiles and captures longitudinal deviations that precede clinical diagnosis and track progression. Together, these findings establish a unified framework for mapping brain disease organization and positioning individual patients within its morphological landscape.},
	URL = {https://www.medrxiv.org/content/10.64898/2026.08.13.26359403v1},
	eprint = {https://www.medrxiv.org/content/10.64898/2026.08.13.26359403v1.full.pdf},
	journal = {medRxiv}
}
```

<hr>
# Acknowledgments


We acknowledge the MVLS Advanced Research System (MARS) at the University of Glasgow for providing high-performance computing resources and technical support.

Some data used in the preparation of this article were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database. As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: [link](http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf). For up-to-date information, see adni.loni.usc.edu.

Some of the data used in the preparation of this article were obtained from the Neuroimaging in Frontotemporal Dementia (NIFD) dataset, part of the Frontotemporal Lobar Degeneration Neuroimaging Initiative (FTLDNI). Data collection and sharing for this project was funded by the Frontotemporal Lobar Degeneration Neuroimaging Initiative (National Institutes of Health Grant R01 AG032306). The study is coordinated through the University of California, San Francisco, Memory and Aging Center. FTLDNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California. For up-to-date information on participation and protocol, see [link](http://memory.ucsf.edu/research/studies/nifd). 

Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

Data were provided [in part] by OASIS-3: Longitudinal Multimodal Neuroimaging: (Principal Investigators: T. Benzinger, D. Marcus, J. Morris); NIH P30 AG066444, P50 AG00561, P30 NS09857781, P01 AG026276, P01 AG003991, R01 AG043434, UL1 TR000448, R01 EB009352. AV-45 doses were provided by Avid Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly.

Data were [in part] obtained from the IXI dataset ([link](https://brain-development.org/ixi-dataset/)).

Data were [in part] provided by the 1000 Functional Connectomes Project (FCP). For access and usage information, see [link](https://fcon_1000.projects.nitrc.org).

<hr>


