# Privileged learning via a multi-task distilled approach

[python-img]: https://img.shields.io/badge/Made%20with-Python-blue
[ama-img]: https://img.shields.io/badge/Ask%20me-anything-yellowgreen
[wip-img]:https://img.shields.io/badge/Work%20in%20progress-8A2BE2

![Made with Python][python-img]
![Ask me anything][ama-img]
![Work in progress][wip-img]

This repository contains the code for the paper _"Privileged learning via a multi-task distilled approach"_. The learning using privileged information paradigm leverages relevant features unavailable at deployment time for model training. In this paper, we propose a multi-task privileged framework that combines two types of tasks. First, the privileged-prediction task involves using regular features (available in both training and deployment) to predict privileged information, working as an intermediate step to guide the learning process. Second, the main learning objective, the target task, uses the predicted privileged information along with the regular features to make the final target prediction. Furthermore, knowledge distillation techniques are included within the target task to enhance the knowledge transfer of privileged information. Experimental results show improvements in tabular datasets and image-related problems compared to state-of-the-art approaches. Additionally, new metrics are introduced to analyze misclassification causes and refine the proposed multi-task privileged learning to correct errors. 

<img width="1228" alt="Screenshot 2025-03-16 at 12 36 52" src="https://github.com/user-attachments/assets/1517f7fe-3352-47ea-99ab-84c7b2ca6bb3" />

<img width="1219" alt="Screenshot 2025-03-16 at 12 37 27" src="https://github.com/user-attachments/assets/072a927f-3a8a-4125-81af-79a583961b32" />

## Content

- `DC_DR.py`. Interpretability for privileged multi-task learning. 
- `MT_weightP.py`. Handles weight processing for the MT model.  
- `MT_main.py`. Main script for running the model.  
- `datasets.py`. Manages dataset loading and preprocessing.  
- `models.py`. Defines model architectures.  
- `utils.py`. Contains utility functions for various tasks.  
- `download_datasets.py`. Script for downloading required datasets.  
- `requirements.txt`. Lists dependencies needed for the project.  

## Contact

Mario Martínez García - mmartinez@bcamath.org



## Citation

The corresponding BiBTeX citation is given below:
