# ExerciseClassification-MGAP
The code repository for "Video-based Exercise Classification and Activated Muscle Group Prediction with Hybrid X3D-SlowFast Network".

The code is presented in the Jupyter Notebook format and was implemented in Google Colab. Further information is found in the notebook files.

The main guides used are PyTorchVideo and PyTorch Lightning. These packages are also used in the implementation.

**Abstract**
This paper introduces a simple yet effective strategy for exercise classification and muscle group activation prediction (MGAP).
These tasks have significant implications for personal fitness, facilitating more affordable, accessible, safer, and simpler
exercise routines. This is particularly relevant for novices and individuals with disabilities. Previous research in the field is
mostly dominated by the reliance on mounted sensors and a limited scope of exercises, reducing practicality for everyday use.
Furthermore, existing MGAP methodologies suffer from a similar dependency on sensors and a restricted range of muscle
groups, often excluding strength training exercises, which are pivotal for a comprehensive fitness regimen. Addressing these
limitations, our research employs a video-based deep learning framework that encompasses a broad spectrum of exercises
and muscle groups, including those vital for strength training. Utilizing the "Workout/Exercises Video" dataset, our approach
integrates the X3D and SlowFast video activity recognition models in an effective way to enhance exercise classification and
MGAP performance. Our findings demonstrate that this hybrid method obtained via a weighted ensemble outperforms existing
baseline models in accuracy. Pretrained models play a crucial role in enhancing overall performance, with optimal channel
reduction values for the SlowFast model identified near 10. Through an ablation study that explores fine-tuning, we further
elucidate the interrelation between the two tasks. Our composite model, a weighted-average ensemble of X3D and SlowFast,
sets a new benchmark in both exercise classification and MGAP across all evaluated categories, offering a robust solution to
the limitations of previous approaches.
