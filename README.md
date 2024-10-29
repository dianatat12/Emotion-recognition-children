# Automatic Emotion Recognition in Children Aged 3-5 Years

## Overview

This thesis investigates facial emotion recognition in young children (aged 3-5) and introduces two labeled datasets with three emotion classes: **happiness**, **surprise**, and **frustration**. This research addresses the challenges in this under-represented age group, often due to the high cost of dataset creation, ethical considerations, and privacy concerns.

Most studies focusing on emotion recognition in children utilize the 2016 OpenFace toolkit, which has limitations with modern advancements in computer vision, particularly with deep learning techniques. These newer methods improve performance, especially in challenging conditions like occluded faces.

## Objectives

This study explores two primary hypotheses:
1. **Testing OpenFace on Children**: To determine if OpenFace, which is pre-trained on adult faces, accurately classifies children’s emotions.
2. **Evaluating Alternative Techniques**: To compare the accuracy of advanced deep learning models with the traditional landmark method in recognizing children’s facial emotions.

## Datasets

- **EmoReact**: An open dataset, filtered for young children.
- **App2Five**: A custom dataset collected in a prior study.

## Findings

### Hypothesis 1
- **Outcome**: The hypothesis that OpenFace could accurately classify children’s emotions was **false**. Action Units and OpenFace results were unreliable for this age group.

### Hypothesis 2
- **Pipeline**: A comprehensive pipeline was developed to test accuracy with:
  - Inference on a state-of-the-art model.
  - Training custom convolutional layers.
  - Training models from scratch.
  - Cross-data evaluation.

- **Results**: 
  - Best accuracy of **65%** was achieved when training from scratch.
  - Cross-data evaluation yielded **54%** accuracy.
  - Accuracy in this study reached the **60% range**, slightly below the **62%** from state-of-the-art models with eight emotion classes. The limited dataset (14 children, 3 emotion classes) impacted diversity and generalization potential.

These results underscore the inherent subjectivity in emotion classification, even among human coders (inter-rater reliability: 0.3).

## Additional Experiments

Based on the feedback I received after defending the thesis, I conducted two additional experiments. These experiments involved splitting subjects based on gender and analyzing the impact of this split on emotion classification accuracy. The results of these additional analyses provide further insights into the variability of emotion recognition performance across different demographic groups.

- **Gender split results**: 
