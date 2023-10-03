## 1. Title and Author

* **Project Title:** Develop a Computationally efficient Transformer based architecture for Facial Expression Recognition(FER) 
* **Prepared for** UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
* Author Name: Kulin Patel
* [LinkedIn](https://www.linkedin.com/in/kulin-patel)

## 2. Background

Facial Expression Recognition (FER) is pivotal in human-computer interaction and emotional analysis. While Convolutional Neural Networks (CNNs) have shown promise in FER, the emergence of Vision Transformers (ViTs) and traditional Transformers designed for language tasks offers new opportunities for improved accuracy. However, these Transformer architectures are often computationally intensive, limiting their real-world applications.

This project aims to develop a computationally efficient Transformer-based FER architecture, drawing inspiration from both ViTs and traditional Transformers. 

The primary objectives of this project are:

* Efficiency: To design a Transformer-based FER architecture that minimizes computational requirements while maintaining competitive accuracy levels.
* Real-time Processing: To enable real-time or near-real-time FER, ensuring the model's suitability for applications such as emotion-aware interfaces and virtual reality.

By addressing these challenges and combining insights from ViTs and Transformers, we can unlock the potential of emotion-aware technology across various domains.

## 3. Data

**Data Source:** [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

**Data size**: 35887 images

**Data size in MB**: 92.5 MB

**Data format**: 48x48 pixel grayscale images of faces

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories 

(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

### 3.1 Data Folder Structure

```bash
├── data
│   ├── archive
│   │   ├── test
│   │   │   ├── angry
│   │   │   ├── disgust
│   │   │   ├── fear
│   │   │   ├── happy
│   │   │   ├── neutral
│   │   │   ├── sad
│   │   │   └── surprise
│   │   └── train
│   │       ├── angry
│   │       ├── disgust
│   │       ├── fear
│   │       ├── happy
│   │       ├── neutral
│   │       ├── sad
│   │       └── surprise
```


