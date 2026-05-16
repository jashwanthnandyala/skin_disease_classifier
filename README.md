# Skin Disease Classifier - Web Application

A Streamlit-based web application for classifying skin diseases using a pre-trained deep learning model.

## Features

✨ **Image Upload & Analysis**
- Upload images of skin conditions (JPG, PNG, BMP)
- Real-time predictions with confidence scores

📊 **Disease Detection**
- Detects 6 types of skin conditions:
  - Acne
  - Carcinoma
  - Eczema
  - Keratosis
  - Milia
  - Rosacea

📝 **Personalized Recommendations**
- Disease-specific next steps and care instructions
- Confidence scores for all detected conditions

Interpretability: Implements Grad-CAM heatmaps to highlight the specific areas of the image the model used for its prediction.

## File Structure

```
skin_disease_classifier_final/
├── app.py                    # Main Streamlit application
├── skin_model.keras          # Trained model (your saved model)
├── train.py                  # Training script
├── train.ipynb              # Training notebook
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── gradcam.py         # explainable AI Implementation
```

## 🛠️ Methodology
1. Data Pipeline & PreprocessingStandardization: All images are resized to 224x224 pixels to match the ResNet50 input requirements.Dataset Splitting: The dataset is programmatically split into Training (80%), Validation (10%), and Testing (10%) sets using the tf.data API to ensure the model generalizes well to unseen data.Batching: Data is processed in batches of 32 images to optimize memory usage during training.

2. Data AugmentationTo prevent overfitting and improve model robustness against variations in image capture, a real-time augmentation pipeline is applied to the training set:Random Horizontal FlippingRandom Rotation (up to 10%)Random Zoom (up to 10%).

3. Model ArchitectureThe system utilizes a Transfer Learning strategy:Base Model: ResNet50 pre-trained on the ImageNet dataset is used as a powerful feature extractor.Custom Head: The top layers are replaced with a Global Average Pooling layer, a Dense layer (1024 units) with ReLU activation, and a Dropout layer (0.5) to further reduce overfitting.Output: A final Dense layer with a Softmax activation function outputs the probability for the 6 skin condition classes.

4. Training StrategyThe training was conducted in two distinct phases:
   Phase 1: Feature Extraction: The ResNet50 base layers were frozen, and only the custom head was trained using the Adam optimizer with a learning rate of $1e-4$.
   Phase 2: Fine-Tuning: The top 20 layers of the ResNet50 base were unfrozen and trained with a much smaller learning rate (1x10-5). This allows the model to adjust its high-level feature maps to the specific textures of skin conditions.

6. Evaluation & ExportPerformance: The final model is evaluated on a dedicated test set to report final accuracy.Persistence: The optimized weights are saved in the modern .keras format for deployment in the Streamlit application.


## Grad-Cam

One of the most critical aspects of this project is its interpretability. Medical AI models shouldn't be "black boxes," so I implemented Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual justifications for every diagnosis.

Interpreting the Results
Warm Colors (Red/Orange): Indicate areas of high importance for the classification.

Cool Colors (Blue/Green): Indicate areas that had little to no impact on the final prediction.
