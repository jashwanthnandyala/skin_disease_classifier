"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for visualizing model predictions
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """Compute Grad-CAM heatmap for a given model and image"""
    
    def __init__(self, model, layer_name=None):
        """
        Args:
            model: TensorFlow/Keras model
            layer_name: Name of the convolutional layer to compute gradients for
                       If None, uses the last convolutional layer
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"Using layer: {self.layer_name}")
    
    def compute_heatmap(self, img_array, pred_index=None):
        """
        Compute Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image array (batch_size, height, width, channels)
            pred_index: Index of the class to compute gradients for.
                       If None, uses the predicted class
        
        Returns:
            heatmap: Normalized heatmap (height, width)
        """
        # Create a model that outputs both predictions and feature maps
        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Get the class index
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the class channel
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Average gradient across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get feature maps
        conv_outputs = conv_outputs[0]
        
        # Weight feature maps by gradients
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap to [0, 1]
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
    
    def overlay_heatmap(self, original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            original_image: PIL Image or numpy array of the original image
            heatmap: Normalized heatmap (height, width)
            alpha: Transparency of the heatmap overlay
            colormap: OpenCV colormap to use
        
        Returns:
            overlaid_image: PIL Image with heatmap overlay
        """
        # Convert PIL image to numpy if needed
        if isinstance(original_image, Image.Image):
            img_np = np.array(original_image)
        else:
            img_np = original_image
        
        # Ensure image is in correct format (height, width, 3)
        if img_np.shape[-1] == 4:  # RGBA
            img_np = img_np[:, :, :3]
        
        # Get original dimensions
        img_height, img_width = img_np.shape[:2]
        
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(
            heatmap,
            (img_width, img_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Convert heatmap to 0-255 range
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Convert BGR to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Convert back to PIL Image
        return Image.fromarray(np.uint8(overlay))
    
    def generate_heatmap_visualization(self, original_image, img_array, pred_index=None):
        """
        Generate both heatmap and overlaid image in one call
        
        Args:
            original_image: PIL Image of the original
            img_array: Preprocessed image array
            pred_index: Index of the class for gradient computation
        
        Returns:
            heatmap: Raw heatmap numpy array
            overlay_image: PIL Image with heatmap overlay
        """
        heatmap = self.compute_heatmap(img_array, pred_index)
        overlay_image = self.overlay_heatmap(original_image, heatmap, alpha=0.5)
        return heatmap, overlay_image
