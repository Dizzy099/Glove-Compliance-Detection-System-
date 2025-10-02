#!/usr/bin/env python3
"""
Gloved vs Ungloved Hand Detection System
Technical Assessment - Part 1

Author: Technical Assessment Submission
Date: September 2025
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from ultralytics import YOLO
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GloveHandDetector:
    """
    Production-ready glove detection system using YOLOv8 + custom classification.
    Combines object detection with color/texture analysis for robust glove identification.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5, device: str = 'auto'):
        """
        Initialize the glove detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: Computing device ('auto', 'cpu', 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        
        # Auto-select device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Successfully loaded model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Define colors for visualization
        self.colors = {
            'gloved_hand': (0, 255, 0),    # Green
            'bare_hand': (255, 0, 0),      # Red
        }
        
        # Define glove color ranges in HSV space
        self.glove_color_ranges = [
            # Blue gloves (medical, industrial)
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # Yellow/orange gloves (safety)
            (np.array([10, 100, 100]), np.array([35, 255, 255])),
            # White gloves (food service, medical)
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Green gloves (gardening, industrial)
            (np.array([40, 50, 50]), np.array([80, 255, 255])),
            # Purple gloves (medical)
            (np.array([130, 50, 50]), np.array([160, 255, 255])),
            # Black gloves
            (np.array([0, 0, 0]), np.array([180, 255, 50]))
        ]
        
        # Define skin tone ranges
        self.skin_color_ranges = [
            (np.array([0, 20, 70]), np.array([20, 255, 255])),
            (np.array([160, 20, 70]), np.array([180, 255, 255]))
        ]
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects using YOLO and filter for hand-related detections.
        """
        # Run YOLO inference
        results = self.model(image, conf=0.1, device=self.device)  # Low conf for initial detection
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Focus on person detections and extract hand regions
                    if class_name == 'person' and confidence > 0.3:
                        hand_regions = self._extract_hand_regions(image, [x1, y1, x2, y2])
                        detections.extend(hand_regions)
        
        return detections
    
    def _extract_hand_regions(self, image: np.ndarray, person_bbox: List[float]) -> List[Dict]:
        """
        Extract potential hand regions from person detection.
        """
        x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
        person_roi = image[y1:y2, x1:x2]
        
        hand_regions = []
        
        if person_roi.size == 0:
            return hand_regions
        
        h, w = person_roi.shape[:2]
        
        # Define hand regions based on typical human pose
        potential_hand_areas = [
            # Left hand (person's right side)
            (0, int(h*0.1), int(w*0.4), int(h*0.8)),
            # Right hand (person's left side)
            (int(w*0.6), int(h*0.1), w, int(h*0.8)),
            # Raised hands area
            (int(w*0.2), 0, int(w*0.8), int(h*0.5)),
        ]
        
        for i, (hx1, hy1, hx2, hy2) in enumerate(potential_hand_areas):
            # Convert to absolute coordinates
            abs_x1 = x1 + hx1
            abs_y1 = y1 + hy1
            abs_x2 = x1 + hx2
            abs_y2 = y1 + hy2
            
            # Extract region
            if abs_y2 > abs_y1 and abs_x2 > abs_x1:
                hand_roi = image[abs_y1:abs_y2, abs_x1:abs_x2]
                
                if hand_roi.size > 500:  # Minimum size threshold
                    # Classify this region
                    classification = self._classify_hand_region(hand_roi)
                    if classification['confidence'] > 0.3:
                        hand_regions.append({
                            'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                            'label': classification['label'],
                            'confidence': classification['confidence'],
                            'scores': classification['scores']
                        })
        
        return hand_regions
    
    def _classify_hand_region(self, roi: np.ndarray) -> Dict:
        """
        Classify a region as gloved_hand or bare_hand using multiple methods.
        """
        if roi.size == 0:
            return {'label': 'bare_hand', 'confidence': 0.0, 'scores': {}}
        
        # Method 1: Color analysis
        color_score = self._analyze_colors(roi)
        
        # Method 2: Texture analysis
        texture_score = self._analyze_texture(roi)
        
        # Method 3: Edge analysis
        edge_score = self._analyze_edges(roi)
        
        # Combine scores with weights
        weights = {'color': 0.5, 'texture': 0.3, 'edge': 0.2}
        final_score = (
            color_score * weights['color'] +
            texture_score * weights['texture'] +
            edge_score * weights['edge']
        )
        
        # Determine label
        label = 'gloved_hand' if final_score > 0.5 else 'bare_hand'
        
        # Confidence based on score certainty
        confidence = 0.5 + abs(final_score - 0.5)
        
        return {
            'label': label,
            'confidence': min(0.95, confidence),
            'scores': {
                'color': color_score,
                'texture': texture_score,
                'edge': edge_score,
                'final': final_score
            }
        }
    
    def _analyze_colors(self, roi: np.ndarray) -> float:
        """
        Analyze colors to determine if region contains gloves.
        Returns score between 0 (skin-like) and 1 (glove-like).
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # Count glove-colored pixels
        glove_pixels = 0
        for lower, upper in self.glove_color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            glove_pixels += cv2.countNonZero(mask)
        
        # Count skin-colored pixels
        skin_pixels = 0
        for lower, upper in self.skin_color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            skin_pixels += cv2.countNonZero(mask)
        
        # Calculate ratio
        total_classified = glove_pixels + skin_pixels
        if total_classified == 0:
            return 0.5  # Neutral if no clear colors detected
        
        return glove_pixels / total_classified
    
    def _analyze_texture(self, roi: np.ndarray) -> float:
        """
        Analyze texture patterns. Gloves typically have more uniform texture.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance (lower = more uniform = more glove-like)
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        avg_variance = np.mean(variance)
        
        # Normalize: lower variance = higher glove score
        texture_score = max(0, min(1, 1 - avg_variance / 1000))
        
        return texture_score
    
    def _analyze_edges(self, roi: np.ndarray) -> float:
        """
        Analyze edge patterns. Gloves often have clearer boundaries.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Higher edge density suggests clearer boundaries (gloves)
        return min(1.0, edge_density * 5)
    
    def process_image(self, image_path: str, output_dir: str, logs_dir: str) -> Dict:
        """
        Process a single image: detect hands, classify, annotate, and log.
        """
        filename = os.path.basename(image_path)
        logger.info(f"Processing: {filename}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Detect and classify
        detections = self.detect_objects(image)
        
        # Filter by confidence threshold
        filtered_detections = [
            det for det in detections 
            if det['confidence'] >= self.confidence_threshold
        ]
        
        # Apply NMS to remove overlapping detections
        final_detections = self._apply_nms(filtered_detections)
        
        # Create detection data in required format
        detection_data = []
        for det in final_detections:
            detection_data.append({
                "label": det['label'],
                "confidence": float(round(det['confidence'], 3)),
                "bbox": [round(coord, 1) for coord in det['bbox']]
            })
        
        # Annotate image
        annotated = self._annotate_image(image, final_detections)
        
        # Save annotated image
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, annotated)
        
        # Create and save log
        log_data = {
            "filename": filename,
            "detections": detection_data
        }
        
        os.makedirs(logs_dir, exist_ok=True)
        log_filename = os.path.splitext(filename)[0] + '.json'
        log_path = os.path.join(logs_dir, log_filename)
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"✅ {filename}: {len(final_detections)} detections")
        return log_data
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for det in detections:
            should_keep = True
            for kept_det in keep:
                iou = self._calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det)
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union of two bounding boxes.
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            label = det['label']
            confidence = det['confidence']
            
            # Get color
            color = self.colors[label]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Background rectangle
            cv2.rectangle(annotated, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(annotated, text, (x1, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    def process_folder(self, input_dir: str, output_dir: str, logs_dir: str, use_multiprocessing: bool = False):
        """
        Process all images in a folder.
        """
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        if use_multiprocessing and len(image_files) > 1:
            self._process_with_multiprocessing(image_files, output_dir, logs_dir)
        else:
            for image_file in image_files:
                self.process_image(str(image_file), output_dir, logs_dir)
        
        logger.info("✅ Batch processing complete")
    
    def _process_with_multiprocessing(self, image_files: List[Path], output_dir: str, logs_dir: str):
        """
        Process images using multiprocessing for speed.
        """
        n_processes = min(mp.cpu_count(), 4)  # Limit to 4 processes
        logger.info(f"Using {n_processes} processes for batch processing")
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            for image_file in image_files:
                future = executor.submit(self.process_image, str(image_file), output_dir, logs_dir)
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Processing failed: {e}")


def create_sample_images():
    """
    Create sample test images for demonstration.
    """
    logger.info("Creating sample images for testing...")
    
    os.makedirs("sample_images", exist_ok=True)
    
    # Sample 1: Blue gloves
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 240
    cv2.rectangle(img1, (100, 150), (200, 300), (255, 100, 0), -1)  # Blue glove
    cv2.rectangle(img1, (400, 100), (500, 250), (255, 120, 20), -1)  # Another blue glove
    cv2.imwrite("sample_images/blue_gloves.jpg", img1)
    
    # Sample 2: Bare hands
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 220
    cv2.rectangle(img2, (150, 120), (250, 280), (120, 180, 220), -1)  # Skin tone
    cv2.rectangle(img2, (350, 150), (450, 320), (100, 160, 210), -1)  # Another skin tone
    cv2.imwrite("sample_images/bare_hands.jpg", img2)
    
    # Sample 3: Mixed
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.rectangle(img3, (80, 100), (180, 250), (255, 100, 0), -1)    # Blue glove
    cv2.rectangle(img3, (400, 120), (500, 270), (120, 180, 220), -1) # Bare hand
    cv2.rectangle(img3, (250, 200), (350, 350), (0, 255, 255), -1)   # Yellow glove
    cv2.imwrite("sample_images/mixed_hands.jpg", img3)
    
    logger.info("✅ Sample images created in sample_images/")


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Glove Detection System for Safety Compliance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python detection_script.py --input images/ --output output/ --logs logs/
  
  # With custom confidence threshold
  python detection_script.py --input images/ --confidence 0.7
  
  # Enable multiprocessing
  python detection_script.py --input images/ --multiprocessing
        """
    )
    
    parser.add_argument('--input', required=True, 
                       help='Input directory containing .jpg images')
    parser.add_argument('--output', default='output',
                       help='Output directory for annotated images')
    parser.add_argument('--logs', default='logs',
                       help='Directory for JSON detection logs')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold (default: 0.5)')
    parser.add_argument('--model', default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Computing device (default: auto)')
    parser.add_argument('--multiprocessing', action='store_true',
                       help='Enable multiprocessing for batch processing')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample images for testing')
    
    args = parser.parse_args()
    
    print("🚀 Glove Detection System v1.0")
    print("=" * 50)
    
    # Create sample images if requested
    if args.create_samples:
        create_sample_images()
        return
    
    # Validate input directory
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return
    
    try:
        # Initialize detector
        detector = GloveHandDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device
        )
        
        # Process images
        detector.process_folder(
            input_dir=args.input,
            output_dir=args.output,
            logs_dir=args.logs,
            use_multiprocessing=args.multiprocessing
        )
        
        logger.info("🎉 Processing complete! Check output/ and logs/ directories.")
        
    except Exception as e:
        logger.error(f"Failed to process images: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()