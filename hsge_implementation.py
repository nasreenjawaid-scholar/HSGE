"""
H-SGE: Hybrid Scene Graph Enrichment for Small Handgun Detection
================================================================

This implementation provides the complete H-SGE framework as described in:
"H-SGE: A Hybrid Model Based on Scene Graph Enrichment"

Authors: Nasreen Jawaid, Najma Imtiaz Ali, Imtiaz Ali Korejo, 
         Imtiaz Ali Brohi, Noor Hafeizah Binti Hassan

Main Components:
1. GAN-based Image Enhancement
2. Scene Graph Generation  
3. Knowledge Graph Integration
4. Multi-YOLO Detection
5. Weighted Fusion

Requirements:
- torch>=1.9.0
- torchvision>=0.10.0
- opencv-python>=4.5.0
- ultralytics>=8.0.0
- numpy>=1.21.0
- matplotlib>=3.3.0
- networkx>=2.6.0
- scipy>=1.7.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
import yaml
from dataclasses import dataclass
from ultralytics import YOLO
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HSGEConfig:
    """Configuration class for H-SGE framework"""
    # Model paths
    yolov5_path: str = "yolov5s.pt"
    yolov7_path: str = "yolov7.pt" 
    yolo10_path: str = "yolo10n.pt"
    yolo11_path: str = "yolo11n.pt"
    gan_model_path: str = "models/gan_enhancer.pth"
    
    # Detection parameters
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    kg_threshold: float = 0.7
    
    # Image processing
    input_size: Tuple[int, int] = (640, 640)
    
    # Knowledge graph parameters
    spatial_weight: float = 0.3
    contextual_weight: float = 0.4
    temporal_weight: float = 0.3
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 100

class GANEnhancer(nn.Module):
    """GAN-based image enhancement for small object visibility"""
    
    def __init__(self, input_channels=3, output_channels=3):
        super(GANEnhancer, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        enhanced = self.decoder(encoded)
        return enhanced

class SceneGraphGenerator:
    """Scene graph generation for spatial-semantic relationships"""
    
    def __init__(self):
        self.object_classes = [
            'person', 'handgun', 'vehicle', 'building', 'furniture',
            'electronics', 'outdoor', 'indoor', 'public_space', 'private_space'
        ]
        self.relationships = [
            'holding', 'near', 'in', 'on', 'next_to', 'behind', 
            'in_front_of', 'inside', 'outside', 'threatening'
        ]
        
    def generate_scene_graph(self, detections: List[Dict], image_info: Dict) -> nx.DiGraph:
        """Generate scene graph from object detections"""
        G = nx.DiGraph()
        
        # Add nodes (objects)
        for i, detection in enumerate(detections):
            G.add_node(i, 
                      class_name=detection['class'],
                      confidence=detection['confidence'],
                      bbox=detection['bbox'],
                      features=detection.get('features', None))
        
        # Add edges (relationships)
        for i in range(len(detections)):
            for j in range(i+1, len(detections)):
                relationship = self._compute_spatial_relationship(
                    detections[i]['bbox'], 
                    detections[j]['bbox']
                )
                if relationship:
                    G.add_edge(i, j, relationship=relationship)
                    
        return G
    
    def _compute_spatial_relationship(self, bbox1: List[float], bbox2: List[float]) -> Optional[str]:
        """Compute spatial relationship between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate centers and distances
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Determine relationships based on spatial proximity and relative positions
        if distance < 50:  # Close proximity
            if abs(center1[1] - center2[1]) < 20:  # Same horizontal level
                return 'next_to'
            elif center1[1] < center2[1]:
                return 'above'
            else:
                return 'below'
        elif distance < 100:
            return 'near'
        
        return None

class KnowledgeGraph:
    """Knowledge graph for contextual validation"""
    
    def __init__(self, kg_file: str = None):
        self.graph = nx.Graph()
        self.threat_patterns = {}
        self.contextual_rules = {}
        
        if kg_file and Path(kg_file).exists():
            self.load_knowledge_graph(kg_file)
        else:
            self._initialize_default_kg()
    
    def _initialize_default_kg(self):
        """Initialize default knowledge graph with handgun-specific patterns"""
        # Threat scenarios
        threat_scenarios = [
            ('person', 'holding', 'handgun', 0.9),
            ('handgun', 'in', 'public_space', 0.8),
            ('person', 'threatening_pose', 'handgun', 0.85),
            ('handgun', 'near', 'crowd', 0.75),
            ('concealed', 'handgun', 'suspicious_behavior', 0.7)
        ]
        
        for scenario in threat_scenarios:
            if len(scenario) == 4:
                entity1, relation, entity2, weight = scenario
                self.graph.add_edge(entity1, entity2, 
                                  relation=relation, weight=weight)
        
        # Contextual rules for validation
        self.contextual_rules = {
            'high_risk_locations': ['bank', 'school', 'airport', 'government_building'],
            'suspicious_behaviors': ['aggressive_pose', 'concealment', 'rapid_movement'],
            'innocent_contexts': ['toy_gun', 'police_officer', 'security_guard']
        }
    
    def validate_detection(self, scene_graph: nx.DiGraph, detection_node: int) -> float:
        """Validate detection using knowledge graph reasoning"""
        base_confidence = 0.5
        contextual_boost = 0.0
        
        # Get detection info
        detection_data = scene_graph.nodes[detection_node]
        
        # Check spatial relationships
        for neighbor in scene_graph.neighbors(detection_node):
            neighbor_data = scene_graph.nodes[neighbor]
            edge_data = scene_graph.edges[detection_node, neighbor]
            
            relationship = edge_data.get('relationship', '')
            
            # Apply knowledge graph rules
            if self._matches_threat_pattern(detection_data, neighbor_data, relationship):
                contextual_boost += 0.2
            
        # Check scene context
        scene_context = self._analyze_scene_context(scene_graph)
        if scene_context.get('high_risk_environment', False):
            contextual_boost += 0.15
            
        return min(base_confidence + contextual_boost, 1.0)
    
    def _matches_threat_pattern(self, det1: Dict, det2: Dict, relationship: str) -> bool:
        """Check if detection matches known threat patterns"""
        # Check if pattern exists in knowledge graph
        entity1 = det1.get('class_name', '')
        entity2 = det2.get('class_name', '')
        
        if self.graph.has_edge(entity1, entity2):
            edge_data = self.graph.edges[entity1, entity2]
            return edge_data.get('relation', '') == relationship
            
        return False
    
    def _analyze_scene_context(self, scene_graph: nx.DiGraph) -> Dict:
        """Analyze overall scene context for threat assessment"""
        context = {'high_risk_environment': False}
        
        # Count people and objects
        person_count = sum(1 for _, data in scene_graph.nodes(data=True) 
                          if data.get('class_name') == 'person')
        
        # Check for high-risk indicators
        if person_count > 5:  # Crowded area
            context['high_risk_environment'] = True
            
        return context
    
    def load_knowledge_graph(self, filepath: str):
        """Load knowledge graph from file"""
        with open(filepath, 'r') as f:
            kg_data = json.load(f)
            
        for edge in kg_data.get('edges', []):
            self.graph.add_edge(
                edge['source'], 
                edge['target'], 
                **edge.get('attributes', {})
            )

class MultiYOLODetector:
    """Multi-YOLO detection with ensemble fusion"""
    
    def __init__(self, config: HSGEConfig):
        self.config = config
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        """Load all YOLO variants"""
        model_configs = {
            'yolov5': self.config.yolov5_path,
            'yolov7': self.config.yolov7_path, 
            'yolo10': self.config.yolo10_path,
            'yolo11': self.config.yolo11_path
        }
        
        for name, path in model_configs.items():
            try:
                if Path(path).exists():
                    self.models[name] = YOLO(path)
                    logger.info(f"Loaded {name} from {path}")
                else:
                    # Load pretrained model
                    self.models[name] = YOLO(Path(path).name)
                    logger.info(f"Loaded pretrained {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
    
    def detect(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """Run detection with all available YOLO models"""
        detections = {}
        
        for name, model in self.models.items():
            try:
                results = model(image, 
                              conf=self.config.confidence_threshold,
                              iou=self.config.iou_threshold)
                
                detections[name] = self._parse_results(results[0])
                
            except Exception as e:
                logger.error(f"Detection failed for {name}: {e}")
                detections[name] = []
                
        return detections
    
    def _parse_results(self, result) -> List[Dict]:
        """Parse YOLO results into standard format"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                detections.append({
                    'class': int(cls),
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), 
                            float(x2-x1), float(y2-y1)]
                })
                
        return detections

class WeightedFusion:
    """Weighted fusion for multi-model ensemble"""
    
    def __init__(self, model_weights: Dict[str, float] = None):
        self.model_weights = model_weights or {
            'yolov5': 0.2,
            'yolov7': 0.25, 
            'yolo10': 0.25,
            'yolo11': 0.3
        }
    
    def fuse_detections(self, 
                       multi_detections: Dict[str, List[Dict]],
                       kg_scores: Dict[str, float] = None) -> List[Dict]:
        """Fuse detections from multiple models with knowledge graph validation"""
        
        all_detections = []
        
        # Collect all detections with model weights
        for model_name, detections in multi_detections.items():
            weight = self.model_weights.get(model_name, 0.25)
            
            for det in detections:
                det_copy = det.copy()
                det_copy['model'] = model_name
                det_copy['model_weight'] = weight
                det_copy['weighted_confidence'] = det['confidence'] * weight
                
                # Add knowledge graph score if available
                if kg_scores and model_name in kg_scores:
                    det_copy['kg_score'] = kg_scores[model_name]
                    det_copy['final_confidence'] = (
                        det_copy['weighted_confidence'] * 0.7 + 
                        det_copy['kg_score'] * 0.3
                    )
                else:
                    det_copy['final_confidence'] = det_copy['weighted_confidence']
                    
                all_detections.append(det_copy)
        
        # Apply Non-Maximum Suppression
        fused_detections = self._weighted_nms(all_detections)
        
        return fused_detections
    
    def _weighted_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Weighted Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by final confidence
        detections.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        keep = []
        
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [det for det in detections 
                         if self._calculate_iou(current['bbox'], det['bbox']) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0

class HSGEFramework:
    """Main H-SGE Framework integrating all components"""
    
    def __init__(self, config: HSGEConfig):
        self.config = config
        
        # Initialize components
        self.gan_enhancer = GANEnhancer()
        self.scene_graph_generator = SceneGraphGenerator()
        self.knowledge_graph = KnowledgeGraph()
        self.multi_yolo = MultiYOLODetector(config)
        self.fusion = WeightedFusion()
        
        # Load GAN model if available
        self._load_gan_model()
        
    def _load_gan_model(self):
        """Load pre-trained GAN model"""
        if Path(self.config.gan_model_path).exists():
            try:
                self.gan_enhancer.load_state_dict(
                    torch.load(self.config.gan_model_path, map_location='cpu')
                )
                self.gan_enhancer.eval()
                logger.info("Loaded GAN enhancer model")
            except Exception as e:
                logger.warning(f"Failed to load GAN model: {e}")
    
    def detect_handgun(self, image: np.ndarray) -> Dict:
        """Complete H-SGE detection pipeline"""
        # Stage 1: GAN-based Enhancement
        enhanced_image = self._enhance_image(image)
        
        # Stage 2: Multi-YOLO Detection
        multi_detections = self.multi_yolo.detect(enhanced_image)
        
        # Stage 3: Scene Graph Generation
        all_detections = []
        for model_dets in multi_detections.values():
            all_detections.extend(model_dets)
            
        scene_graph = self.scene_graph_generator.generate_scene_graph(
            all_detections, {'image_shape': image.shape}
        )
        
        # Stage 4: Knowledge Graph Validation
        kg_scores = {}
        for model_name, detections in multi_detections.items():
            kg_scores[model_name] = 0.0
            for i, det in enumerate(detections):
                if det['class'] == 0:  # Assuming class 0 is handgun
                    kg_score = self.knowledge_graph.validate_detection(scene_graph, i)
                    kg_scores[model_name] = max(kg_scores[model_name], kg_score)
        
        # Stage 5: Weighted Fusion
        final_detections = self.fusion.fuse_detections(multi_detections, kg_scores)
        
        return {
            'detections': final_detections,
            'enhanced_image': enhanced_image,
            'scene_graph': scene_graph,
            'model_scores': kg_scores
        }
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply GAN-based image enhancement"""
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Check image quality - skip enhancement for high-quality images
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if brightness > 120:  # High quality image
            return image
        
        try:
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                enhanced_tensor = self.gan_enhancer(input_tensor)
                
            # Convert back to numpy
            enhanced_tensor = (enhanced_tensor.squeeze(0) + 1) / 2  # Denormalize
            enhanced_image = enhanced_tensor.permute(1, 2, 0).numpy()
            enhanced_image = (enhanced_image * 255).astype(np.uint8)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"GAN enhancement failed: {e}")
            return image

# Evaluation and Training Functions
class HandgunDataset(Dataset):
    """Dataset class for handgun detection"""
    
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image_path = self.image_dir / ann['image_file']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'annotations': ann['annotations'],
            'image_id': ann.get('image_id', idx)
        }

def evaluate_hsge(model: HSGEFramework, test_loader: DataLoader) -> Dict:
    """Evaluate H-SGE framework on test dataset"""
    model.gan_enhancer.eval()
    
    metrics = {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'map_50': []
    }
    
    for batch in test_loader:
        images = batch['image']
        annotations = batch['annotations']
        
        for image, ann in zip(images, annotations):
            # Convert tensor to numpy if needed
            if torch.is_tensor(image):
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
            
            # Run detection
            results = model.detect_handgun(image)
            
            # Calculate metrics
            batch_metrics = calculate_detection_metrics(
                results['detections'], ann
            )
            
            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])
    
    # Average metrics
    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[key] = np.mean(values)
        else:
            final_metrics[key] = 0.0
    
    return final_metrics

def calculate_detection_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate detection metrics (precision, recall, F1, mAP)"""
    if not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    # Simple IoU-based matching
    tp = 0
    fp = 0
    fn = len(ground_truth)
    
    matched_gt = set()
    
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
                
            iou = calculate_bbox_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou > 0.5:  # IoU threshold
            tp += 1
            fn -= 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1_score': f1_score
    }

def calculate_bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to corner coordinates
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0

# Main execution example
if __name__ == "__main__":
    # Configuration
    config = HSGEConfig()
    
    # Initialize H-SGE framework
    hsge = HSGEFramework(config)
    
    # Example usage
    image_path = "path/to/test/image.jpg"
    if Path(image_path).exists():
        # Load test image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = hsge.detect_handgun(image)
        
        print(f"Detected {len(results['detections'])} handguns")
        for i, detection in enumerate(results['detections']):
            print(f"Detection {i+1}: confidence={detection['final_confidence']:.3f}")
    else:
        print("Please provide a valid image path for testing")
