#!/usr/bin/env python3
"""
Training script for H-SGE framework components
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from hsge_framework import (
    HSGEFramework, 
    HSGEConfig, 
    GANEnhancer, 
    HandgunDataset,
    evaluate_hsge
)

def setup_logging(log_file):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to HSGEConfig object
    config = HSGEConfig()
    
    # Update with loaded values
    if 'models' in config_dict:
        for key, value in config_dict['models'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    if 'detection' in config_dict:
        for key, value in config_dict['detection'].items():
            if hasattr(config, key):
                setattr(config, key, value)
                
    if 'training' in config_dict:
        for key, value in config_dict['training'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config, config_dict

def train_gan_enhancer(config, config_dict):
    """Train GAN enhancer component"""
    logger = logging.getLogger(__name__)
    logger.info("Starting GAN enhancer training...")
    
    # Initialize model
    device = torch.device(config_dict.get('hardware', {}).get('device', 'cuda'))
    generator = GANEnhancer().to(device)
    
    # Discriminator for adversarial training
    discriminator = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 1, 4, 1, 0),
        nn.Sigmoid()
    ).to(device)
    
    # Optimizers
    opt_g = optim.AdamW(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    opt_d = optim.AdamW(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()
    
    # Data loading
    train_dataset = HandgunDataset(
        config_dict['data']['train_images'],
        config_dict['data']['train_annotations']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config_dict.get('hardware', {}).get('num_workers', 4)
    )
    
    # Training loop
    losses_g = []
    losses_d = []
    
    for epoch in range(config.epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            # Prepare data (assuming low-quality/high-quality pairs)
            low_qual = batch['image'].to(device).float()
            high_qual = batch.get('target', low_qual).to(device).float()  # Use same image if no target
            
            batch_size = low_qual.size(0)
            
            # Train Discriminator
            opt_d.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
            real_output = discriminator(high_qual)
            real_loss = adversarial_loss(real_output, real_labels)
            
            # Fake images
            fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)
            fake_images = generator(low_qual)
            fake_output = discriminator(fake_images.detach())
            fake_loss = adversarial_loss(fake_output, fake_labels)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            
            # Train Generator
            opt_g.zero_grad()
            
            fake_output = discriminator(fake_images)
            g_adversarial_loss = adversarial_loss(fake_output, real_labels)
            g_reconstruction_loss = reconstruction_loss(fake_images, high_qual)
            
            g_loss = g_adversarial_loss + 100 * g_reconstruction_loss
            g_loss.backward()
            opt_g.step()
            
            epoch_loss_g += g_loss.item()
            epoch_loss_d += d_loss.item()
            
            progress_bar.set_postfix({
                'G_Loss': f"{g_loss.item():.4f}",
                'D_Loss': f"{d_loss.item():.4f}"
            })
        
        avg_loss_g = epoch_loss_g / len(train_loader)
        avg_loss_d = epoch_loss_d / len(train_loader)
        
        losses_g.append(avg_loss_g)
        losses_d.append(avg_loss_d)
        
        logger.info(f"Epoch {epoch+1}: G_Loss={avg_loss_g:.4f}, D_Loss={avg_loss_d:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            torch.save(generator.state_dict(), 
                      checkpoint_dir / f"gan_generator_epoch_{epoch+1}.pth")
    
    # Save final model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), models_dir / "gan_enhancer.pth")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses_g, label='Generator Loss')
    plt.plot(losses_d, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()
    
    logger.info("GAN training completed!")

def fine_tune_yolo_models(config, config_dict):
    """Fine-tune YOLO models on handgun dataset"""
    logger = logging.getLogger(__name__)
    logger.info("Fine-tuning YOLO models...")
    
    from ultralytics import YOLO
    
    models = {
        'yolov5': config.yolov5_path,
        'yolov7': config.yolov7_path,
        'yolo10': config.yolo10_path,
        'yolo11': config.yolo11_path
    }
    
    # Prepare dataset in YOLO format
    dataset_yaml = {
        'train': config_dict['data']['train_images'],
        'val': config_dict['data']['val_images'],
        'nc': 1,  # Number of classes (handgun)
        'names': ['handgun']
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    # Fine-tune each model
    for model_name, model_path in models.items():
        logger.info(f"Fine-tuning {model_name}...")
        
        try:
            model = YOLO(model_path)
            
            # Train
            results = model.train(
                data='dataset.yaml',
                epochs=50,
                imgsz=640,
                batch=config.batch_size,
                lr0=config.learning_rate,
                name=f'{model_name}_handgun',
                save=True,
                cache=True
            )
            
            # Save fine-tuned model
            model.save(f"models/{model_name}_handgun.pt")
            
            logger.info(f"{model_name} fine-tuning completed!")
            
        except Exception as e:
            logger.error(f"Failed to fine-tune {model_name}: {e}")

def evaluate_framework(config, config_dict):
    """Evaluate complete H-SGE framework"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating H-SGE framework...")
    
    # Initialize framework
    hsge = HSGEFramework(config)
    
    # Load test dataset
    test_dataset = HandgunDataset(
        config_dict['data']['test_images'],
        config_dict['data']['test_annotations']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for evaluation
        shuffle=False
    )
    
    # Evaluate
    metrics = evaluate_hsge(hsge, test_loader)
    
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    results_file = Path("results") / "evaluation_results.yaml"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        yaml.dump(metrics, f)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train H-SGE Framework")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=['gan', 'yolo', 'full', 'eval'],
                       default='full', help="Training mode")
    parser.add_argument("--log_file", type=str, default="logs/training.log",
                       help="Path to log file")
    
    args = parser.parse_args()
    
    # Setup logging
    Path(args.log_file).parent.mkdir(exist_ok=True)
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config, config_dict = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create necessary directories
    for directory in ['models', 'checkpoints', 'logs', 'results']:
        Path(directory).mkdir(exist_ok=True)
    
    # Execute based on mode
    if args.mode == 'gan' or args.mode == 'full':
        train_gan_enhancer(config, config_dict)
    
    if args.mode == 'yolo' or args.mode == 'full':
        fine_tune_yolo_models(config, config_dict)
    
    if args.mode == 'eval' or args.mode == 'full':
        evaluate_framework(config, config_dict)
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main()
