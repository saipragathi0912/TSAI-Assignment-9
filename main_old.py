"""
ResNet50 Training Script - Properly Formatted
Complete training pipeline with checkpointing, logging, and validation
"""

import os
import sys
import math
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import ImageFile
from tqdm import tqdm
from datetime import datetime
from model import ResNet50

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars"""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(log_dir='logs'):
    """
    Setup logging to both file and console with timestamp
    
    Args:
        log_dir: Directory to store log files
    
    Returns:
        log_filename: Path to the created log file
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging format
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (compatible with tqdm)
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return log_filename


# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


def load_datasets(data_dir='/data/', batch_size=256):
    """
    Load training and validation datasets with proper logging
    
    Args:
        data_dir: Root directory containing train/ and val/ folders
        batch_size: Batch size for data loaders
    
    Returns:
        train_loader, val_loader, num_classes, class_names
    """
    logging.info("\n" + "="*60)
    logging.info("📂 LOADING DATASETS")
    logging.info("="*60)
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Verify directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    logging.info(f"📁 Train directory: {train_dir}")
    logging.info(f"📁 Validation directory: {val_dir}")
    
    # Enhanced data transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(160, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.7),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((176, 176)),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    logging.info("\n⏳ Loading training dataset...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    logging.info(f"✅ Loaded {len(train_dataset):,} training images")
    
    logging.info("\n⏳ Loading validation dataset...")
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    logging.info(f"✅ Loaded {len(val_dataset):,} validation images")
    
    # Dataset statistics
    num_classes = len(train_dataset.classes)
    logging.info(f"\n📊 Dataset Summary:")
    logging.info(f" • Training samples: {len(train_dataset):,}")
    logging.info(f" • Validation samples: {len(val_dataset):,}")
    logging.info(f" • Number of classes: {num_classes}")
    logging.info(f" • Batch size: {batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    logging.info(f" • Training batches: {len(train_loader)}")
    logging.info(f" • Validation batches: {len(val_loader)}")
    logging.info("="*60 + "\n")
    
    return train_loader, val_loader, num_classes, train_dataset.classes


def try_compile(model_to_compile: nn.Module) -> nn.Module:
    """Try to compile model with torch.compile if available"""
    # Disabled torch.compile to avoid CUDA graph tensor overwrite issues
    logging.info("ℹ️ torch.compile disabled (to avoid CUDA graph issues)")
    return model_to_compile


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_loss, train_acc,
                   val_loss, val_acc, checkpoint_dir, epochs, is_best=False):
    """Save comprehensive checkpoint with all training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    if epoch % 5 == 0 or epoch == epochs - 1:
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:02d}.pth')
        torch.save(checkpoint, epoch_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_dir):
    """
    Load latest checkpoint if it exists
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        scaler: GradScaler to load state into
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        start_epoch: Epoch to start/resume training from
        best_val_acc: Best validation accuracy from checkpoint
    """
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    if not os.path.exists(latest_path):
        logging.info("ℹ️ No checkpoint found. Starting training from scratch.")
        return 0, 0.0
    
    logging.info("\n" + "="*60)
    logging.info("📂 LOADING CHECKPOINT")
    logging.info("="*60)
    logging.info(f"📁 Loading from: {latest_path}")
    
    try:
        checkpoint = torch.load(latest_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        train_loss = checkpoint.get('train_loss', 0.0)
        train_acc = checkpoint.get('train_acc', 0.0)
        val_loss = checkpoint.get('val_loss', 0.0)
        val_acc = checkpoint.get('val_acc', 0.0)
        
        # Try to load best validation accuracy from best_model.pth
        best_val_acc = val_acc
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_path):
            try:
                best_checkpoint = torch.load(best_path, map_location=device)
                best_val_acc = best_checkpoint.get('val_acc', val_acc)
            except Exception:
                pass
        
        logging.info(f"✅ Checkpoint loaded successfully!")
        logging.info(f" • Resuming from epoch: {start_epoch}")
        logging.info(f" • Last train loss: {train_loss:.4f}")
        logging.info(f" • Last train accuracy: {train_acc:.2f}%")
        if val_loss > 0:
            logging.info(f" • Last val loss: {val_loss:.4f}")
            logging.info(f" • Last val accuracy: {val_acc:.2f}%")
        logging.info(f" • Best val accuracy: {best_val_acc:.2f}%")
        logging.info(f" • Timestamp: {checkpoint.get('timestamp', 'N/A')}")
        logging.info("="*60 + "\n")
        
        return start_epoch, best_val_acc
        
    except Exception as e:
        logging.error(f"❌ Error loading checkpoint: {e}")
        logging.info("⚠️ Starting training from scratch.")
        return 0, 0.0


def save_for_inference(model, num_classes, class_names, checkpoint_dir):
    """
    Save model in a format optimized for CPU inference
    Saves model state dict, architecture config, and class names
    """
    logging.info("\n" + "="*60)
    logging.info("💾 SAVING MODEL FOR INFERENCE")
    logging.info("="*60)
    
    # Get the underlying model if it was compiled
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod
    else:
        model_to_save = model
    
    # Move model to CPU for inference
    model_to_save = model_to_save.cpu()
    model_to_save.eval()
    
    inference_dict = {
        'model_state_dict': model_to_save.state_dict(),
        'num_classes': num_classes,
        'class_names': class_names,
        'model_config': {
            'drop_path_rate': 0.2,
            'dropout': 0.2
        },
        'input_size': (160, 160),
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    inference_path = os.path.join(checkpoint_dir, 'resnet50_inference.pth')
    torch.save(inference_dict, inference_path)
    
    logging.info(f"✅ Model saved for CPU inference: {inference_path}")
    logging.info(f" • Number of classes: {num_classes}")
    logging.info(f" • Input size: 160x160")
    logging.info(f" • Device: CPU-compatible")
    logging.info("\n📝 To load for inference:")
    logging.info("  checkpoint = torch.load('resnet50_inference.pth', map_location='cpu')")
    logging.info("  model = ResNet50(num_classes=checkpoint['num_classes'])")
    logging.info("  model.load_state_dict(checkpoint['model_state_dict'])")
    logging.info("  model.eval()")
    logging.info("="*60 + "\n")


def train_epoch(model, loader, optimizer, criterion, device, epoch, scaler, writer):
    """Training loop for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    accumulation_steps = 2  # Effective batch size = 256 * 2 = 512
    
    with tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1:02d}", leave=True) as t:
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, labels) in t:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            last_loss = loss.item() * accumulation_steps
            running_loss += last_loss * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            acc_pct = 100.0 * correct / total if total > 0 else 0.0
            
            # Log batch metrics to TensorBoard
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Batch/train_loss', last_loss, global_step)
            writer.add_scalar('Batch/train_accuracy', acc_pct, global_step)
            
            t.set_postfix_str(f"Loss={last_loss:.4f} Acc={acc_pct:.2f}%")
    
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validation loop with mixed precision"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = test_loss / total if total > 0 else 0.0
    acc_pct = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, acc_pct


def main():
    """Main training function"""
    # Setup logging first
    log_filename = setup_logging()
    
    logging.info("="*60)
    logging.info("🚀 ResNet50 Training Script Started")
    logging.info("="*60)
    logging.info(f"📝 Log file: {log_filename}")
    logging.info(f"🖥️ Using device: {device}")
    
    # Hyperparameters
    batch_size = 256
    epochs = 50
    warmup_epochs = 5
    validate_every = max(1, epochs // 5)
    
    # Load datasets
    train_loader, val_loader, num_classes, class_names = load_datasets(
        data_dir='/data/',
        batch_size=batch_size
    )
    
    # Initialize model
    logging.info("🔨 Initializing ResNet50 model...")
    model = ResNet50(num_classes=num_classes, drop_path_rate=0.2, dropout=0.2).to(device)
    logging.info(f"✅ Model initialized with {num_classes} classes")
    
    # Try to compile model
    model = try_compile(model)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with separate weight decay for BN parameters
    bn_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            bn_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.SGD([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': bn_params, 'weight_decay': 0.0}
    ], lr=0.1, momentum=0.9, nesterov=True)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float((epoch + 1) / warmup_epochs)
        progress = float((epoch - warmup_epochs) / max(1, (epochs - warmup_epochs)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Setup directories
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load checkpoint if exists (resume training)
    start_epoch, best_val_acc = load_checkpoint(
        model, optimizer, scheduler, scaler, checkpoint_dir
    )
    
    # TensorBoard
    log_dir = os.path.join('runs', f'resnet50_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"📊 TensorBoard logs: {log_dir}")
    
    # Training loop
    logging.info("\n" + "="*60)
    logging.info("🚀 STARTING TRAINING")
    logging.info("="*60)
    logging.info(f" • Epochs: {epochs}")
    if start_epoch > 0:
        logging.info(f" • Resuming from epoch: {start_epoch + 1}")
    logging.info(f" • Batch size: {batch_size} (effective: {batch_size * 2})")
    logging.info(f" • Learning rate: 0.1 (with warmup and cosine decay)")
    logging.info(f" • Validation frequency: every {validate_every} epoch(s)")
    logging.info("="*60 + "\n")
    
    for epoch in range(start_epoch, epochs):
        logging.info(f"\n{'='*60}")
        logging.info(f"EPOCH {epoch + 1}/{epochs}")
        logging.info(f"{'='*60}")
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler, writer
        )
        
        # Validation
        did_validate = False
        val_loss = val_acc = 0.0
        is_best = False
        
        if (epoch % validate_every == 0) or (epoch == epochs - 1):
            did_validate = True
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                logging.info(f" 🏆 New best model saved! Val accuracy: {val_acc:.2f}%")
        
        # Always save checkpoint (latest and periodic)
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch,
            train_loss, train_acc, val_loss, val_acc,
            checkpoint_dir, epochs, is_best
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to TensorBoard
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/learning_rate', current_lr, epoch)
        
        if did_validate:
            writer.add_scalar('Epoch/val_loss', val_loss, epoch)
            writer.add_scalar('Epoch/val_accuracy', val_acc, epoch)
            writer.add_scalar('Epoch/best_val_accuracy', best_val_acc, epoch)
            writer.add_scalars('Loss/train_vs_val', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('Accuracy/train_vs_val', {'train': train_acc, 'val': val_acc}, epoch)
        
        # Print epoch summary
        logging.info(f"\n📊 Epoch {epoch + 1} Summary:")
        logging.info(f" • Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        if did_validate:
            logging.info(f" • Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            logging.info(f" • Best Val Acc: {best_val_acc:.2f}%")
        logging.info(f" • Learning Rate: {current_lr:.6f}")
    
    writer.close()
    
    # Save model for inference
    save_for_inference(model, num_classes, class_names, checkpoint_dir)
    
    # Final summary
    logging.info("\n" + "="*60)
    logging.info("🎉 TRAINING COMPLETED!")
    logging.info("="*60)
    logging.info(f" 🏆 Best validation accuracy: {best_val_acc:.2f}%")
    logging.info(f" 💾 Checkpoints saved in: {checkpoint_dir}/")
    logging.info(f" 📊 TensorBoard logs: {log_dir}/")
    logging.info(f" 📝 Training log: {log_filename}")
    logging.info(f" 🔍 View logs: tensorboard --logdir=runs")
    logging.info("="*60 + "\n")


if __name__ == '__main__':
    main()
