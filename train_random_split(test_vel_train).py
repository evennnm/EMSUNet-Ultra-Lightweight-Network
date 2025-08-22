import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from lib.networks import EMSUNet
from lib.ImageLoader2D import PolypDataset
import gc
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
import time
import csv
from datetime import datetime
from thop import profile
from ptflops import get_model_complexity_info


PRIMARY_OUTPUT_INDEX = 3



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):

        if isinstance(pred, list):
            pred = pred[PRIMARY_OUTPUT_INDEX]

        pred = torch.sigmoid(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice



def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
        A.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22),
                 always_apply=True),
    ])



class AugmentedPolypDataset(Dataset):
    def __init__(self, base_dataset, augment=None):
        self.base_dataset = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]


        image_np = image.permute(1, 2, 0).numpy() * 255.0
        image_np = image_np.astype(np.uint8)
        mask_np = mask.squeeze(0).numpy() * 255
        mask_np = mask_np.astype(np.uint8)

        if self.augment:
            augmented = self.augment(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']


        image_tensor = transforms.ToTensor()(image_np)
        mask_tensor = torch.from_numpy((mask_np >= 127).astype(np.float32)).unsqueeze(0)

        return image_tensor, mask_tensor



def evaluate_loss_and_dice(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if isinstance(outputs, list):
                main_output = outputs[PRIMARY_OUTPUT_INDEX]
            else:
                main_output = outputs

            loss = criterion(main_output, masks)
            total_loss += loss.item() * images.size(0)


            preds = (torch.sigmoid(main_output) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)


    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()


    dice = f1_score(all_targets, all_preds)

    return avg_loss, dice



def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)


            if isinstance(outputs, list):
                outputs = outputs[PRIMARY_OUTPUT_INDEX]

            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)


    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()

    dice = f1_score(targets_flat, preds_flat)
    miou = jaccard_score(targets_flat, preds_flat)
    precision = precision_score(targets_flat, preds_flat)
    recall = recall_score(targets_flat, preds_flat)
    accuracy = accuracy_score(targets_flat, preds_flat)

    return dice, miou, precision, recall, accuracy


# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/', help='Training data path')

    parser.add_argument('--test_path_kvasir', type=str, default=r'Kvasir/',
                        help='Test data path1')
    parser.add_argument('--test_path_ClinicDB', type=str, default=r'CVC-ClinicDB/',
                        help='Test data path2')
    parser.add_argument('--test_path_ColonDB', type=str, default=r'CVC-ColonDB/',
                        help='Test data path3')
    parser.add_argument('--test_path_CVC300', type=str, default=r'CVC-300/',
                        help='Test data path4')
    parser.add_argument('--test_path_ETIS', type=str, default=r'ETIS-LaribPolypDB/',
                        help='Test data path5')

    parser.add_argument('--num_classes', type=int, default=1, help='Number of output channels (binary classification task)')
    parser.add_argument('--img_size', type=int, default=320, help='Network input size')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--max_epochs', type=int, default=350, help='Maximum number of training rounds')
    '''    !!!!!!!!Recommendation: Due to  the model structure and the simplicity of training strategy, 
    which leads to slower convergence,  it is suggested to train the model for at least 300 epochs, preferably 500 epochs, with a batch size of 6.
    Sometimes EMSUNet has a low probability of converging unstably, it is recommended to change the training strategy or train multiple times to solve this issue.!!!!!!!!!'''
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--deterministic', type=int, default=1, help='Whether to use deterministic training')
    parser.add_argument('--seed', type=int, default=58800, help='random seed')#20020319 0.26 58800 2742
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--patience', type=int, default=1000, help='Early stop patience value')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    seed_value = args.seed
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    cudnn.deterministic = args.deterministic == 1

    # Create dataset
    train_dataset = PolypDataset(
        folder_path=args.data_path,
        img_size=(args.img_size, args.img_size),
    )


    dataset_size = len(train_dataset)
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    # Ensure integer division, sum equals dataset_size
    train_size = int(train_ratio * dataset_size)
    valid_size = int(valid_ratio * dataset_size)
    test_size = dataset_size - train_size - valid_size

    # A random division into three subsets
    generator = torch.Generator().manual_seed(seed_value)
    train_base, valid_base, test_dataset = random_split(
        train_dataset,
        [train_size, valid_size, test_size],
        generator=generator
    )

    test_dataset_kvasir = PolypDataset(
        folder_path=args.test_path_kvasir,
        img_size=(args.img_size, args.img_size),
    )

    test_dataset_ClinicDB = PolypDataset(
        folder_path=args.test_path_ClinicDB,
        img_size=(args.img_size, args.img_size),
    )
    test_dataset_ColonDB = PolypDataset(
        folder_path=args.test_path_ColonDB,
        img_size=(args.img_size, args.img_size),
    )
    test_dataset_CVC300 = PolypDataset(
        folder_path=args.test_path_CVC300,
        img_size=(args.img_size, args.img_size),
    )
    test_dataset_ETIS = PolypDataset(
        folder_path=args.test_path_ETIS,
        img_size=(args.img_size, args.img_size),
    )

    # Apply data augmentation
    augmentations = get_augmentations()
    train_dataset_aug = AugmentedPolypDataset(train_base, augment=augmentations)
    valid_dataset = valid_base

    # Create data loader
    train_loader = DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    test_loader_kvasir= DataLoader(test_dataset_kvasir, batch_size=args.batch_size, num_workers=4)
    test_loader_ClinicDB= DataLoader(test_dataset_ClinicDB, batch_size=args.batch_size, num_workers=4)
    test_loader_ColonDB = DataLoader(test_dataset_ColonDB, batch_size=args.batch_size, num_workers=4)
    test_loader_CVC300= DataLoader(test_dataset_CVC300, batch_size=args.batch_size, num_workers=4)
    test_loader_ETIS= DataLoader(test_dataset_ETIS, batch_size=args.batch_size, num_workers=4)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EMSUNet(num_classes=args.num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate FLOPs
    flops = None
    if profile is not None:
        input_tensor = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        flops, _ = profile(model, inputs=(input_tensor,))

    # Print model information
    print("\n" + "=" * 50)
    print(f"Model Parameters: {total_params:,} (Total)")
    print(f"Trainable Parameters: {trainable_params:,}")
    if flops is not None:
        print(f"Model FLOPs: {flops:,} (for input size {args.img_size}x{args.img_size})")
    print("=" * 50 + "\n")


    # Optimizer and learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-6)

    # Learning rate decay strategy
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )


    criterion = DiceLoss()

    # Training Log
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = os.path.join(args.output_dir, f'progress_{current_time}.csv')
    model_path = os.path.join(args.output_dir, f'best_model_{current_time}.pth')
    results_path = os.path.join(args.output_dir, f'results_{current_time}.txt')

    # Create a CSV log file and write the table header
    with open(progress_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'test_loss', 'test_dice', 'lr'])

    best_val_dice = 0.0
    patience_counter = 0

    for epoch in range(args.max_epochs):
        # training phase
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)


            if isinstance(outputs, list):
                main_output = outputs[PRIMARY_OUTPUT_INDEX]
            else:
                main_output = outputs

            loss = criterion(main_output, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Verification phase
        val_loss, val_dice = evaluate_loss_and_dice(model, valid_loader, criterion, device)

        # test phase
        test_loss, test_dice = evaluate_loss_and_dice(model, test_loader, criterion, device)

        # Update learning rate
        lr_scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record to CSV
        with open(progress_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_dice, test_loss, test_dice, current_lr])

        # print log
        print(f'Epoch {epoch + 1}/{args.max_epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Dice: {val_dice:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Dice: {test_dice:.4f}, '
              f'LR: {current_lr:.2e}')

        # Save the best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            print(f"Saved best model with val_dice: {val_dice:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in val_dice for {patience_counter}/{args.patience} epochs")

            # 早停检查
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # final assessment
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate the training set, validation set, and test set.
    print("Evaluating training set...")
    train_dice, train_miou, train_precision, train_recall, train_accuracy = evaluate_model(
        model, DataLoader(train_base, batch_size=args.batch_size, num_workers=4), device)

    print("Evaluating validation set...")
    valid_dice, valid_miou, valid_precision, valid_recall, valid_accuracy = evaluate_model(
        model, valid_loader, device)

    print("Evaluating test set...")
    test_dice, test_miou, test_precision, test_recall, test_accuracy = evaluate_model(
        model, test_loader, device)

    print("Evaluating kvasir test set...")
    test_dice_kvasir, test_miou_kvasir, test_precision_kvasir, test_recall_kvasir, test_accuracy_kvasir = evaluate_model(
        model, test_loader_kvasir, device)

    print("Evaluating kvasir test set...")
    test_dice_ClinicDB, test_miou_ClinicDB, test_precision_ClinicDB, test_recall_ClinicDB, test_accuracy_ClinicDB = evaluate_model(
        model, test_loader_ClinicDB, device)

    print("Evaluating ColonDB test set...")
    test_dice_ColonDB, test_miou_ColonDB, test_precision_ColonDB, test_recall_ColonDB, test_accuracy_ColonDB = evaluate_model(
        model, test_loader_ColonDB, device)
    print("Evaluating CVC300 test set...")
    test_dice_CVC300, test_miou_CVC300, test_precision_CVC300, test_recall_CVC300, test_accuracy_CVC300= evaluate_model(
        model, test_loader_CVC300, device)
    print("Evaluating ETIS test set...")
    test_dice_ETIS, test_miou_ETIS, test_precision_ETIS, test_recall_ETIS, test_accuracy_ETIS = evaluate_model(
        model, test_loader_ETIS, device)

    # Save final results
    with open(results_path, 'w') as f:
        f.write(f'Training Data: {args.data_path}\n')
        f.write(f'Test Data: {args.test_path}\n')
        f.write(f'Model: {model_path}\n\n')


        f.write('== Training Set Metrics ==\n')
        f.write(f'Dice Coefficient: {train_dice:.4f}\n')
        f.write(f'mIoU: {train_miou:.4f}\n')
        f.write(f'Precision: {train_precision:.4f}\n')
        f.write(f'Recall: {train_recall:.4f}\n')
        f.write(f'Accuracy: {train_accuracy:.4f}\n\n')


        f.write('== Validation Set Metrics ==\n')
        f.write(f'Dice Coefficient: {valid_dice:.4f}\n')
        f.write(f'mIoU: {valid_miou:.4f}\n')
        f.write(f'Precision: {valid_precision:.4f}\n')
        f.write(f'Recall: {valid_recall:.4f}\n')
        f.write(f'Accuracy: {valid_accuracy:.4f}\n\n')

        f.write('== Test Set Metrics ==\n')
        f.write(f'Dice Coefficient: {test_dice:.4f}\n')
        f.write(f'mIoU: {test_miou:.4f}\n')
        f.write(f'Precision: {test_precision:.4f}\n')
        f.write(f'Recall: {test_recall:.4f}\n')
        f.write(f'Accuracy: {test_accuracy:.4f}\n')

        f.write('== Test _kvasir Set Metrics ==\n')
        f.write(f'Dice Coefficient: {test_dice_kvasir:.4f}\n')
        f.write(f'mIoU: {test_miou_kvasir:.4f}\n')
        f.write(f'Precision: {test_precision_kvasir:.4f}\n')
        f.write(f'Recall: {test_recall_kvasir:.4f}\n')
        f.write(f'Accuracy: {test_accuracy_kvasir:.4f}\n')

        f.write('== Test _ClinicDB Set Metrics ==\n')
        f.write(f'Dice Coefficient: {test_dice_ClinicDB:.4f}\n')
        f.write(f'mIoU: {test_miou_ClinicDB:.4f}\n')
        f.write(f'Precision: {test_precision_ClinicDB:.4f}\n')
        f.write(f'Recall: {test_recall_ClinicDB:.4f}\n')
        f.write(f'Accuracy: {test_accuracy_ClinicDB:.4f}\n')

        f.write('== _ColonDB Test Set Metrics ==\n')
        f.write(f'Dice Coefficient: {test_dice_ColonDB:.4f}\n')
        f.write(f'mIoU: {test_miou_ColonDB:.4f}\n')
        f.write(f'Precision: {test_precision_ColonDB:.4f}\n')
        f.write(f'Recall: {test_recall_ColonDB:.4f}\n')
        f.write(f'Accuracy: {test_accuracy_ColonDB:.4f}\n')

        f.write('==_CVC300 Test Set Metrics ==\n')
        f.write(f'Dice Coefficient: {test_dice_CVC300:.4f}\n')
        f.write(f'mIoU: {test_miou_CVC300:.4f}\n')
        f.write(f'Precision: {test_precision_CVC300:.4f}\n')
        f.write(f'Recall: {test_recall_CVC300:.4f}\n')
        f.write(f'Accuracy: {test_accuracy_CVC300:.4f}\n')

        f.write('==_ETIS Test Set Metrics ==\n')
        f.write(f'Dice Coefficient: {test_dice_ETIS:.4f}\n')
        f.write(f'mIoU: {test_miou_ETIS:.4f}\n')
        f.write(f'Precision: {test_precision_ETIS:.4f}\n')
        f.write(f'Recall: {test_recall_ETIS:.4f}\n')
        f.write(f'Accuracy: {test_accuracy_ETIS:.4f}\n')


    print(f"Evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()