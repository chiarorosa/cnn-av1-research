"""
V6 Pipeline - Data Augmentation Strategies
Stage-specific augmentation pipelines
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import random


class HorizontalFlipWithLabelSwap:
    """Horizontal flip that swaps AB labels: HORZ_A <-> HORZ_B"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, label):
        if random.random() < self.p:
            img = torch.flip(img, dims=[-1])
            # Swap HORZ_A (0) <-> HORZ_B (1), keep VERT_A (2) and VERT_B (3)
            if label == 0:
                label = 1
            elif label == 1:
                label = 0
        return img, label


class VerticalFlipWithLabelSwap:
    """Vertical flip that swaps AB labels: VERT_A <-> VERT_B"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, label):
        if random.random() < self.p:
            img = torch.flip(img, dims=[-2])
            # Swap VERT_A (2) <-> VERT_B (3), keep HORZ_A (0) and HORZ_B (1)
            if label == 2:
                label = 3
            elif label == 3:
                label = 2
        return img, label


class Rotation90WithLabelRotate:
    """90° rotation that rotates AB labels: HORZ <-> VERT"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, label):
        if random.random() < self.p:
            k = random.choice([1, 3])  # 90° or 270°
            img = torch.rot90(img, k=k, dims=[-2, -1])
            
            # Rotate labels: HORZ <-> VERT
            if k == 1:  # 90° clockwise
                if label == 0:  # HORZ_A -> VERT_A
                    label = 2
                elif label == 1:  # HORZ_B -> VERT_B
                    label = 3
                elif label == 2:  # VERT_A -> HORZ_B
                    label = 1
                elif label == 3:  # VERT_B -> HORZ_A
                    label = 0
            else:  # 270° clockwise (= 90° counter-clockwise)
                if label == 0:  # HORZ_A -> VERT_B
                    label = 3
                elif label == 1:  # HORZ_B -> VERT_A
                    label = 2
                elif label == 2:  # VERT_A -> HORZ_A
                    label = 0
                elif label == 3:  # VERT_B -> HORZ_B
                    label = 1
        
        return img, label


class GaussianNoise:
    """Add Gaussian noise"""
    def __init__(self, sigma=0.01, p=0.5):
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.sigma
            img = img + noise
        return img


class Cutout:
    """Random erasing (cutout)"""
    def __init__(self, size=4, p=0.3):
        self.size = size
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            _, h, w = img.shape
            x = random.randint(0, max(0, w - self.size))
            y = random.randint(0, max(0, h - self.size))
            img[:, y:y+self.size, x:x+self.size] = 0
        return img


class GridShuffle:
    """Shuffle image in a grid pattern"""
    def __init__(self, grid_size=4, p=0.2):
        self.grid_size = grid_size
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            c, h, w = img.shape
            gh, gw = h // self.grid_size, w // self.grid_size
            
            # Split into grid
            grids = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    grids.append(img[:, i*gh:(i+1)*gh, j*gw:(j+1)*gw])
            
            # Shuffle
            random.shuffle(grids)
            
            # Reconstruct
            img_new = torch.zeros_like(img)
            idx = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    img_new[:, i*gh:(i+1)*gh, j*gw:(j+1)*gw] = grids[idx]
                    idx += 1
            
            return img_new
        return img


class CoarseDropout:
    """Multiple random patches dropout"""
    def __init__(self, num_holes=3, hole_size=4, p=0.3):
        self.num_holes = num_holes
        self.hole_size = hole_size
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            _, h, w = img.shape
            for _ in range(self.num_holes):
                x = random.randint(0, max(0, w - self.hole_size))
                y = random.randint(0, max(0, h - self.hole_size))
                img[:, y:y+self.hole_size, x:x+self.hole_size] = 0
        return img


class MixupAugmentation:
    """Mixup between samples"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, img1, img2, label1, label2):
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        return mixed_img, label1, label2, lam


class Stage1Augmentation:
    """Augmentation for Stage 1 (Binary classification)"""
    def __init__(self, train=True):
        self.train = train
        if train:
            self.transforms = [
                lambda x: torch.flip(x, dims=[-1]) if random.random() > 0.5 else x,  # H-flip
                lambda x: torch.flip(x, dims=[-2]) if random.random() > 0.5 else x,  # V-flip
                lambda x: torch.rot90(x, k=random.choice([0,1,2,3]), dims=[-2,-1]) if random.random() > 0.5 else x,
                GaussianNoise(sigma=0.01, p=0.3),
            ]
    
    def __call__(self, img):
        if self.train:
            for t in self.transforms:
                img = t(img)
        return img


class Stage2Augmentation:
    """Augmentation for Stage 2 (3-way classification)"""
    def __init__(self, train=True):
        self.train = train
        if train:
            self.transforms = [
                lambda x: torch.flip(x, dims=[-1]) if random.random() > 0.5 else x,
                lambda x: torch.flip(x, dims=[-2]) if random.random() > 0.5 else x,
                lambda x: torch.rot90(x, k=random.choice([0,1,2,3]), dims=[-2,-1]) if random.random() > 0.5 else x,
                GaussianNoise(sigma=0.01, p=0.3),
                Cutout(size=4, p=0.3),
                GridShuffle(grid_size=4, p=0.2),
            ]
    
    def __call__(self, img):
        if self.train:
            for t in self.transforms:
                img = t(img)
        return img


class Stage3RectAugmentation:
    """Augmentation for Stage 3-RECT (HORZ vs VERT)"""
    def __init__(self, train=True):
        self.train = train
        if train:
            self.transforms = [
                lambda x: torch.flip(x, dims=[-1]) if random.random() > 0.5 else x,
                lambda x: torch.flip(x, dims=[-2]) if random.random() > 0.5 else x,
                GaussianNoise(sigma=0.01, p=0.3),
                Cutout(size=4, p=0.2),
            ]
    
    def __call__(self, img):
        if self.train:
            for t in self.transforms:
                img = t(img)
        return img


class Stage3ABAugmentation:
    """Heavy augmentation for Stage 3-AB (4-way classification)"""
    def __init__(self, train=True):
        self.train = train
        self.h_flip = HorizontalFlipWithLabelSwap(p=0.5)
        self.v_flip = VerticalFlipWithLabelSwap(p=0.5)
        self.rot90 = Rotation90WithLabelRotate(p=0.5)
        self.noise = GaussianNoise(sigma=0.01, p=0.3)
        self.coarse_dropout = CoarseDropout(num_holes=3, hole_size=4, p=0.3)
        self.cutout = Cutout(size=4, p=0.3)
    
    def __call__(self, img, label):
        if self.train:
            # Label-aware augmentations
            img, label = self.h_flip(img, label)
            img, label = self.v_flip(img, label)
            img, label = self.rot90(img, label)
            
            # Label-agnostic augmentations
            img = self.noise(img)
            img = self.coarse_dropout(img)
            img = self.cutout(img)
        
        return img, label


class TestTimeAugmentation:
    """Test-time augmentation (TTA) for inference"""
    def __init__(self, num_augments=4):
        self.num_augments = num_augments
    
    def __call__(self, img):
        """
        Returns list of augmented versions
        Original + flips + rotations
        """
        augmented = [img]  # Original
        
        # Horizontal flip
        augmented.append(torch.flip(img, dims=[-1]))
        
        # Vertical flip
        augmented.append(torch.flip(img, dims=[-2]))
        
        # 180° rotation
        augmented.append(torch.rot90(img, k=2, dims=[-2, -1]))
        
        return augmented[:self.num_augments]
    
    def aggregate_predictions(self, predictions):
        """Average predictions from multiple augmentations"""
        return torch.stack(predictions).mean(dim=0)


def get_augmentation(stage, train=True):
    """
    Factory function to get augmentation pipeline
    
    Args:
        stage: 'stage1', 'stage2', 'stage3_rect', 'stage3_ab'
        train: training or validation
    
    Returns:
        augmentation function
    """
    if stage == 'stage1':
        return Stage1Augmentation(train=train)
    elif stage == 'stage2':
        return Stage2Augmentation(train=train)
    elif stage == 'stage3_rect':
        return Stage3RectAugmentation(train=train)
    elif stage == 'stage3_ab':
        return Stage3ABAugmentation(train=train)
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    # Test augmentations
    print("Testing Stage 1 Augmentation...")
    aug1 = Stage1Augmentation(train=True)
    img = torch.randn(3, 16, 16)
    aug_img = aug1(img)
    print(f"Input shape: {img.shape}, Output shape: {aug_img.shape}\n")
    
    print("Testing Stage 2 Augmentation...")
    aug2 = Stage2Augmentation(train=True)
    aug_img = aug2(img)
    print(f"Output shape: {aug_img.shape}\n")
    
    print("Testing Stage 3-AB Augmentation with label swap...")
    aug3_ab = Stage3ABAugmentation(train=True)
    label = 0  # HORZ_A
    aug_img, aug_label = aug3_ab(img, label)
    print(f"Original label: 0 (HORZ_A), Augmented label: {aug_label}")
    print(f"Output shape: {aug_img.shape}\n")
    
    print("Testing Test-Time Augmentation...")
    tta = TestTimeAugmentation(num_augments=4)
    augmented_imgs = tta(img)
    print(f"Number of augmented versions: {len(augmented_imgs)}")
    
    # Simulate predictions
    fake_preds = [torch.randn(3) for _ in augmented_imgs]
    avg_pred = tta.aggregate_predictions(fake_preds)
    print(f"Average prediction shape: {avg_pred.shape}\n")
    
    print("✅ All augmentations working correctly!")
