import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import extract_keypoints_from_heatmaps, compute_pck, plot_pck_curves, visualize_predictions
from baseline import ablation_study, analyze_failure_cases

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

def _collect_outputs(model, loader):
    """Return stacked outputs, targets, and images (on CPU)."""
    model.eval()
    outputs, targets, images = [], [], []
    with torch.no_grad():
        for image, target in loader:
            output = model(image.to(DEVICE)).cpu()
            outputs.append(output)
            targets.append(target.cpu())
            images.append(image.cpu())
    return torch.cat(outputs), torch.cat(targets), torch.cat(images)

def train_heatmap_model(model, train_loader, val_loader, num_epochs=30):
    """
    Train the heatmap-based model.

    Uses MSE loss between predicted and target heatmaps.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    # Log losses and save best model
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    heatmap_size = getattr(getattr(val_loader, "dataset", None), "heatmap_size", 64)
    images, _ = next(iter(val_loader))
    images = images.to(DEVICE)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[Heatmap] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/heatmap_model.pth")

        if (epoch + 1) in [1, 10, 20]:
            model.eval()
            with torch.no_grad():
                preds = model(images)

            for i in range(3):
                for k in range(preds.size(1)):
                    plt.imshow(preds[i, k].cpu().numpy(), cmap="hot")
                    plt.title(f"Epoch {epoch} Sample {i} Keypoint {k}")
                    plt.colorbar()
                    plt.savefig(os.path.join("results/visualizations", f"heatmap_epoch{epoch}_sample{i}_kp{k}.png"))
                    plt.close()

    return history
    pass


def train_regression_model(model, train_loader, val_loader, num_epochs=30):
    """
    Train the direct regression model.

    Uses MSE loss between predicted and target coordinates.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    # Log losses and save best model
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"[Regression] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/regression_model.pth")

    return history
    pass


def main():
    # Train both models with same data
    # Save training logs for comparison
    os.makedirs("results/visualizations", exist_ok=True)

    train_img_dir = 'datasets/keypoints/train'
    train_ann_file = 'datasets/keypoints/train_annotations.json'
    val_img_dir = 'datasets/keypoints/val'
    val_ann_file = 'datasets/keypoints/val_annotations.json'
    train_dataset_hm = KeypointDataset(train_img_dir, train_ann_file, output_type="heatmap", heatmap_size=64)
    val_dataset_hm   = KeypointDataset(val_img_dir,   val_ann_file,   output_type="heatmap", heatmap_size=64)
    train_dataset_reg = KeypointDataset(train_img_dir, train_ann_file, output_type="regression")
    val_dataset_reg   = KeypointDataset(val_img_dir,   val_ann_file,   output_type="regression")
    train_loader_hm = DataLoader(train_dataset_hm, batch_size=32, shuffle=True)
    val_loader_hm   = DataLoader(val_dataset_hm,   batch_size=32, shuffle=False)
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
    val_loader_reg   = DataLoader(val_dataset_reg,   batch_size=32, shuffle=False)
    heatmap_model = HeatmapNet(num_keypoints=5, heatmap_size=64).to(DEVICE)
    heatmap_history = train_heatmap_model(heatmap_model, train_loader_hm, val_loader_hm, num_epochs=30)
    regression_model = RegressionNet(num_keypoints=5).to(DEVICE)
    regression_history = train_regression_model(regression_model, train_loader_reg, val_loader_reg, num_epochs=30)
    logs = {"heatmap": heatmap_history, "regression": regression_history}
    with open("results/training_log.json", "w") as f:
        json.dump(logs, f, indent=4)

    heatmap_model.load_state_dict(torch.load("results/heatmap_model.pth", map_location=DEVICE))
    regression_model.load_state_dict(torch.load("results/regression_model.pth", map_location=DEVICE))

    hm_o, hm_t, hm_i = _collect_outputs(heatmap_model, val_loader_hm)
    reg_o, reg_t, reg_i = _collect_outputs(regression_model, val_loader_reg)
    heatmap_size = getattr(val_dataset_hm, "heatmap_size", 64)
    scale = 128.0 / float(heatmap_size)
    hm_pred_pts = extract_keypoints_from_heatmaps(hm_o).float() * scale
    hm_gt_pts   = extract_keypoints_from_heatmaps(hm_t).float() * scale
    reg_pred_pts = (reg_o.view(-1, 5, 2) * 128.0).float()
    reg_gt_pts   = (reg_t.view(-1, 5, 2) * 128.0).float()
    thresholds = [0.05, 0.1, 0.15, 0.2]
    pck_heatmap = compute_pck(hm_pred_pts, hm_gt_pts, thresholds, normalize_by="bbox")
    pck_regression = compute_pck(reg_pred_pts, reg_gt_pts, thresholds, normalize_by="bbox")
    plot_pck_curves(pck_heatmap, pck_regression, save_path="results/visualizations/pck_curve.png")
    for i in range(5):
        visualize_predictions(hm_i[i].squeeze().numpy(), reg_pred_pts[i].numpy(), reg_gt_pts[i].numpy(), f"results/visualizations/heatmap_sample_{i}.png")
        visualize_predictions(reg_i[i].squeeze().numpy(), reg_pred_pts[i].numpy(), reg_gt_pts[i].numpy(), f"results/visualizations/regression_sample_{i}.png")
    
    ablation_study(train_dataset_hm)
    analyze_failure_cases(heatmap_model, regression_model, val_loader_hm, val_loader_reg)

    pass

if __name__ == "__main__":
    main()
