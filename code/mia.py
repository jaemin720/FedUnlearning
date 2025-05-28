import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import json
from tqdm import tqdm

def get_confidences(model, dataset, indices, device):
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False)
    confidences = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            top_conf, _ = torch.max(probs, dim=1)
            confidences.extend(top_conf.cpu().numpy())
    return confidences

def evaluate_mia(model, dataset, target_user_idx, retain_idxs, forget_idxs, shadow_idxs, device, save_path, args, k=10):
    print("[MIA] Starting MIA Evaluation...")

    n_shadow = k
    shadow_size = len(shadow_idxs) // n_shadow

    all_shadow_results = []
    print(f"[LiRA] Evaluating with {n_shadow} shadow models...")

    for i in tqdm(range(n_shadow)):
        shadow_subset = shadow_idxs[i * shadow_size: (i + 1) * shadow_size]
        shadow_subset = [idx for idx in shadow_subset if idx < len(dataset)]

        forget_set = forget_idxs[:len(shadow_subset)]
        retain_set = retain_idxs[:len(shadow_subset)]

        conf_forget = get_confidences(model, dataset, forget_set, device)
        conf_retain = get_confidences(model, dataset, retain_set, device)

        X_shadow = np.array(conf_forget + conf_retain).reshape(-1, 1)
        y_shadow = np.array([1] * len(conf_forget) + [0] * len(conf_retain))

        clf = LogisticRegression(solver='liblinear').fit(X_shadow, y_shadow)

        eval_conf_forget = get_confidences(model, dataset, forget_set, device)
        eval_conf_retain = get_confidences(model, dataset, retain_set, device)

        X_eval = np.array(eval_conf_forget + eval_conf_retain).reshape(-1, 1)
        y_eval = np.array([1] * len(eval_conf_forget) + [0] * len(eval_conf_retain))

        y_pred = clf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1])

        all_shadow_results.append({"acc": acc, "auc": auc})

    avg_acc = np.mean([r['acc'] for r in all_shadow_results])
    avg_auc = np.mean([r['auc'] for r in all_shadow_results])

    print(f"[LiRA] Avg Accuracy: {avg_acc:.4f}, Avg AUC: {avg_auc:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_shadow_results, f, indent=2)
