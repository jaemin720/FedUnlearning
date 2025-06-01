import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json


def get_confidences(model, dataset, idxs, device):
    model.eval()
    loader = DataLoader(Subset(dataset, idxs), batch_size=128, shuffle=False)
    confidences = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            confidences.extend(max_conf.cpu().numpy())

    return confidences


def evaluate_mia(model, dataset, forget_idxs, retain_idxs, shadow_idxs, device, save_path=None):
    # 1. shadow 모델 훈련용 confidence 수집
    conf_retain = get_confidences(model, dataset, shadow_idxs, device)
    conf_forget = get_confidences(model, dataset, forget_idxs, device)

    X_shadow = np.array(conf_retain + conf_forget).reshape(-1, 1)
    y_shadow = np.array([0] * len(conf_retain) + [1] * len(conf_forget))

    # 2. 공격 모델 학습
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_shadow, y_shadow)

    # 3. 평가 대상 confidence 수집
    eval_conf_retain = get_confidences(model, dataset, retain_idxs, device)
    eval_conf_forget = get_confidences(model, dataset, forget_idxs, device)

    X_eval = np.array(eval_conf_retain + eval_conf_forget).reshape(-1, 1)
    y_eval = np.array([0] * len(eval_conf_retain) + [1] * len(eval_conf_forget))

    # 4. AUC 계산
    pred_probs = clf.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, pred_probs)

    result = {
        'auc': float(auc),
        'n_forget': len(forget_idxs),
        'n_retain': len(retain_idxs),
        'n_shadow': len(shadow_idxs)
    }

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)

    return result
