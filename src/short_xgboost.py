import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
import xgboost as xgb
from sklearn.decomposition import PCA

# ===== 1. 加载数据 =====
# 加载元数据（制表符分隔）
metadata = pd.read_csv('../data/metadata.short.txt', sep='\t')
# 加载基因表达矩阵（制表符分隔，第一列为基因ID）
counts = pd.read_csv('../processed_data/short.counts.txt', sep='\t', index_col=0)
print(metadata.head())
print(counts.head())

# 按metadata顺序提取表达数据
X_scaled = counts
print(f"表达矩阵形状: {X_scaled.shape}")
# 标签：NC=0，其余癌症=1
y = (metadata['label'] != 'HD').astype(int).values

train_mask = (metadata['dataset'] == 'train').values 
test_mask = (metadata['dataset'] == 'test').values
X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print(f"{X_train.shape[0]} samples in discovery set")
print(f"{X_test.shape[0]} samples in validation set")

# ===== XGBoost 分类器（也可替换为随机森林） =====
# 如需使用随机森林，注释掉下面这行并取消注释后面的代码
model = xgb.XGBClassifier(random_state=666, use_label_encoder=False, eval_metric='logloss')

# 参数网格（可根据需要调整）
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# 网格搜索（5折交叉验证，以AUROC为评分）
grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=1,           # Windows下避免多进程错误
    verbose=1
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("最佳参数:", grid.best_params_)
print("交叉验证最佳AUROC: {:.4f}".format(grid.best_score_))

# ===== 验证集评估 =====
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_pred_proba)
print("验证集AUROC: {:.4f}".format(auroc))

# ===== 绘制ROC曲线 =====
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost on short Data')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figure/xgboost_roc_short.png', dpi=150)
plt.show()


# ===== 可选：特征重要性 =====
plt.figure(figsize=(8, 4))
importance = best_model.feature_importances_
feat_names = counts.T.index
sorted_idx = np.argsort(importance)[-10:][::-1]
plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feat_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance (XGBoost)')
plt.title('Top Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../figure/xgboost_feature_importance_short.png', dpi=150)
plt.show()
