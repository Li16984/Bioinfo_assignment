import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV

# ===== 1. 加载数据 =====
# 加载元数据（制表符分隔）
metadata = pd.read_csv('../data/metadata.long.txt', sep='\t')
# 加载基因表达矩阵（制表符分隔，第一列为基因ID）
counts = pd.read_csv('../data/count.matrix.long.txt', sep='\t', index_col=0)
print(metadata.head())
print(counts.head())

# 按metadata顺序提取表达数据
X_list = counts
X = np.array(X_list).T  # 转置后 shape: (样本数, 基因数)
print(X)
print(f"表达矩阵形状: {X.shape}")
# 标签：NC=0，其余癌症=1
y = (metadata['label'] != 'NC').astype(int).values

# 标准化

X_log = np.log2(X + 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

pca = PCA(n_components=300)     
X_scaled = pca.fit_transform(X_scaled)

pca = PCA(n_components=2)                              # 保留2个主成分
X_pca = pca.fit_transform(X_scaled)       # 在发现集上拟合并变换
plt.figure(figsize=(10, 6))
sc1 = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y, cmap='coolwarm', alpha=0.6, edgecolors='k',
    marker='o', label='Discovery set'
)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA of qPCR Data (Discovery + Validation)')
plt.colorbar(sc1, label='Label (0=NC, 1=HCC)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('PCA1.png', dpi=150)
plt.show()

# 数据集划分（按元数据中的 dataset 列）
train_mask = metadata['dataset'] == 'train'
test_mask = metadata['dataset'] == 'test'
X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
print(f"基因数: {X_train.shape[1]}")

# ===== 4. 训练线性SVM（L1正则化实现特征选择） =====
# 使用L1惩罚，dual=False（适用于L1），solver='liblinear'
# C值通过网格搜索优化（也可固定为1）

param_grid = {'C': [0.01, 0.1, 1]}
svc = LinearSVC(penalty='l2', dual=False, random_state=42, max_iter=10000)
grid = GridSearchCV(svc, param_grid, cv=5, scoring='roc_auc', refit=True, verbose=4)
selector = RFECV(
    estimator=grid,
    step=1,
    cv=5,
    scoring='roc_auc',
    importance_getter=lambda clf:clf.best_estimator_.coef_
)
selector.fit(X_train, y_train)

best_model = selector.estimator_.best_estimator_
print(f"最佳C值: {selector.estimator_.best_params_['C']}")
print(f"交叉验证AUC: {selector.estimator_.best_score_:.4f}")

# ===== 5. 在测试集上评估 =====
X_test_selected = selector.transform(X_test)
y_pred_proba = best_model.decision_function(X_test_selected)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
test_auc = auc(fpr, tpr)
print(f"测试集AUC: {test_auc:.4f}")

# 输出选中的特征（非零系数的基因）
'''
coef = best_model.coef_.flatten()
selected_genes = counts.index[np.abs(coef) > 1e-6]
print(f"选中的基因数: {len(selected_genes)}")
if len(selected_genes) <= 20:
    print("选中的基因:", selected_genes.tolist())
else:
    print("前10个选中的基因:", selected_genes[:10].tolist())
'''
# ===== 6. 绘制ROC曲线 =====
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, 'b-', label=f'Test AUC = {test_auc:.4f}', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance', lw=1)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Cancer vs Normal)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../figure/svm_cancer_vs_normal_roc1.png', dpi=150)
plt.show()