import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 尝试导入批次校正库（ComBat）
try:
    from neuroCombat import neuroCombat
    COMBAT_AVAILABLE = True
    print("neurocombat 导入成功，将执行批次校正。")
except ImportError:
    COMBAT_AVAILABLE = False
    print("未安装 neurocombat (pip install neurocombat)，跳过批次校正步骤。")

# ===== 0. 加载数据 =====
metadata = pd.read_csv('../data/metadata.short.txt', sep='\t')
# 读取表达矩阵（第一列为基因ID，列为样本）
counts = pd.read_csv('../data/count.matrix.short.txt', sep='\t', index_col=0)
metadata['RNA Isolation batch'] = metadata['RNA Isolation batch'].replace(9, 8)
print(f"原始矩阵形状: {counts.shape}（{counts.shape[0]} 个基因，{counts.shape[1]} 个样本）")

# ===== 1. 过滤低表达基因 =====
# 标准：在 ≥10% 的样本中 CPM > 1 的基因保留
N = counts.shape[1]
# 计算 CPM（Counts Per Million）
cpm = (counts / counts.sum(axis=0)) * 1e6
# 保留那些在至少 10% 样本中 CPM > 1 的基因
keep_genes = (cpm > 1).sum(axis=1) > (N * 0.1)
counts_filtered = counts.loc[keep_genes]
print(f"过滤后保留基因数: {counts_filtered.shape[0]}（去除了 {counts.shape[0] - counts_filtered.shape[0]} 个低表达基因）")

# ===== 2. 归一化（Normalization）=====
# 使用 log2(CPM + 1) 进行归一化，消除测序深度差异并稳定方差
cpm_norm = (counts_filtered / counts_filtered.sum(axis=0)) * 1e6
X_log = np.log2(cpm_norm + 1)  # 此时形状为 (基因, 样本)
# 转置为 (样本, 基因) 以便后续 ML 处理
X_norm = X_log.T  # 形状 (样本数, 基因数)

# (注：更精细的归一化可使用 DESeq2 的 median-of-ratios 或 TMM，此处采用最通用的 logCPM)

# ===== 3. 去除批次效应（Batch Effect Removal）=====
# 使用 ComBat 方法，批次列选择 metadata 中的 'source'
if COMBAT_AVAILABLE:
    # 构造协变量数据框（包含批次和需要保留的生物学变量）
    covars_df = metadata[['RNA Isolation batch', 'label']].copy()
    # 将 label 转换为数值（ComBat 需要数值型协变量）
    covars_df['label'] = covars_df['label'].astype('category').cat.codes

    # 运行 ComBat
    data_combat = neuroCombat(
        dat=X_log.values,        # 表达矩阵 (基因, 样本)
        covars=covars_df,        # 协变量数据框 (样本, 协变量)
        batch_col='RNA Isolation batch',      # 批次信息所在的列名
        categorical_cols=['label'],  # 需要保留的分类协变量（此处 label 是分类变量）
        continuous_cols=None,    # 连续协变量（无）
        eb=True,                 # 使用经验贝叶斯（默认）
        parametric=True,         # 使用参数调整（默认）
        mean_only=False          # 是否仅校正均值（False 则同时校正均值和方差）
    )

    X_corrected = data_combat['data'].T  # 转置为 (样本, 基因)
    print("批次校正完成。")
else:
    X_corrected = X_norm.copy()   # 未安装时直接使用归一化数据
    print("跳过批次校正，使用归一化数据。")

# ===== 4. 数据可视化（评估预处理效果） =====
def plot_pca(X_data, title_prefix):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)
    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_plot['label'] = metadata['label'].values
    df_plot['batch'] = metadata['RNA Isolation batch'].values
    
    plt.figure(figsize=(14, 5))
    # 按 label（癌症/正常）着色
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='label', alpha=0.7, palette='Set2')
    plt.title(f'{title_prefix} - Label_colored')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    # 按 source（批次来源）着色
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='batch', alpha=0.7, palette='tab10')
    plt.title(f'{title_prefix} - Batch_colored')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    plt.tight_layout()
    plt.savefig(f'../figure/short_{title_prefix}_pca.png', dpi=150)
    plt.show()

# 绘制批次校正前
print("绘制批次校正前的 PCA...")
plot_pca(X_norm, "Before_BatchCorrection")

# 绘制批次校正后（如果执行了）
if COMBAT_AVAILABLE:
    print("绘制批次校正后的 PCA...")
    plot_pca(X_corrected, "After_BatchCorrection")

# ===== 5. 特征 Scaling（标准化） =====
scaler = StandardScaler()
X_final = scaler.fit_transform(X_corrected)  # 最终用于 ML 的数据 (样本, 基因)

print(f"最终预处理后的数据形状: {X_final.shape}")
print("预处理完成，X_final 可直接用于 XGBoost 或 SVM 等模型。")

genes = counts_filtered.index.tolist()   # 基因名
samples = metadata['sample id'].tolist() # 样本名

# 构建 DataFrame（注意：基因作为行，样本作为列，跟你的原始矩阵一致）
df = pd.DataFrame(X_final, index=samples, columns=genes)

# 保存为 TSV（sep='\t' 指定制表符）
df.to_csv('../processed_data/short.counts.txt', sep='\t', header=True, index=True)


from sklearn.manifold import TSNE
import umap
def tsne(X_data, title_prefix):
    tsne = TSNE(n_components=2, random_state=42, perplexity=80)  # 小样本设perplexity较小
    X_tsne = tsne.fit_transform(X_data)
    df_plot = pd.DataFrame(X_tsne, columns=['tsne1', 'tsne2'])
    df_plot['label'] = metadata['label'].values
    df_plot['batch'] = metadata['RNA Isolation batch'].values
    
    plt.figure(figsize=(14, 5))
    # 按 label（癌症/正常）着色
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_plot, x='tsne1', y='tsne2', hue='label', alpha=0.7, palette='Set2')
    plt.title(f'{title_prefix} - Label_colored')
    plt.xlabel(f't_sne1')
    plt.ylabel(f't_sne2')
    
    # 按 source（批次来源）着色
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_plot, x='tsne1', y='tsne2', hue='batch', alpha=0.7, palette='tab10')
    plt.title(f'{title_prefix} - Batch_colored')
    plt.xlabel(f't_sne1')
    plt.ylabel(f't_sne2')
    
    plt.tight_layout()
    plt.savefig(f'../figure/short_{title_prefix}_tsne.png', dpi=150)
    plt.show()

def u_map(X_data, title_prefix):
    umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=80)  # 小样本调低n_neighbors
    X_umap = umap_model.fit_transform(X_data)
    df_plot = pd.DataFrame(X_umap, columns=['umap1', 'umap2'])
    df_plot['label'] = metadata['label'].values
    df_plot['batch'] = metadata['RNA Isolation batch'].values
    
    plt.figure(figsize=(14, 5))
    # 按 label（癌症/正常）着色
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_plot, x='umap1', y='umap2', hue='label', alpha=0.7, palette='Set2')
    plt.title(f'{title_prefix} - Label_colored')
    plt.xlabel(f'UMAP1')
    plt.ylabel(f'UMAP2')
    
    # 按 source（批次来源）着色
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_plot, x='umap1', y='umap2', hue='batch', alpha=0.7, palette='tab10')
    plt.title(f'{title_prefix} - Batch_colored')
    plt.xlabel(f'UMAP1')
    plt.ylabel(f'UMAP2')
    
    plt.tight_layout()
    plt.savefig(f'../figure/short_{title_prefix}_umap.png', dpi=150)
    plt.show()

tsne(X_corrected, "After_BatchCorrection")
u_map(X_corrected, "After_BatchCorrection")