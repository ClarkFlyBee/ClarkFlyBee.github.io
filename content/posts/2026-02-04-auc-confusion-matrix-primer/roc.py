# generate_fraud_roc.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.family'] = 'SimHei' # Windows 示例
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 构造数据
np.random.seed(42)

# 100个垃圾邮件：90个高概率，10个低概率
fraud_scores = np.concatenate([
    np.random.uniform(0.8, 0.95, 90),
    np.random.uniform(0.1, 0.3, 10)
])

# 900个正常邮件：50个高概率（误判），850个低概率
normal_scores = np.concatenate([
    np.random.uniform(0.6, 0.75, 50),  # 难以区分的正常邮件
    np.random.uniform(0.05, 0.2, 850)
])

y_true = np.array([1]*100 + [0]*900)
y_score = np.concatenate([fraud_scores, normal_scores])

# 计算 ROC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# 绘图
fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

# ROC 曲线
ax.plot(fpr, tpr, color='#e63946', linewidth=3, 
        label=f'垃圾模型检测模型 (AUC = {roc_auc:.3f})')

# 对角线（随机猜测）
ax.plot([0, 1], [0, 1], color='#adb5bd', linewidth=2, 
        linestyle='--', label='随机猜测 (AUC = 0.500)')

# 样式
ax.set_xlim([-0.02, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel('FPR = FP / (FP + TN)', fontsize=12)
ax.set_ylabel('TPR = TP / (TP + FN)', fontsize=12)
ax.set_title('ROC', 
             fontsize=14, fontweight='bold', pad=20)

ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('roc_fraud_detection.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ AUC = {roc_auc:.3f}")
print("✅ ROC图已保存: roc_fraud_detection.png")
plt.show()