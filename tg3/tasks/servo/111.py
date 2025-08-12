import matplotlib.pyplot as plt

mask_ratio = [25, 35, 45, 55, 65, 75, 85]
top1_acc = [75.13, 75.24, 75.45, 76.02, 76.32, 76.73, 76.45]

plt.figure(figsize=(10, 5))
plt.plot(mask_ratio, top1_acc, '-o', color='blue', linewidth=2, markersize=8)

# 标注数据点
for x, y in zip(mask_ratio, top1_acc):
    if y == max(top1_acc):  # 最大点向下偏移
        plt.text(x, y-0.08, f"{y:.2f}", ha='center', va='top', fontsize=13)
    else:
        plt.text(x, y+0.05, f"{y:.2f}", ha='center', va='bottom', fontsize=13)

plt.xlabel("Mask Ratio (%)", fontsize=15)
plt.ylabel("Top-1 Acc (%)", fontsize=15)
plt.xticks(mask_ratio, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# 保存高清PDF/PNG/SVG
plt.savefig("mask_ratio.pdf", bbox_inches='tight')
plt.savefig("mask_ratio.png", dpi=300, bbox_inches='tight')
plt.savefig("mask_ratio.svg", bbox_inches='tight')
plt.show()
