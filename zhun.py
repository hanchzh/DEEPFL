import numpy as np
import matplotlib.pyplot as plt

#matplotlib inline

plt.style.use("ggplot")
shops = ["10", "30", "50", "70", "90"]
sales_product_1 = [0.9, 0.87, 0.86, 0.82, 0.80]
sales_product_2 = [0.92, 0.86, 0.80, 0.75, 0.70]
sales_product_3 = [0.35, 0.25, 0.22, 0.20, 0.18]

# 创建分组柱状图，需要自己控制x轴坐标
xticks = np.arange(len(shops))

fig, ax = plt.subplots(figsize=(10, 7))
# 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
ax.bar(xticks, sales_product_1, width=0.25, label="common layer detection (Ours)", color="red")
# 所有门店第二种产品的销量，通过微调x轴坐标来调整新增柱子的位置
ax.bar(xticks + 0.25, sales_product_2, width=0.25, label="The full-parameter anomaly detection scheme", color="blue")
# 所有门店第三种产品的销量，继续微调x轴坐标调整新增柱子的位置
ax.bar(xticks + 0.5, sales_product_3, width=0.25, label="No anomaly detection scheme", color="green")

#ax.set_title("User Computing Cost Comparison", fontsize=15)
ax.set_xlabel("Percentage of number i (λ)")
ax.set_ylabel("Model Accuracy")
ax.legend()

# 最后调整x轴标签的位置
ax.set_xticks(xticks + 0.25)
ax.set_xticklabels(shops)
plt.show()
