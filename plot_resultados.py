import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar matriz de confusión y f-scores
cm = np.loadtxt('cmatriz.csv', delimiter=',', dtype=int)
fs = np.loadtxt('fscores.csv', delimiter=',')

# Crear figura
fig = plt.figure(figsize=(8, 5))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1])

# ---- Parte superior: matriz de confusión ----
ax0 = fig.add_subplot(gs[0, :])
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", cbar=False, ax=ax0,
            xticklabels=[1, -1], yticklabels=[1, -1])
ax0.set_xlabel("Real Value")
ax0.set_ylabel("Predicted Value")
ax0.set_title("Confusion Matrix")

# ---- Parte inferior: texto f-score ----
ax1 = fig.add_subplot(gs[1, :])
ax1.axis("off")
f1, f2 = fs[0]*100, fs[1]*100
texto = f"F-score( %):     Clase#1={f1:.2f}     Clase#2= {f2:.2f}"
ax1.text(0.5, 0.5, texto, fontsize=16, weight='bold', color='blue', ha='center', va='center')
ax1.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, fill=False, edgecolor='red', linewidth=2))

# Guardar figura
plt.tight_layout()
plt.savefig("visual_resultados_rrk.png")
plt.show()
