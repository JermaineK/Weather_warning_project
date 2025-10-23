# summarize_lead_decay.py
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results/best_fbeta_thresholds.csv")
print(df[["lead_h","F1","P_f1","R_f1","Fbeta"]])

plt.plot(df["lead_h"], df["Fbeta"], label="Fβ=0.5")
plt.plot(df["lead_h"], df["F1"], label="F1")
plt.xlabel("Lead time (h)")
plt.ylabel("Score")
plt.title("Lead-time decay of forecast skill")
plt.legend()
plt.grid(True)
plt.savefig("results/lead_decay.png", dpi=150)
plt.show()