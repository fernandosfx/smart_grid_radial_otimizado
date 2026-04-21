import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# -----------------------------
# Configurações gerais
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
ARQ = os.path.join(base_dir, "variaveis_otimizadas/variaveis_otimizadas.csv")

OUTDIR = Path("figuras_artigo")
OUTDIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})

# -----------------------------
# Leitura e pré-processamento
# -----------------------------
df_raw = pd.read_csv(ARQ)

# Separa linhas de INFO (ex: custo_total) e dados por período
info = df_raw[df_raw["tempo"] == "INFO"].copy()
df = df_raw[df_raw["tempo"] != "INFO"].copy()

df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

# Pivot por tempo x variável (soma das entidades)
pivot = df.pivot_table(index="tempo",
                       columns="variavel",
                       values="valor",
                       aggfunc="sum").sort_index()

# Garante colunas esperadas, mesmo que algumas não existam
for col in ["PS", "PW", "CS", "CW", "P_ch", "P_dis", "E", "F", "Ploss"]:
    if col not in pivot.columns:
        pivot[col] = 0.0

# Métricas gerais para artigo
custo_total = (
    info.loc[info["variavel"] == "custo_total", "valor"]
    .astype(float).iloc[0]
    if (info["variavel"] == "custo_total").any() else np.nan
)
metricas = {
    "Custo total": custo_total,
    "Energia solar total": pivot["PS"].sum(),
    "Energia eólica total": pivot["PW"].sum(),
    "Energia carregada na bateria": pivot["P_ch"].sum(),
    "Energia final na bateria": pivot["E"].iloc[-1],
    "Fluxo acumulado linha 1-2": pivot["F"].sum(),
    "Perdas acumuladas": pivot["Ploss"].sum()
}
print("Métricas principais:")
for k, v in metricas.items():
    print(f"  {k:30s}: {v:8.4f}")

# -----------------------------
# Figura 1 – Despacho por hora
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(pivot.index, pivot["PS"], label="Solar PV", linewidth=2)
ax.plot(pivot.index, pivot["PW"], label="Eólica", linewidth=2)
# Bateria líquida: descarga positiva, carga negativa
bat_net = pivot["P_dis"] - pivot["P_ch"]
ax.bar(pivot.index, bat_net, label="Bateria (net)", alpha=0.4, color="gray")

ax.set_title("Despacho por hora")
ax.set_xlabel("Hora")
ax.set_ylabel("Potência [MW]")
ax.legend(loc="best")
ax.set_xticks(pivot.index)

fig.tight_layout()
fig.savefig(OUTDIR / "fig1_despacho.png", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Figura 2 – Estado de carga, fluxos e perdas
# -----------------------------
fig, ax1 = plt.subplots(figsize=(7, 4))

# SoC / energia armazenada
ax1.plot(pivot.index, pivot["E"], "o-", color="tab:blue",
         label="Energia armazenada (bateria)")
ax1.set_xlabel("Hora")
ax1.set_ylabel("Energia [MWh]", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xticks(pivot.index)

# Segundo eixo para fluxos/perdas
ax2 = ax1.twinx()
ax2.plot(pivot.index, pivot["F"], "-.", color="tab:orange",
         label="Fluxo ativo (linha 1-2)")
ax2.plot(pivot.index, pivot["Ploss"], "--", color="tab:red",
         label="Perdas ativas (total)")
ax2.set_ylabel("Fluxo / perdas [unid.]", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

# Legenda combinada
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

fig.suptitle("Estado de carga, fluxo e perdas", y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "fig2_soc_fluxos_perdas.png", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Figura 3 – Acumulado por variável
# -----------------------------
agg = (df.groupby("variavel", as_index=False)["valor"]
         .sum()
         .sort_values("valor", ascending=False))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(agg["variavel"], agg["valor"], color="tab:blue", alpha=0.8)

ax.set_title("Energia acumulada por variável")
ax.set_xlabel("Variável")
ax.set_ylabel("Soma no horizonte")

# Rótulo numérico acima de cada barra
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h,
            f"{h:.2f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(OUTDIR / "fig3_acumulado_variaveis.png", bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Figura 4 – Heatmap por série (entidade + variável)
# -----------------------------
# Focamos nas principais variáveis para não poluir demais
vars_keep = ["PS", "PW", "CS", "CW", "P_ch", "P_dis", "E", "F", "Ploss"]
df_sub = df[df["variavel"].isin(vars_keep)].copy()

# Cria um identificador de série: barra/linha/bateria + variável
df_sub["serie"] = df_sub["entidade"] + " | " + df_sub["variavel"]

mat = (df_sub.pivot_table(index="serie",
                          columns="tempo",
                          values="valor",
                          aggfunc="sum")
               .fillna(0))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(mat, ax=ax, cmap="viridis", cbar_kws={"label": "Valor"})
ax.set_title("Mapa de calor das principais séries (entidade x variável)")
ax.set_xlabel("Hora")
ax.set_ylabel("Série")

fig.tight_layout()
fig.savefig(OUTDIR / "fig4_heatmap_series.png", bbox_inches="tight")
plt.close(fig)

print(f"Figuras salvas em: {OUTDIR.resolve()}")