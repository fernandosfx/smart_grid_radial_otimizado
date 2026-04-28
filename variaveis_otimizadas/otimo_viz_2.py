import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10
})


def _preprocess_csv(arquivo_csv):
    """Lê CSV no formato do modelo e separa INFO dos dados por tempo."""
    df_raw = pd.read_csv(arquivo_csv)
    info = df_raw[df_raw["tempo"] == "INFO"].copy()
    df = df_raw[df_raw["tempo"] != "INFO"].copy()

    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    # Alguns arquivos individuais podem não ter coluna cenario; trata isso
    if "cenario" not in df.columns:
        df["cenario"] = 0
    if "cenario" not in info.columns:
        info["cenario"] = 0

    return df, info


def gerar_figuras_cenario(arquivo_csv, pasta_saida="figuras_artigo_cenario"):
    """
    Gera figuras detalhadas para um único cenário (arquivo CSV).
    Similar ao otimo_viz original, mas parametrizado e com extensões.
    """
    OUTDIR = Path(pasta_saida)
    OUTDIR.mkdir(exist_ok=True, parents=True)

    print(f"Lendo arquivo de cenário: {arquivo_csv}")
    df, info = _preprocess_csv(arquivo_csv)

    # Descobre cenario
    cenarios = df["cenario"].unique()
    cenario = int(cenarios[0]) if len(cenarios) == 1 else cenarios

    # Pivot tempo x variável (soma em todas entidades)
    pivot = df.pivot_table(index="tempo",
                           columns="variavel",
                           values="valor",
                           aggfunc="sum").sort_index()

    # Garantir colunas
    for col in ["PS", "PW", "CS", "CW", "P_ch", "P_dis", "E", "F", "Ploss"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    # Métricas resumidas
    custo_total = (
        info.loc[info["variavel"] == "custo_total", "valor"]
        .astype(float).iloc[0]
        if (info["variavel"] == "custo_total").any() else np.nan
    )
    metricas = {
        "cenario": cenario,
        "Custo total": custo_total,
        "Energia solar total": pivot["PS"].sum(),
        "Energia eólica total": pivot["PW"].sum(),
        "Curtailment solar total": pivot["CS"].sum(),
        "Curtailment eólico total": pivot["CW"].sum(),
        "Energia carregada bateria": pivot["P_ch"].sum(),
        "Energia descarregada bateria": pivot["P_dis"].sum(),
        "Energia final bateria": pivot["E"].iloc[-1],
        "Perdas acumuladas": pivot["Ploss"].sum()
    }
    print("  Métricas principais do cenário:")
    for k, v in metricas.items():
        print(f"    {k:32s}: {v:8.4f}")

    # -------- Figura 1 – Despacho horário --------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pivot.index, pivot["PS"], label="Solar PV", linewidth=2)
    ax.plot(pivot.index, pivot["PW"], label="Eólica", linewidth=2)
    bat_net = pivot["P_dis"] - pivot["P_ch"]
    ax.bar(pivot.index, bat_net, label="Bateria (net)", alpha=0.4, color="gray")
    ax.set_title(f"Despacho por hora – cenário {cenario}")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Potência [MW]")
    ax.legend(loc="best")
    ax.set_xticks(pivot.index)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"fig1_despacho_c{cenario}.png", bbox_inches="tight")
    plt.close(fig)

    # -------- Figura 2 – SoC, fluxos e perdas --------
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(pivot.index, pivot["E"], "o-", color="tab:blue",
             label="Energia armazenada (bateria)")
    ax1.set_xlabel("Hora")
    ax1.set_ylabel("Energia [MWh]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(pivot.index)

    ax2 = ax1.twinx()
    ax2.plot(pivot.index, pivot["F"], "-.", color="tab:orange",
             label="Fluxo ativo (agregado)")
    ax2.plot(pivot.index, pivot["Ploss"], "--", color="tab:red",
             label="Perdas ativas (total)")
    ax2.set_ylabel("Fluxo / perdas [u.]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.suptitle(f"SoC, fluxos e perdas – cenário {cenario}", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"fig2_soc_fluxos_perdas_c{cenario}.png",
                bbox_inches="tight")
    plt.close(fig)

    # -------- Figura 3 – Energia acumulada por variável --------
    agg = (df.groupby("variavel", as_index=False)["valor"]
             .sum()
             .sort_values("valor", ascending=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(agg["variavel"], agg["valor"],
                  color="tab:blue", alpha=0.8)
    ax.set_title(f"Energia acumulada por variável – cenário {cenario}")
    ax.set_xlabel("Variável")
    ax.set_ylabel("Soma no horizonte")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"fig3_acumulado_variaveis_c{cenario}.png",
                bbox_inches="tight")
    plt.close(fig)

    # -------- Figura 4 – Heatmap séries (entidade|variável × tempo) --------
    vars_keep = ["PS", "PW", "CS", "CW", "P_ch", "P_dis", "E", "F", "Ploss"]
    df_sub = df[df["variavel"].isin(vars_keep)].copy()
    df_sub["serie"] = df_sub["entidade"] + " | " + df_sub["variavel"]

    mat = (df_sub.pivot_table(index="serie",
                              columns="tempo",
                              values="valor",
                              aggfunc="sum")
                 .fillna(0))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(mat, ax=ax, cmap="viridis", cbar_kws={"label": "Valor"})
    ax.set_title(f"Mapa de calor séries × hora – cenário {cenario}")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Série")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"fig4_heatmap_series_c{cenario}.png",
                bbox_inches="tight")
    plt.close(fig)

    return {
        "cenario": cenario,
        "metricas": metricas,
        "pasta_saida": str(OUTDIR)
    }


def gerar_analise_consolidada(arquivo_csv_consolidado,
                              pasta_saida="figuras_artigo_global"):
    """
    Gera todas as figuras e tabelas consolidadas entre cenários a partir do
    arquivo variaveis_otimizadas_consolidado.csv.
    """
    OUTDIR = Path(pasta_saida)
    OUTDIR.mkdir(exist_ok=True, parents=True)

    print(f"Lendo CSV consolidado: {arquivo_csv_consolidado}")
    df_raw = pd.read_csv(arquivo_csv_consolidado)
    info = df_raw[df_raw["tempo"] == "INFO"].copy()
    df = df_raw[df_raw["tempo"] != "INFO"].copy()

    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["cenario"] = pd.to_numeric(df["cenario"], errors="coerce")

    # ---------------- Métricas agregadas por cenário ----------------
    # custo_total
    custo_por_cenario = (
        info[info["variavel"] == "custo_total"]
        .groupby("cenario")["valor"].sum()
        .rename("custo_total")
    )

    # soma por variavel/cenario ao longo do horizonte
    soma_var_cen = (
        df.groupby(["cenario", "variavel"])["valor"]
          .sum()
          .unstack("variavel")
          .fillna(0)
    )

    for col in ["PS", "PW", "CS", "CW", "P_ch", "P_dis", "E", "Ploss"]:
        if col not in soma_var_cen.columns:
            soma_var_cen[col] = 0.0

    # energia final em baterias (E no último tempo) por cenário
    last_E = (
        df[df["variavel"] == "E"]
        .sort_values(["cenario", "tempo"])
        .groupby("cenario")["valor"]
        .last()
        .rename("E_final")
    )

    metricas_cen = soma_var_cen.copy()
    metricas_cen["custo_total"] = custo_por_cenario
    metricas_cen["E_final"] = last_E

    metricas_cen = metricas_cen.rename(columns={
        "PS": "E_solar_total",
        "PW": "E_eolica_total",
        "CS": "curtailment_solar",
        "CW": "curtailment_eolica",
        "P_ch": "E_carga_bat",
        "P_dis": "E_descarga_bat",
        "Ploss": "perdas_totais"
    })

    metricas_cen.to_csv(OUTDIR / "tabela_metricas_por_cenario.csv")

    print("Métricas por cenário:")
    print(metricas_cen)

    # -------- Figura G1 – Barras de métricas por cenário --------
    cols_plot = [
        "custo_total",
        "E_solar_total",
        "E_eolica_total",
        "perdas_totais",
        "curtailment_solar",
        "curtailment_eolica",
        "E_carga_bat",
        "E_descarga_bat"
    ]
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 6))
    axes = axes.ravel()
    cen_idx = metricas_cen.index.astype(int)

    for ax, col in zip(axes, cols_plot):
        if col not in metricas_cen.columns:
            ax.axis("off")
            continue
        ax.bar(cen_idx, metricas_cen[col], color="tab:blue")
        ax.set_title(col)
        ax.set_xlabel("Cenário")
        ax.set_xticks(cen_idx)

    fig.suptitle("Métricas agregadas por cenário", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_g1_metricas_barras.png", bbox_inches="tight")
    plt.close(fig)

    # -------- Figura G2 – Boxplot de perdas horárias por cenário --------
    df_ploss = df[df["variavel"] == "Ploss"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_ploss, x="cenario", y="valor", ax=ax)
    ax.set_title("Distribuição horária de perdas por cenário")
    ax.set_xlabel("Cenário")
    ax.set_ylabel("Perdas horárias")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_g2_boxplot_perdas.png", bbox_inches="tight")
    plt.close(fig)

    # -------- Figura G3 – Séries de renováveis por cenário --------
    df_ren = df[df["variavel"].isin(["PS", "PW"])].copy()
    df_ren_agg = (
        df_ren.groupby(["cenario", "tempo"])["valor"]
              .sum()
              .reset_index()
              .rename(columns={"valor": "P_ren"})
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for cen, grupo in df_ren_agg.groupby("cenario"):
        ax.plot(grupo["tempo"], grupo["P_ren"], alpha=0.6, label=f"cen {int(cen)}")
    ax.set_title("Potência renovável total por hora – todos cenários")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Potência renovável total")
    ax.set_xticks(sorted(df_ren_agg["tempo"].unique()))
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_g3_series_renovaveis_cenarios.png",
                bbox_inches="tight")
    plt.close(fig)

    # -------- Figura G4 – Renováveis vs perdas (dispersão) --------
    df_ploss_hour = (
        df[df["variavel"] == "Ploss"]
        .groupby(["cenario", "tempo"])["valor"]
        .sum()
        .reset_index()
        .rename(columns={"valor": "Ploss"})
    )

    df_merge = pd.merge(df_ren_agg, df_ploss_hour,
                        on=["cenario", "tempo"], how="inner")

    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(df_merge["P_ren"], df_merge["Ploss"],
                         c=df_merge["cenario"], cmap="viridis", alpha=0.7)
    ax.set_title("Perdas vs potência renovável (pontos = hora×cenário)")
    ax.set_xlabel("P_ren [MW]")
    ax.set_ylabel("Perdas [u.]")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cenário")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_g4_scatter_renovaveis_perdas.png",
                bbox_inches="tight")
    plt.close(fig)

    # -------- Figura G5 – Heatmap PS barra×hora (agregado em cenários) --------
    df_ps = df[df["variavel"] == "PS"].copy()
    if not df_ps.empty:
        ps_mat = (df_ps.groupby(["entidade", "tempo"])["valor"]
                          .sum()
                          .unstack("tempo")
                          .fillna(0))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(ps_mat, ax=ax, cmap="magma", cbar_kws={"label": "PS"})
        ax.set_title("Mapa de calor PS (barra × hora, agreg. cenários)")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Barra")
        fig.tight_layout()
        fig.savefig(OUTDIR / "fig_g5_heatmap_PS_barra_hora.png",
                    bbox_inches="tight")
        plt.close(fig)

    # -------- Figura G6 – Heatmap PW barra×hora (agregado em cenários) --------
    df_pw = df[df["variavel"] == "PW"].copy()
    if not df_pw.empty:
        pw_mat = (df_pw.groupby(["entidade", "tempo"])["valor"]
                          .sum()
                          .unstack("tempo")
                          .fillna(0))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pw_mat, ax=ax, cmap="viridis", cbar_kws={"label": "PW"})
        ax.set_title("Mapa de calor PW (barra × hora, agreg. cenários)")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Barra")
        fig.tight_layout()
        fig.savefig(OUTDIR / "fig_g6_heatmap_PW_barra_hora.png",
                    bbox_inches="tight")
        plt.close(fig)

    # -------- Figura G7 – Correlação entre variáveis agregadas --------
    # Construímos features (por cenário, tempo)
    df_feat = df_merge.copy()
    # bateria líquida
    df_bat = df[df["variavel"].isin(["P_ch", "P_dis"])].copy()
    df_bat_piv = (
        df_bat.pivot_table(index=["cenario", "tempo"],
                           columns="variavel",
                           values="valor",
                           aggfunc="sum")
               .fillna(0)
    )
    df_feat = df_feat.set_index(["cenario", "tempo"])
    df_feat["P_ch"] = df_bat_piv.get("P_ch", 0.0)
    df_feat["P_dis"] = df_bat_piv.get("P_dis", 0.0)
    df_feat["P_bat_net"] = df_feat["P_dis"] - df_feat["P_ch"]

    # curtailment total por hora
    df_curt = df[df["variavel"].isin(["CS", "CW"])].copy()
    df_curt_hour = (
        df_curt.groupby(["cenario", "tempo"])["valor"]
        .sum()
        .rename("curtailment")
    )
    df_feat["curtailment"] = df_curt_hour

    corr = df_feat[["P_ren", "Ploss", "P_bat_net", "curtailment"]].corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlação entre variáveis agregadas")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_g7_corr_features.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "pasta_saida": str(OUTDIR),
        "tabela_metricas": metricas_cen
    }