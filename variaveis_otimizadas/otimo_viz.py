import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
import re
import os

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


def gerar_figuras(arquivo_csv, pasta_saida="figuras_artigo"):
    """
    Gera figuras de visualização a partir de um arquivo CSV de resultados.
    
    Cria 4 figuras no total:
    1. fig1_despacho.png - Despacho de geração solar, eólica e bateria por hora
    2. fig2_soc_fluxos_perdas.png - Estado de carga, fluxos e perdas
    3. fig3_acumulado_variaveis.png - Energia acumulada por variável
    4. fig4_heatmap_series.png - Mapa de calor das principais séries
    
    Args:
        arquivo_csv (str): Caminho para o arquivo CSV com resultados da otimização
        pasta_saida (str): Nome da pasta para salvar as figuras (default: "figuras_artigo")
    
    Returns:
        dict: Dicionário com informações sobre as figuras geradas
    """
    
    # Configurações gerais
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10
    })
    
    # Criar pasta de saída
    OUTDIR = Path(f"figuras_artigo_{pasta_saida}")
    OUTDIR.mkdir(exist_ok=True, parents=True)
    
    print(f"  Lendo arquivo CSV: {arquivo_csv}")
    
    # Leitura e pré-processamento
    df_raw = pd.read_csv(arquivo_csv)
    
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
    print(f"  Métricas principais:")
    for k, v in metricas.items():
        print(f"    {k:30s}: {v:8.4f}")
    
    # -------------------------
    # Figura 1 – Despacho por hora
    # -------------------------
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
    
    # -------------------------
    # Figura 2 – Estado de carga, fluxos e perdas
    # -------------------------
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
    
    # -------------------------
    # Figura 3 – Acumulado por variável
    # -------------------------
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
    
    # -------------------------
    # Figura 4 – Heatmap por série (entidade + variável)
    # -------------------------
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
    
    print(f"  ✓ Figuras salvas em: {OUTDIR.resolve()}")
    
    return {
        "pasta_saida": str(OUTDIR),
        "metricas": metricas,
        "figuras_geradas": 4
    }


def _carregar_topologia(instancia_csv):
    """Lê BARRAS e LINHAS de dados_microrrede_k.csv."""
    df = pd.read_csv(instancia_csv)
    barras = df[df["tipo"] == "BARRA"]["i"].astype(int).tolist()
    linhas = df[df["tipo"] == "LINHA"].copy()
    linhas["i"] = linhas["i"].astype(int)
    linhas["j"] = linhas["j"].astype(int)
    # F_max está na coluna 'F_max'
    return barras, linhas


def _parse_linha_entidade(nome):
    """
    Converte 'linha_2-3' -> (2, 3).
    """
    m = re.match(r"linha_(\d+)-(\d+)", str(nome))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def plot_topologia_basica(instancia_csv, pasta_saida="figuras_topologia"):
    """
    Diagrama unifilar 'limpo' com as 33 barras e as linhas.
    """
    OUTDIR = Path(pasta_saida)
    OUTDIR.mkdir(exist_ok=True, parents=True)

    barras, linhas = _carregar_topologia(instancia_csv)

    G = nx.Graph()
    G.add_nodes_from(barras)
    for _, row in linhas.iterrows():
        G.add_edge(int(row["i"]), int(row["j"]))

    # Layout em árvore a partir da barra 1 (se existir)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        # fallback: layout em árvore raiz 1, ou spring_layout
        if 1 in G.nodes:
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        else:
            pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="white",
                           edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes},
                            font_size=8, ax=ax)

    ax.set_axis_off()
    fig.suptitle("Topologia radial da microrrede (diagrama unifilar)", y=0.98)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_topologia_basica.png", bbox_inches="tight")
    plt.close(fig)


def plot_topologia_carregada(instancia_csv,
                             variaveis_csv,
                             pasta_saida="figuras_topologia",
                             limite_norm=1.0):
    """
    Diagrama unifilar destacando linhas mais carregadas
    (largura/cor proporcional a max |F|/F_max).
    """
    OUTDIR = Path(pasta_saida)
    OUTDIR.mkdir(exist_ok=True, parents=True)

    barras, linhas = _carregar_topologia(instancia_csv)

    # dicionário F_max[(i,j)] = Fmax
    Fmax_dict = {}
    for _, row in linhas.iterrows():
        i, j = int(row["i"]), int(row["j"])
        Fmax = float(row["F_max"])
        Fmax_dict[(i, j)] = Fmax
        Fmax_dict[(j, i)] = Fmax  # grafo não orientado

    # carrega fluxos do CSV de variáveis
    df_var = pd.read_csv(variaveis_csv)
    df_F = df_var[df_var["variavel"] == "F"].copy()

    # mapeia entidade 'linha_i-j' -> (i,j)
    df_F["i"], df_F["j"] = zip(*df_F["entidade"].map(_parse_linha_entidade))
    df_F = df_F.dropna(subset=["i", "j"])
    df_F["i"] = df_F["i"].astype(int)
    df_F["j"] = df_F["j"].astype(int)

    # max |F| por linha
    maxF = (df_F.groupby(["i", "j"])["valor"]
              .apply(lambda x: x.abs().max())
              .rename("F_abs_max")
              .reset_index())

    # adiciona F_max
    maxF["F_max"] = maxF.apply(
        lambda r: Fmax_dict.get((int(r["i"]), int(r["j"])), np.nan),
        axis=1
    )
    # fator de carregamento
    maxF["loading"] = maxF["F_abs_max"] / maxF["F_max"]

    # grafo novamente
    G = nx.Graph()
    G.add_nodes_from(barras)
    for _, row in linhas.iterrows():
        G.add_edge(int(row["i"]), int(row["j"]))

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # normalizar loading para [0,1] para cor/espessura
    load_dict = {}
    for _, r in maxF.iterrows():
        load_dict[(int(r["i"]), int(r["j"]))] = r["loading"]

    # construir lista de cores e larguras na mesma ordem de edges
    edges = list(G.edges())
    load_vals = []
    for (u, v) in edges:
        l = max(load_dict.get((u, v), 0.0),
                load_dict.get((v, u), 0.0))
        load_vals.append(min(l, limite_norm))  # corta em limite_norm

    # colormap
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=limite_norm + 0.001)
    colors = [cmap(norm(l)) for l in load_vals]
    widths = [1 + 4 * (l / limite_norm if limite_norm > 0 else 0)
              for l in load_vals]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="white",
                           edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths,
                           edge_color=colors, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes},
                            font_size=8, ax=ax)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("max |F| / F_max")

    fig.suptitle("Topologia radial – destaque por carregamento de linha", y=0.98)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_topologia_carregada.png", bbox_inches="tight")
    plt.close(fig)


def plot_duration_linhas(variaveis_csv,
                         instancia_csv,
                         pasta_saida="figuras_duration",
                         top_n=5):
    """
    Para um cenário (variaveis_csv), plota duration curves de |F|/F_max
    para as top_n linhas com maior carregamento máximo.
    """
    OUTDIR = Path(pasta_saida)
    OUTDIR.mkdir(exist_ok=True, parents=True)

    barras, linhas = _carregar_topologia(instancia_csv)
    Fmax_dict = {}
    for _, row in linhas.iterrows():
        i, j = int(row["i"]), int(row["j"])
        Fmax_dict[(i, j)] = float(row["F_max"])
        Fmax_dict[(j, i)] = float(row["F_max"])

    df_var = pd.read_csv(variaveis_csv)
    df_F = df_var[df_var["variavel"] == "F"].copy()
    df_F["i"], df_F["j"] = zip(*df_F["entidade"].map(_parse_linha_entidade))
    df_F = df_F.dropna(subset=["i", "j"])
    df_F["i"] = df_F["i"].astype(int)
    df_F["j"] = df_F["j"].astype(int)

    # calcula loading por linha/tempo
    def _loading_row(r):
        Fmax = Fmax_dict.get((int(r["i"]), int(r["j"])),
                             Fmax_dict.get((int(r["j"]), int(r["i"])), np.nan))
        return abs(r["valor"]) / Fmax if Fmax and not np.isnan(Fmax) else np.nan

    df_F["loading"] = df_F.apply(_loading_row, axis=1)
    df_F = df_F.dropna(subset=["loading"])

    # top_n linhas pelo max loading
    max_load = (df_F.groupby(["i", "j"])["loading"]
                  .max()
                  .sort_values(ascending=False))
    top_lines = list(max_load.head(top_n).index)

    fig, ax = plt.subplots(figsize=(7, 4))
    for (i, j) in top_lines:
        serie = df_F[(df_F["i"] == i) & (df_F["j"] == j)]["loading"].values
        if len(serie) == 0:
            continue
        # curva de duração: ordena decrescente
        serie_sorted = np.sort(serie)[::-1]
        x = np.arange(1, len(serie_sorted) + 1)
        ax.plot(x, serie_sorted, label=f"linha_{i}-{j}")

    ax.set_xlabel("Horas (ordenadas por carregamento)")
    ax.set_ylabel("|F| / F_max")
    ax.set_title(f"Curvas de duração de carregamento – top {top_n} linhas")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_duration_linhas_topN.png", bbox_inches="tight")
    plt.close(fig)


def _carregar_demanda(instancia_csv):
    """Retorna DataFrame DEMANDA com colunas: barra, tempo, demanda."""
    df = pd.read_csv(instancia_csv)
    df_dem = df[df["tipo"] == "DEMANDA"].copy()
    df_dem["i"] = df_dem["i"].astype(int)
    df_dem["t"] = df_dem["t"].astype(int)
    df_dem["valor"] = pd.to_numeric(df_dem["valor"], errors="coerce")
    return df_dem.rename(columns={"i": "barra", "t": "tempo", "valor": "demanda"})


def plot_duration_ren_netload(variaveis_csv,
                              instancia_csv,
                              pasta_saida="figuras_duration"):
    """
    Para um cenário, calcula:
      P_ren(t) = sum_b (PS + PW)
      P_bat(t) = sum_bat (P_dis - P_ch)
      Demanda(t) = sum_b DEMANDA
      net_load(t) = Demanda(t) - P_ren(t) - P_bat(t)
    e gera curvas de duração de P_ren e net_load.
    """
    OUTDIR = Path(pasta_saida)
    OUTDIR.mkdir(exist_ok=True, parents=True)

    df_var = pd.read_csv(variaveis_csv)
    df_dem = _carregar_demanda(instancia_csv)

    # renováveis: PS e PW
    df_ren = df_var[df_var["variavel"].isin(["PS", "PW"])].copy()
    df_ren["tempo"] = pd.to_numeric(df_ren["tempo"], errors="coerce")
    P_ren = (df_ren.groupby("tempo")["valor"]
                    .sum()
                    .rename("P_ren"))

    # bateria
    df_bat = df_var[df_var["variavel"].isin(["P_ch", "P_dis"])].copy()
    df_bat["tempo"] = pd.to_numeric(df_bat["tempo"], errors="coerce")
    df_bat_piv = (df_bat.pivot_table(index="tempo",
                                     columns="variavel",
                                     values="valor",
                                     aggfunc="sum")
                        .fillna(0))
    df_bat_piv["P_bat"] = df_bat_piv.get("P_dis", 0) - df_bat_piv.get("P_ch", 0)

    # demanda total por hora
    Dem = df_dem.groupby("tempo")["demanda"].sum().rename("Demanda")

    df_all = pd.concat([P_ren, df_bat_piv["P_bat"], Dem], axis=1).fillna(0)
    df_all["net_load"] = df_all["Demanda"] - df_all["P_ren"] - df_all["P_bat"]

    # duration curves
    for col, nome in [("P_ren", "Potência renovável total"),
                      ("net_load", "Net-load (Dem - Ren - Bateria)")]:
        serie = df_all[col].values
        serie_sorted = np.sort(serie)[::-1]
        x = np.arange(1, len(serie_sorted) + 1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, serie_sorted, linewidth=2)
        ax.set_xlabel("Horas (ordenadas por valor)")
        ax.set_ylabel("Potência [MW]")
        ax.set_title(f"Curva de duração – {nome}")
        fig.tight_layout()
        fig.savefig(OUTDIR / f"fig_duration_{col}.png", bbox_inches="tight")
        plt.close(fig)


def gerar_tabela_parametros_cenarios(pasta_conjuntos="conjuntos_instancias",
                                     out_csv="tabela_parametros_cenarios.csv"):
    """
    Lê dados_microrrede_k.csv para k=0..10 e gera uma tabela de
    'parâmetros' agregados por cenário:
      - demanda total diária
      - demanda pico
      - capacidade solar total (soma dos máximos por barra)
      - capacidade eólica total (soma dos máximos por barra)
      - média / min / max de F_max
    """
    pasta = Path(pasta_conjuntos)
    linhas = []

    for k in range(0, 11):
        arq = pasta / f"dados_microrrede_{k}.csv"
        if not arq.exists():
            continue
        df = pd.read_csv(arq)

        # demanda
        df_dem = df[df["tipo"] == "DEMANDA"].copy()
        df_dem["valor"] = pd.to_numeric(df_dem["valor"], errors="coerce")
        df_dem["t"] = pd.to_numeric(df_dem["t"], errors="coerce")
        Dem_t = df_dem.groupby("t")["valor"].sum()
        Dem_total = Dem_t.sum()
        Dem_pico = Dem_t.max()

        # solar/eólica – capacidade instalada (máx por barra)
        df_sol = df[df["tipo"] == "SOLAR"].copy()
        df_eol = df[df["tipo"] == "EOLICA"].copy()
        for d in (df_sol, df_eol):
            d["valor"] = pd.to_numeric(d["valor"], errors="coerce")
            d["i"] = pd.to_numeric(d["i"], errors="coerce")

        cap_solar = (df_sol.groupby("i")["valor"].max().sum()
                     if not df_sol.empty else 0.0)
        cap_eolica = (df_eol.groupby("i")["valor"].max().sum()
                      if not df_eol.empty else 0.0)

        # F_max
        df_lin = df[df["tipo"] == "LINHA"].copy()
        df_lin["F_max"] = pd.to_numeric(df_lin["F_max"], errors="coerce")
        Fmax_mean = df_lin["F_max"].mean()
        Fmax_min = df_lin["F_max"].min()
        Fmax_max = df_lin["F_max"].max()

        linhas.append({
            "cenario": k,
            "Dem_total": Dem_total,
            "Dem_pico": Dem_pico,
            "Cap_solar": cap_solar,
            "Cap_eolica": cap_eolica,
            "Fmax_media": Fmax_mean,
            "Fmax_min": Fmax_min,
            "Fmax_max": Fmax_max,
        })

    df_par = pd.DataFrame(linhas).set_index("cenario").sort_index()
    df_par.to_csv(out_csv)
    return df_par


def gerar_tabela_linhas_criticas(consolidado_csv,
                                 instancia_csv,
                                 out_csv="tabela_linhas_criticas.csv",
                                 top_n=5):
    """
    Para cada cenário, seleciona as top_n linhas com maior max |F|/F_max
    e calcula:
      - max_loading
      - horas acima de 80% e 90% do limite.
    Gera um CSV com colunas: cenario, linha, i, j, max_loading,
                             horas_>0.8, horas_>0.9.
    """
    barras, linhas = _carregar_topologia(instancia_csv)
    Fmax_dict = {}
    for _, row in linhas.iterrows():
        i, j = int(row["i"]), int(row["j"])
        Fmax_dict[(i, j)] = float(row["F_max"])
        Fmax_dict[(j, i)] = float(row["F_max"])

    df_var = pd.read_csv(consolidado_csv)
    df_F = df_var[df_var["variavel"] == "F"].copy()
    df_F["i"], df_F["j"] = zip(*df_F["entidade"].map(_parse_linha_entidade))
    df_F = df_F.dropna(subset=["i", "j"])
    df_F["i"] = df_F["i"].astype(int)
    df_F["j"] = df_F["j"].astype(int)
    df_F["cenario"] = pd.to_numeric(df_F["cenario"], errors="coerce")
    df_F["tempo"] = pd.to_numeric(df_F["tempo"], errors="coerce")

    def _load_row(r):
        Fmax = Fmax_dict.get((int(r["i"]), int(r["j"])),
                             Fmax_dict.get((int(r["j"]), int(r["i"])), np.nan))
        return abs(r["valor"]) / Fmax if Fmax and not np.isnan(Fmax) else np.nan

    df_F["loading"] = df_F.apply(_load_row, axis=1)
    df_F = df_F.dropna(subset=["loading"])

    registros = []
    for cen, grupo_c in df_F.groupby("cenario"):
        # max loading por linha
        max_load = (grupo_c.groupby(["i", "j"])["loading"]
                           .max()
                           .sort_values(ascending=False))
        for (i, j), val in max_load.head(top_n).items():
            g_l = grupo_c[(grupo_c["i"] == i) & (grupo_c["j"] == j)]
            horas_08 = (g_l["loading"] >= 0.8).sum()
            horas_09 = (g_l["loading"] >= 0.9).sum()
            registros.append({
                "cenario": int(cen),
                "linha": f"linha_{i}-{j}",
                "i": int(i),
                "j": int(j),
                "max_loading": val,
                "horas_ge_0.8": int(horas_08),
                "horas_ge_0.9": int(horas_09),
            })

    df_crit = pd.DataFrame(registros)
    df_crit.to_csv(out_csv, index=False)
    return df_crit


def gerar_tabela_regime_baterias(consolidado_csv,
                                 out_csv="tabela_regime_baterias.csv",
                                 tol=1e-4):
    """
    Para cada cenário e cada bateria:
      - energia total carregada (sum P_ch)
      - energia total descarregada (sum P_dis)
      - horas em carga (P_ch > tol)
      - horas em descarga (P_dis > tol)
      - SoC mínimo e máximo (E).
    """
    df = pd.read_csv(consolidado_csv)
    df["cenario"] = pd.to_numeric(df["cenario"], errors="coerce")
    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    # baterias: entidade 'bateria_X'
    mask_bat = df["entidade"].str.startswith("bateria_")
    df_bat = df[mask_bat].copy()

    regs = []
    for (cen, ent), grupo in df_bat.groupby(["cenario", "entidade"]):
        g_ch = grupo[grupo["variavel"] == "P_ch"]
        g_dis = grupo[grupo["variavel"] == "P_dis"]
        g_E = grupo[grupo["variavel"] == "E"]

        E_ch = g_ch["valor"].sum()
        E_dis = g_dis["valor"].sum()
        horas_ch = (g_ch["valor"] > tol).sum()
        horas_dis = (g_dis["valor"] > tol).sum()
        SoC_min = g_E["valor"].min() if not g_E.empty else np.nan
        SoC_max = g_E["valor"].max() if not g_E.empty else np.nan

        regs.append({
            "cenario": int(cen),
            "bateria": ent,
            "E_carga_total": E_ch,
            "E_descarga_total": E_dis,
            "horas_carga": int(horas_ch),
            "horas_descarga": int(horas_dis),
            "SoC_min": SoC_min,
            "SoC_max": SoC_max,
        })

    df_reg = pd.DataFrame(regs)
    df_reg.to_csv(out_csv, index=False)
    return df_reg




# ============================================================
# EXECUÇÃO PADRÃO (compatibilidade)
# ============================================================

#if __name__ == "__main__":
#    # Executar com parâmetros padrão
#    resultado = gerar_figuras(
#        arquivo_csv=f"variaveis_otimizadas/variaveis_otimizadas_{idx_cenario}.csv",
#        pasta_saida="artigo"
#    )
#    print("\nFiguras geradas com sucesso!")