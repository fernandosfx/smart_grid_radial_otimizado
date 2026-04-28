import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# Configurações
# -------------------------------------------------------
# Arquivo da instância base (instância 0, já com overbuild solar/eólico)
BASE_FILE = "dados_microrrede_0.csv"

# Pasta de saída para as instâncias modificadas
OUT_DIR = Path()
OUT_DIR.mkdir(exist_ok=True)


def carregar_base():
    df = pd.read_csv(BASE_FILE)
    # Garante tipos numéricos em t e valor
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df


def salvar(df, sufixo):
    path = OUT_DIR / f"dados_microrrede_{sufixo}.csv"
    df.to_csv(path, index=False)
    print(f"Salvo: {path}")


# -------------------------------------------------------
# Variante 1 – Eólica reduzida pela metade; solar mantida
# -------------------------------------------------------
def instancia_1(base):
    df = base.copy()
    mask_eol = df["tipo"] == "EOLICA"
    df.loc[mask_eol, "valor"] = df.loc[mask_eol, "valor"] * 0.5
    return df


# -------------------------------------------------------
# Variante 2 – Solar reduzida pela metade; eólica mantida
# -------------------------------------------------------
def instancia_2(base):
    df = base.copy()
    mask_sol = df["tipo"] == "SOLAR"
    df.loc[mask_sol, "valor"] = df.loc[mask_sol, "valor"] * 0.5
    return df


# -------------------------------------------------------
# Variante 3 – Sem baterias efetivas
# -------------------------------------------------------
def instancia_3(base):
    df = base.copy()
    mask_bat = (df["tipo"] == "BATERIA") | (df["ativo"] == "bateria")
    df = df.drop(df[mask_bat].index)
    df = df.reset_index(drop=True)  # Opcional: reseta índices
    return df


# -------------------------------------------------------
# Variante 4 – Baterias com grande capacidade de energia
# -------------------------------------------------------
def instancia_4(base, fator=4.0):
    df = base.copy()
    mask_bat = df["tipo"] == "BATERIA"

    for c in ["E_min", "E_max", "E0"]:
        if c in df.columns:
            df.loc[mask_bat, c] = (
                pd.to_numeric(df.loc[mask_bat, c], errors="coerce") * fator
            ).round(6)
    return df


# -------------------------------------------------------
# Helper para redistribuir DEMANDA mantendo perfil temporal
# -------------------------------------------------------
def redistribuir_demanda(df_base, pesos_por_barra, escala=1.0):
    df = df_base.copy()
    mask_dem = df["tipo"] == "DEMANDA"
    dem = df[mask_dem].copy()

    dem["t"] = pd.to_numeric(dem["t"], errors="coerce")
    dem["valor"] = pd.to_numeric(dem["valor"], errors="coerce")

    barras = sorted(dem["i"].unique())
    w = np.array([pesos_por_barra.get(int(b), 1.0) for b in barras], dtype=float)
    w_sum = w.sum()

    for t in sorted(dem["t"].unique()):
        mask_t = dem["t"] == t
        D_t = dem.loc[mask_t, "valor"].sum() * escala
        valores = D_t * w / w_sum
        for b, v in zip(barras, valores):
            idx = dem.index[(dem["t"] == t) & (dem["i"] == b)][0]
            dem.at[idx, "valor"] = round(float(v), 6)

    df.loc[mask_dem, "valor"] = dem["valor"]
    return df


# -------------------------------------------------------
# Variante 5 – Demanda maior perto das baterias
# -------------------------------------------------------
def instancia_5(base):
    df = base.copy()
    # Baterias em 6 e 25 → considerar vizinhas 5, 6, 7, 24, 25, 26
    barras_alta = {5, 6, 7, 24, 25, 26}
    pesos = {b: (3.0 if b in barras_alta else 1.0) for b in range(1, 34)}
    df_mod = redistribuir_demanda(df, pesos_por_barra=pesos, escala=1.0)
    return df_mod


# -------------------------------------------------------
# Variante 6 – Demanda homogênea residencial, baixa
# -------------------------------------------------------
def instancia_6(base):
    df = base.copy()
    # Mesma distribuição espacial e perfil temporal, mas metade da demanda
    mask_dem = df["tipo"] == "DEMANDA"
    df.loc[mask_dem, "valor"] = (
        pd.to_numeric(df.loc[mask_dem, "valor"], errors="coerce") * 0.5
    )
    return df


# -------------------------------------------------------
# Execução principal
# -------------------------------------------------------
if __name__ == "__main__":
    base = carregar_base()

    df1 = instancia_1(base)
    salvar(df1, 1)

    df2 = instancia_2(base)
    salvar(df2, 2)

    df3 = instancia_3(base)
    salvar(df3, 3)

    df4 = instancia_4(base)
    salvar(df4, 4)

    df5 = instancia_5(base)
    salvar(df5, 5)

    df6 = instancia_6(base)
    salvar(df6, 6)