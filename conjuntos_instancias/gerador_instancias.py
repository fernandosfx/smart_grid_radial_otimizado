"""
gerador_instancias.py
=====================
Gera instâncias CSV (dados_microrrede_1.csv … dados_microrrede_10.csv)
a partir da instância base (dados_microrrede_0.csv).

Cada função é PURA: recebe o DataFrame base e devolve uma cópia modificada,
sem efeitos colaterais — facilita testes unitários e reprodução dos experimentos.

Cenários
--------
1  – Baixa irradiância        : SOLAR × 0.5
2  – Baixo vento              : EOLICA × 0.5
3  – Bateria muito maior      : E_max×3, P_ch_max×2, P_dis_max×2
4  – Sem bateria              : remove linhas BATERIA / ativo==bateria
5  – Vertimento barato        : c_curt × 0.2 (≈ 1/5 do base)
6  – Vertimento muito caro    : c_curt × 7.5
7  – Perdas caras             : c_loss × 3 em todas as linhas
8  – Carga industrial (DG/BESS): redistribui demanda — barras c/ DG/BESS recebem 3×
9  – Carga leve residencial   : DEMANDA × 0.5 (perfil e distribuição mantidos)
10 – Rede limitada            : F_max × 0.4 nas linhas críticas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# ── Configuração ──────────────────────────────────────────────────────────────
# Configuração para tornar o script versátil. Ajuste estes valores conforme o dataset.
CONFIG = {
    'columns': {
        'tipo': 'tipo',
        't': 't',
        'valor': 'valor',
        'i': 'i',
        'j': 'j',
        'E_max': 'E_max',
        'P_ch_max': 'P_ch_max',
        'P_dis_max': 'P_dis_max',
        'c_curt': 'c_curt',  # ou None se não existir
        'c_loss': 'c_loss',  # ou None se não existir
        'F_max': 'F_max',    # ou None se não existir
        'ativo': 'ativo',    # ou None se não existir
    },
    'barras_dg_bess': {6, 14, 18, 25, 30},  # barras com DG solar/eólica ou baterias
    'barras_vizinhas_dg': {5, 6, 7, 13, 14, 15, 17, 18, 19, 24, 25, 26, 29, 30, 31},  # vizinhas das DGs
    'linhas_criticas_fator': {
        # (i, j): fator de redução de F_max
        (1, 2): 0.40,
        (2, 3): 0.40,
        (3, 4): 0.50,
        (4, 5): 0.50,
        (2, 19): 0.40,
        (19, 20): 0.50,
    },
}

# ── I/O ───────────────────────────────────────────────────────────────────────
def carregar_base(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_t = CONFIG['columns']['t']
    col_valor = CONFIG['columns']['valor']
    if col_t in df.columns:
        df[col_t] = pd.to_numeric(df[col_t], errors="coerce")
    if col_valor in df.columns:
        df[col_valor] = pd.to_numeric(df[col_valor], errors="coerce")
    return df


def salvar(df: pd.DataFrame, sufixo: int | str, out_dir: Path) -> None:
    path = out_dir / f"dados_microrrede_{sufixo}.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ Salvo: {path}  ({len(df)} linhas)")


# ── Helper: escalar coluna numérica em linhas filtradas ───────────────────────
def _escalar(df: pd.DataFrame, mask, col: str, fator: float) -> pd.DataFrame:
    """Multiplica `col` por `fator` nas linhas indicadas por `mask`."""
    if col in df.columns:
        df.loc[mask, col] = (
            pd.to_numeric(df.loc[mask, col], errors="coerce") * fator
        ).round(6)
    return df


# ── Helper: redistribuir demanda mantendo total por período ───────────────────
def redistribuir_demanda(
    df: pd.DataFrame,
    pesos_por_barra: dict[int, float],
    escala: float = 1.0,
) -> pd.DataFrame:
    """
    Redistribui a demanda espacialmente mantendo o total por período.
    `escala` permite também escalar o total (ex.: 0.5 para carga leve).
    """
    df = df.copy()
    col_tipo = CONFIG['columns']['tipo']
    col_i = CONFIG['columns']['i']
    col_t = CONFIG['columns']['t']
    col_valor = CONFIG['columns']['valor']
    mask_dem = df[col_tipo] == "DEMANDA"
    dem = df.loc[mask_dem].copy()
    dem[col_t] = pd.to_numeric(dem[col_t], errors="coerce")
    dem[col_valor] = pd.to_numeric(dem[col_valor], errors="coerce")

    barras = sorted(dem[col_i].unique())
    w = np.array([pesos_por_barra.get(int(b), 1.0) for b in barras], dtype=float)
    w_sum = w.sum()

    for t in sorted(dem[col_t].unique()):
        mask_t = dem[col_t] == t
        D_t = dem.loc[mask_t, col_valor].sum() * escala
        vals = D_t * w / w_sum
        for b, v in zip(barras, vals):
            idx = dem.index[(dem[col_t] == t) & (dem[col_i] == b)]
            if len(idx):
                dem.at[idx[0], col_valor] = round(float(v), 6)

    df.loc[mask_dem, col_valor] = dem[col_valor].values
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 1 — Baixa irradiância (SOLAR × 0.5)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_1(base: pd.DataFrame) -> pd.DataFrame:
    """
    Representa dias nublados / estação menos ensolarada.
    Potencial solar reduzido a 50 %; eólica e demanda inalteradas.
    Esperado: menos curtailment solar, maior pressão sobre baterias
    para cobrir balanço diurno.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    mask = df[col_tipo] == "SOLAR"
    col_valor = CONFIG['columns']['valor']
    df = _escalar(df, mask, col_valor, 0.5)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 2 — Baixo vento (EOLICA × 0.5)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_2(base: pd.DataFrame) -> pd.DataFrame:
    """
    Simula ano ruim de vento.
    Potencial eólico reduzido a 50 %; solar e demanda inalteradas.
    Esperado: menor geração noturna, mais pressão sobre solar + bateria.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    mask = df[col_tipo] == "EOLICA"
    col_valor = CONFIG['columns']['valor']
    df = _escalar(df, mask, col_valor, 0.5)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 3 — Bateria muito maior (E_max×3, P_ch_max×2, P_dis_max×2)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_3(base: pd.DataFrame) -> pd.DataFrame:
    """
    Quantifica o valor marginal de expandir armazenamento.
    E_max triplicado; E_min e E0 mantidos; capacidades de potência dobradas.
    Esperado: forte redução de curtailment e mudanças nos fluxos de linha.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    mask = df[col_tipo] == "BATERIA"

    # Energia: E_max × 3 (E_min e E0 ficam iguais para manter SoC inicial válido)
    col_E_max = CONFIG['columns']['E_max']
    if col_E_max in df.columns:
        df = _escalar(df, mask, col_E_max, 3.0)

    # Potência: P_ch_max e P_dis_max × 2
    col_P_ch_max = CONFIG['columns']['P_ch_max']
    col_P_dis_max = CONFIG['columns']['P_dis_max']
    if col_P_ch_max in df.columns:
        df = _escalar(df, mask, col_P_ch_max, 2.0)
    if col_P_dis_max in df.columns:
        df = _escalar(df, mask, col_P_dis_max, 2.0)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 4 — Sem bateria (remove linhas BATERIA e ativo==bateria)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_4(base: pd.DataFrame) -> pd.DataFrame:
    """
    Isola o papel do armazenamento.
    Remove todas as linhas de parametrização de baterias.
    Esperado: curtailment aumenta em horas de overbuild, fluxos maiores.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    mask_bat = (df[col_tipo] == "BATERIA")
    col_ativo = CONFIG['columns']['ativo']
    if col_ativo in df.columns:
        mask_bat = mask_bat | (
            df[col_ativo].astype(str).str.lower().str.strip() == "bateria"
        )
    df = df.loc[~mask_bat].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 5 — Vertimento barato (c_curt × 0.2)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_5(base: pd.DataFrame) -> pd.DataFrame:
    """
    Testa sensibilidade à penalidade de vertimento.
    c_curt reduzido para ~1/5: verter renovável fica mais barato que usar bateria.
    Esperado: baterias menos utilizadas, curtailment cresce.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    col_c_curt = CONFIG['columns']['c_curt']
    col_valor = CONFIG['columns']['valor']

    # c_curt pode estar em coluna dedicada (tipo==CUSTO_CURT) ou em coluna lateral
    if (df[col_tipo] == "CUSTO_CURT").any():
        mask = df[col_tipo] == "CUSTO_CURT"
        df = _escalar(df, mask, col_valor, 0.2)
    elif col_c_curt in df.columns:
        df[col_c_curt] = pd.to_numeric(df[col_c_curt], errors="coerce") * 0.2
    else:
        # Fallback: escalar coluna "valor" onde tipo é SOLAR ou EOLICA e
        # a linha for de custo (marcada como CUSTO)
        mask = df[col_tipo].isin(["CUSTO_SOLAR", "CUSTO_EOLICA"])
        if mask.any():
            df = _escalar(df, mask, col_valor, 0.2)
        else:
            print("  ⚠️  c_curt não encontrado — verifique o schema do CSV base.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 6 — Vertimento muito caro (c_curt × 7.5)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_6(base: pd.DataFrame) -> pd.DataFrame:
    """
    Aproxima VOLL alto: perder renovável é quase proibido.
    c_curt × 7.5 (5–10× conforme especificação).
    Esperado: baterias usadas ao máximo, fluxos e perdas podem crescer.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    col_c_curt = CONFIG['columns']['c_curt']
    col_valor = CONFIG['columns']['valor']

    if (df[col_tipo] == "CUSTO_CURT").any():
        mask = df[col_tipo] == "CUSTO_CURT"
        df = _escalar(df, mask, col_valor, 7.5)
    elif col_c_curt in df.columns:
        df[col_c_curt] = pd.to_numeric(df[col_c_curt], errors="coerce") * 7.5
    else:
        mask = df[col_tipo].isin(["CUSTO_SOLAR", "CUSTO_EOLICA"])
        if mask.any():
            df = _escalar(df, mask, col_valor, 7.5)
        else:
            print("  ⚠️  c_curt não encontrado — verifique o schema do CSV base.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 7 — Perdas caras (c_loss × 3 em todas as linhas)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_7(base: pd.DataFrame) -> pd.DataFrame:
    """
    Energia de rede "escassa": penaliza fortemente perdas Joule.
    c_loss triplicado em todas as linhas.
    Esperado: modelo ajusta fluxos e uso de bateria para minimizar perdas,
    podendo aceitar mais curtailment em troca de fluxos menores.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    col_c_loss = CONFIG['columns']['c_loss']
    col_valor = CONFIG['columns']['valor']

    if (df[col_tipo] == "LINHA").any():
        mask = df[col_tipo] == "LINHA"
        if col_c_loss in df.columns:
            df = _escalar(df, mask, col_c_loss, 3.0)
        elif col_valor in df.columns:
            # Se o CSV armazena c_loss na coluna valor para linhas
            df = _escalar(df, mask, col_valor, 3.0)
    else:
        if col_c_loss in df.columns:
            df[col_c_loss] = pd.to_numeric(df[col_c_loss], errors="coerce") * 3.0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 8 — Carga industrial próxima às DGs/BESS
# ══════════════════════════════════════════════════════════════════════════════
def instancia_8(base: pd.DataFrame) -> pd.DataFrame:
    """
    Representa microrrede com polos industriais junto às DGs e baterias.
    Demanda redistribuída: barras com DG/BESS e vizinhas recebem peso 3×;
    total por período é conservado.
    Esperado: fluxos longos diminuem, baterias atendem cargas locais.
    """
    barras_alta = CONFIG['barras_dg_bess'] | CONFIG['barras_vizinhas_dg']
    col_tipo = CONFIG['columns']['tipo']
    col_i = CONFIG['columns']['i']
    barras_all = sorted(base.loc[base[col_tipo] == "DEMANDA", col_i].unique())
    pesos = {int(b): (3.0 if int(b) in barras_alta else 1.0) for b in barras_all}
    return redistribuir_demanda(base, pesos_por_barra=pesos, escala=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 9 — Carga leve residencial (DEMANDA × 0.5)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_9(base: pd.DataFrame) -> pd.DataFrame:
    """
    Regime de sobrepenetração renovável em bairro residencial de baixa carga.
    Perfil temporal e distribuição espacial mantidos; total × 0.5.
    Esperado: baterias enchem cedo, curtailment persiste mesmo com armazenamento.
    """
    col_tipo = CONFIG['columns']['tipo']
    col_i = CONFIG['columns']['i']
    barras_all = sorted(base.loc[base[col_tipo] == "DEMANDA", col_i].unique())
    pesos = {int(b): 1.0 for b in barras_all}   # distribuição original
    return redistribuir_demanda(base, pesos_por_barra=pesos, escala=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 10 — Rede limitada / congestionamento (F_max × 0.4 nas críticas)
# ══════════════════════════════════════════════════════════════════════════════
def instancia_10(base: pd.DataFrame) -> pd.DataFrame:
    """
    Introduz congestionamento realista nas linhas de maior carga.
    F_max reduzido nas linhas críticas conforme LINHAS_CRITICAS_FATOR.
    Linhas não críticas mantêm F_max original.
    Esperado: redespacho de geração, mais uso local de bateria, curtailment local.
    """
    df = base.copy()
    col_tipo = CONFIG['columns']['tipo']
    col_i = CONFIG['columns']['i']
    col_j = CONFIG['columns']['j']
    col_F_max = CONFIG['columns']['F_max']
    col_valor = CONFIG['columns']['valor']

    if (df[col_tipo] == "LINHA").sum() == 0:
        print("  ⚠️  Nenhuma linha encontrada com tipo==LINHA. Verifique o schema.")
        return df

    mask_linha = df[col_tipo] == "LINHA"

    for (vi, vj), fator in CONFIG['linhas_criticas_fator'].items():
        # Tenta localizar pelo par (i, j) — adapte os nomes de coluna se necessário
        cond_i = df[col_i].astype(str) == str(vi)
        cond_j = df[col_j].astype(str) == str(vj) if col_j in df.columns else pd.Series(False, index=df.index)
        mask_par = mask_linha & cond_i & cond_j

        if mask_par.any():
            if col_F_max in df.columns:
                df = _escalar(df, mask_par, col_F_max, fator)
            elif col_valor in df.columns:
                df = _escalar(df, mask_par, col_valor, fator)
        else:
            print(f"  ⚠️  Linha ({vi},{vj}) não encontrada no CSV — ignorada.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Execução principal
# ══════════════════════════════════════════════════════════════════════════════
VARIANTES = {
    1:  (instancia_1,  "Baixa irradiância (SOLAR×0.5)"),
    2:  (instancia_2,  "Baixo vento (EOLICA×0.5)"),
    3:  (instancia_3,  "Bateria muito maior (E_max×3, Pot×2)"),
    4:  (instancia_4,  "Sem bateria"),
    5:  (instancia_5,  "Vertimento barato (c_curt×0.2)"),
    6:  (instancia_6,  "Vertimento muito caro (c_curt×7.5)"),
    7:  (instancia_7,  "Perdas caras (c_loss×3)"),
    8:  (instancia_8,  "Carga industrial próxima às DGs/BESS"),
    9:  (instancia_9,  "Carga leve residencial (Demanda×0.5)"),
    10: (instancia_10, "Rede limitada — congestionamento (F_max×0.4)"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera instâncias CSV a partir de uma base.")
    parser.add_argument('--base_file', type=str, default="dados_microrrede_0.csv", help="Arquivo CSV base")
    parser.add_argument('--out_dir', type=str, default="", help="Diretório de saída")
    args = parser.parse_args()

    base_file = args.base_file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Carregando base: {base_file}")
    base = carregar_base(base_file)
    print(f"  {len(base)} linhas carregadas.\n")

    for idx, (func, descricao) in VARIANTES.items():
        print(f"[Cenário {idx:>2}] {descricao}")
        df_var = func(base)
        salvar(df_var, idx, out_dir)

    print("\nTodas as instâncias geradas com sucesso.")
