# ============================================================
# MODULARIZAÇÃO DA DEFINIÇÃO DE PARÂMETROS
# ============================================================

def carregar_dados(arquivo_csv):
    """
    Carrega o arquivo CSV e realiza padronização básica.
    
    Args:
        arquivo_csv (str): Caminho do arquivo CSV
        
    Returns:
        pd.DataFrame: DataFrame padronizado
    """
    df = pd.read_csv(arquivo_csv)
    df["tipo"] = df["tipo"].astype(str).str.upper().str.strip()
    return df

def definir_horizonte():
    """
    Define o horizonte de tempo do modelo.
    
    Returns:
        tuple: (T, Delta_t) onde T é o conjunto de períodos e Delta_t é o intervalo
    """
    T = range(1, 25)  # 24 períodos
    Delta_t = 1.0
    return T, Delta_t

def extrair_barras(df, T):
    """
    Extrai o conjunto de barras N do DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame do modelo
        T (range): Horizonte temporal
        
    Returns:
        list: Lista de barras ordenadas
    """
    df_barras = df[df["tipo"] == "BARRA"].copy()
    
    if not df_barras.empty:
        N = sorted(df_barras["i"].dropna().astype(int).unique())
    else:
        # fallback: inferir barras de qualquer coluna i/j existente
        barras_i = df["i"].dropna().astype(int).unique() if "i" in df.columns else []
        barras_j = df["j"].dropna().astype(int).unique() if "j" in df.columns else []
        N = sorted(set(barras_i).union(set(barras_j)))
    
    return N

def extrair_ativos(df):
    """
    Extrai os conjuntos de ativos renováveis e de armazenamento.
    
    Args:
        df (pd.DataFrame): DataFrame do modelo
        
    Returns:
        tuple: (GS, GW, B) onde GS é solar, GW é eólica, B é bateria
    """
    df_ativos = df[df["tipo"] == "ATIVO"].copy()
    df_ativos["ativo"] = df_ativos["ativo"].astype(str).str.lower().str.strip()
    
    GS = set(df_ativos.loc[df_ativos["ativo"] == "solar", "i"].dropna().astype(int).tolist())
    GW = set(df_ativos.loc[df_ativos["ativo"] == "eolica", "i"].dropna().astype(int).tolist())
    B = set(df_ativos.loc[df_ativos["ativo"] == "bateria", "i"].dropna().astype(int).tolist())
    
    return GS, GW, B

def extrair_linhas(df):
    """
    Extrai o conjunto de linhas L e seus parâmetros elétricos.
    
    Args:
        df (pd.DataFrame): DataFrame do modelo
        
    Returns:
        tuple: (L, F_max, b, R, c_loss) com listas e dicionários de parâmetros
    """
    df_linhas = df[df["tipo"] == "LINHA"].copy()
    df_linhas["i"] = df_linhas["i"].astype(int)
    df_linhas["j"] = df_linhas["j"].astype(int)
    
    L = [(row.i, row.j) for row in df_linhas.itertuples(index=False)]
    
    F_max = {(row.i, row.j): float(row.F_max) for row in df_linhas.itertuples(index=False)}
    b = {(row.i, row.j): float(row.b) for row in df_linhas.itertuples(index=False)}
    R = {(row.i, row.j): float(row.R) for row in df_linhas.itertuples(index=False)}
    c_loss = {(row.i, row.j): float(row.c_loss) for row in df_linhas.itertuples(index=False)}
    
    return L, F_max, b, R, c_loss

def extrair_demanda(df, N, T):
    """
    Extrai a demanda por barra e período.
    
    Args:
        df (pd.DataFrame): DataFrame do modelo
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        
    Returns:
        dict: Dicionário com demandas D[(i, t)]
    """
    df_dem = df[df["tipo"] == "DEMANDA"].copy()
    
    D = {(i, t): 0.0 for i in N for t in T}
    for row in df_dem.itertuples(index=False):
        i = int(row.i)
        t = int(row.t)
        if (i in N) and (t in T):
            D[(i, t)] = float(row.valor)
    
    return D

def extrair_disponibilidade_renovavel(df, N, T):
    """
    Extrai a disponibilidade de geração solar e eólica.
    
    Args:
        df (pd.DataFrame): DataFrame do modelo
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        
    Returns:
        tuple: (PS_avail, PW_avail) dicionários de disponibilidade
    """
    df_sol = df[df["tipo"] == "SOLAR"].copy()
    df_eol = df[df["tipo"] == "EOLICA"].copy()
    
    PS_avail = {(i, t): 0.0 for i in N for t in T}
    PW_avail = {(i, t): 0.0 for i in N for t in T}
    
    for row in df_sol.itertuples(index=False):
        i = int(row.i)
        t = int(row.t)
        if (i in N) and (t in T):
            PS_avail[(i, t)] = float(row.valor)
    
    for row in df_eol.itertuples(index=False):
        i = int(row.i)
        t = int(row.t)
        if (i in N) and (t in T):
            PW_avail[(i, t)] = float(row.valor)
    
    return PS_avail, PW_avail

def extrair_custo_curtimento(df, N):
    """
    Extrai o custo de vertimento (curtimento) por barra.
    
    Args:
        df (pd.DataFrame): DataFrame do modelo
        N (list): Conjunto de barras
        
    Returns:
        dict: Dicionário com custos c_curt[i]
    """
    df_curt = df[df["tipo"] == "CUSTO_CURT"].copy()
    
    c_curt = {i: 0.0 for i in N}
    for row in df_curt.itertuples(index=False):
        i = int(row.i)
        if i in N:
            c_curt[i] = float(row.valor)
    
    return c_curt

def extrair_parametros_bateria(df, B):
    """
    Extrai os parâmetros das baterias.
    
    Args:
        df (pd.DataFrame): DataFrame do modelo
        B (set): Conjunto de índices de baterias
        
    Returns:
        dict: Dicionário com todos os parâmetros de bateria
    """
    df_bat = df[df["tipo"] == "BATERIA"].copy()
    
    parametros = {
        "eta_ch": {},
        "eta_dis": {},
        "P_ch_max": {},
        "P_dis_max": {},
        "E_min": {},
        "E_max": {},
        "E0": {},
        "c_ch": {},
        "c_dis": {},
    }
    
    for row in df_bat.itertuples(index=False):
        i = int(row.i)
        if i in B:
            parametros["eta_ch"][i] = float(row.eta_ch)
            parametros["eta_dis"][i] = float(row.eta_dis)
            parametros["P_ch_max"][i] = float(row.P_ch_max)
            parametros["P_dis_max"][i] = float(row.P_dis_max)
            parametros["E_min"][i] = float(row.E_min)
            parametros["E_max"][i] = float(row.E_max)
            parametros["E0"][i] = float(row.E0)
            parametros["c_ch"][i] = float(row.c_ch)
            parametros["c_dis"][i] = float(row.c_dis)
    
    return parametros

def calcular_aproximacao_perdas(L, F_max, R, Kmax=4):
    """
    Calcula a aproximação por partes lineares das perdas nas linhas.
    
    Args:
        L (list): Conjunto de linhas
        F_max (dict): Fluxo máximo por linha
        R (dict): Resistência por linha
        Kmax (int): Número de segmentos da aproximação
        
    Returns:
        tuple: (K, f, p) onde K é o conjunto de segmentos, f são os fluxos e p são as perdas
    """
    K = range(0, Kmax + 1)
    
    f = {}
    p = {}
    
    for ell in L:
        for k in K:
            f[(k, ell)] = k * (F_max[ell] / Kmax)
            p[(k, ell)] = R[ell] * (f[(k, ell)] ** 2)
    
    return K, f, p

def definir_barra_referencia(N):
    """
    Define a barra de referência para o modelo (menor índice).
    
    Args:
        N (list): Conjunto de barras
        
    Returns:
        int: Índice da barra de referência
    """
    return min(N)

# ============================================================
# CARREGAMENTO E INICIALIZAÇÃO DOS DADOS
# ============================================================

# Carregar dados do CSV
arquivo_csv = "./conjuntos_instancias/dados_microrrede.csv"
df = carregar_dados(arquivo_csv)

# Definir horizonte
T, Delta_t = definir_horizonte()

# Extrair conjuntos e parâmetros
N = extrair_barras(df, T)
GS, GW, B = extrair_ativos(df)
L, F_max, b, R, c_loss = extrair_linhas(df)
D = extrair_demanda(df, N, T)
PS_avail, PW_avail = extrair_disponibilidade_renovavel(df, N, T)
c_curt = extrair_custo_curtimento(df, N)

# Extrair parâmetros de bateria
bat_params = extrair_parametros_bateria(df, B)
eta_ch = bat_params["eta_ch"]
eta_dis = bat_params["eta_dis"]
P_ch_max = bat_params["P_ch_max"]
P_dis_max = bat_params["P_dis_max"]
E_min = bat_params["E_min"]
E_max = bat_params["E_max"]
E0 = bat_params["E0"]
c_ch = bat_params["c_ch"]
c_dis = bat_params["c_dis"]

# Aproximação de perdas
K, f, p = calcular_aproximacao_perdas(L, F_max, R, Kmax=4)

# Barra de referência
i_ref = definir_barra_referencia(N)