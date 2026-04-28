# ============================================================
# MODULARIZAÇÃO: Funções para construção do modelo PuLP
# ============================================================

import pulp

def criar_modelo(nome="Despacho_Economico_Microrrede"):
    """
    Cria um novo problema de otimização linear no PuLP.
    
    Args:
        nome (str): Nome do problema
        
    Returns:
        pulp.LpProblem: Problema de minimização
    """
    return pulp.LpProblem(nome, pulp.LpMinimize)

def criar_variaveis_renovaveis(N, T):
    """
    Cria as variáveis de decisão para geração solar e eólica.
    
    Args:
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        
    Returns:
        tuple: (PS, PW, CS, CW) dicionários de variáveis PuLP
    """
    PS = pulp.LpVariable.dicts("PS", [(i, t) for i in N for t in T], lowBound=0)
    PW = pulp.LpVariable.dicts("PW", [(i, t) for i in N for t in T], lowBound=0)
    CS = pulp.LpVariable.dicts("CS", [(i, t) for i in N for t in T], lowBound=0)
    CW = pulp.LpVariable.dicts("CW", [(i, t) for i in N for t in T], lowBound=0)
    
    return PS, PW, CS, CW

def criar_variaveis_bateria(B, T):
    """
    Cria as variáveis de decisão para operação das baterias.
    
    Args:
        B (set): Conjunto de índices de baterias
        T (range): Horizonte temporal
        
    Returns:
        tuple: (P_ch, P_dis, E) dicionários de variáveis PuLP
    """
    P_ch = pulp.LpVariable.dicts("P_ch", [(i, t) for i in B for t in T], lowBound=0)
    P_dis = pulp.LpVariable.dicts("P_dis", [(i, t) for i in B for t in T], lowBound=0)
    E = pulp.LpVariable.dicts("E", [(i, t) for i in B for t in T])
    
    return P_ch, P_dis, E

def criar_variaveis_fluxo(N, L, T, K):
    """
    Cria as variáveis de decisão para fluxos de potência, ângulos e perdas.
    
    Args:
        N (list): Conjunto de barras
        L (list): Conjunto de linhas
        T (range): Horizonte temporal
        K (range): Conjunto de segmentos da aproximação
        
    Returns:
        tuple: (theta, F, F_pos, F_neg, Ploss, lam)
    """
    theta = pulp.LpVariable.dicts("theta", [(i, t) for i in N for t in T])
    F = pulp.LpVariable.dicts("F", [(l, t) for l in L for t in T])
    F_pos = pulp.LpVariable.dicts("F_pos", [(l, t) for l in L for t in T], lowBound=0)
    F_neg = pulp.LpVariable.dicts("F_neg", [(l, t) for l in L for t in T], lowBound=0)
    Ploss = pulp.LpVariable.dicts("Ploss", [(l, t) for l in L for t in T], lowBound=0)
    lam = pulp.LpVariable.dicts("lam", [(k, l, t) for k in K for l in L for t in T], lowBound=0)
    
    return theta, F, F_pos, F_neg, Ploss, lam

def definir_funcao_objetivo(prob, c_curt, CS, CW, N, T, c_ch, P_ch, c_dis, P_dis, B, c_loss, Ploss, L):
    """
    Define a função objetivo de minimização de custos.
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        c_curt (dict): Custo de curtimento por barra
        CS, CW: Variáveis de curtimento solar e eólico
        N, T: Conjuntos de barras e períodos
        c_ch, c_dis: Custos de carga e descarga de bateria
        P_ch, P_dis: Variáveis de potência de bateria
        B: Conjunto de baterias
        c_loss: Custo de perdas por linha
        Ploss: Variável de perdas
        L: Conjunto de linhas
    """
    prob += (
        pulp.lpSum(
            c_curt[i] * (CS[(i, t)] + CW[(i, t)])
            for i in N for t in T
        )
        +
        pulp.lpSum(
            c_ch[i] * P_ch[(i, t)] + c_dis[i] * P_dis[(i, t)]
            for i in B for t in T
        )
        +
        pulp.lpSum(
            c_loss[l] * Ploss[(l, t)]
            for l in L for t in T
        )
    ), "Custo_Total"

def adicionar_restricoes_renovaveis(prob, PS, CS, PS_avail, PW, CW, PW_avail, T, GS, GW, N):
    """
    Adiciona as restrições de geração solar e eólica (curtimento).
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        PS, CS: Variáveis de geração e curtimento solar
        PS_avail: Disponibilidade solar
        PW, CW: Variáveis de geração e curtimento eólico
        PW_avail: Disponibilidade eólica
        T (range): Horizonte temporal
        GS, GW: Conjuntos de geradores solar e eólico
        N (list): Conjunto de barras
    """
    # Restrições para barras sem geração renovável (PS = PW = 0)
    for t in T:
        for i in N:
            if i not in GS and i not in GW:
                prob += PS[(i, t)] <= 0
                prob += PW[(i, t)] <= 0
                prob += CS[(i, t)] <= 0
                prob += CW[(i, t)] <= 0
            
    
    # Restrições de geração solar
    for t in T:
        for i in GS:
            prob += PS[(i, t)] <= PS_avail[(i, t)]
            prob += CS[(i, t)] <= PS_avail[(i, t)]
            prob += PS[(i, t)] + CS[(i, t)] == PS_avail[(i, t)]
    
    # Restrições de geração eólica
    for t in T:
        for i in GW:
            prob += PW[(i, t)] <= PW_avail[(i, t)]
            prob += CW[(i, t)] <= PW_avail[(i, t)]
            prob += PW[(i, t)] + CW[(i, t)] == PW_avail[(i, t)]

def adicionar_restricoes_bateria(prob, B, T, E, E0, P_ch, P_ch_max, P_dis, P_dis_max, E_min, E_max, eta_ch, eta_dis, Delta_t):
    """
    Adiciona as restrições de operação das baterias.
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        B (set): Conjunto de baterias
        T (range): Horizonte temporal
        E: Variável de estado de carga
        E0: Estado inicial de carga
        P_ch, P_dis: Variáveis de potência de carga e descarga
        P_ch_max, P_dis_max: Limites de potência
        E_min, E_max: Limites de energia
        eta_ch, eta_dis: Eficiências
        Delta_t (float): Intervalo de tempo
    """
    for i in B:
        # Condição inicial
        prob += E[(i, min(T))] == E0[i]
        
        for t in T:
            # Limites de potência
            prob += P_ch[(i, t)] <= P_ch_max[i]
            prob += P_dis[(i, t)] <= P_dis_max[i]
            
            # Limites de energia
            prob += E_min[i] <= E[(i, t)]
            prob += E[(i, t)] <= E_max[i]
            
            # Dinâmica (para t < último período)
            if t < max(T):
                prob += (
                    E[(i, t + 1)]
                    == E[(i, t)]
                    + eta_ch[i] * P_ch[(i, t)] * Delta_t
                    - (1.0 / eta_dis[i]) * P_dis[(i, t)] * Delta_t
                )

def adicionar_restricoes_fluxo(prob, theta, F, F_max, N, T, i_ref, b, L):
    """
    Adiciona as restrições de fluxo de potência e ângulos de tensão.
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        theta: Variável de ângulo de tensão
        F: Variável de fluxo
        F_max: Limite de fluxo por linha
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        i_ref (int): Barra de referência
        b (dict): Susceptância por linha
        L (list): Conjunto de linhas
    """
    for t in T:
        # Referência
        prob += theta[(i_ref, t)] == 0
        
        for (i, j) in L:
            prob += F[((i, j), t)] == b[(i, j)] * (theta[(i, t)] - theta[(j, t)])
            prob += F[((i, j), t)] <= F_max[(i, j)]
            prob += F[((i, j), t)] >= -F_max[(i, j)]

def construir_incidencia(N, L):
    """
    Constrói os conjuntos de linhas incidentes em cada barra.
    
    Args:
        N (list): Conjunto de barras
        L (list): Conjunto de linhas
        
    Returns:
        tuple: (delta_plus, delta_minus) com linhas saindo e chegando
    """
    delta_plus = {i: [] for i in N}
    delta_minus = {i: [] for i in N}
    
    for (u, v) in L:
        delta_plus[u].append((u, v))
        delta_minus[v].append((u, v))
    
    return delta_plus, delta_minus

def adicionar_restricoes_balanço_potencia(prob, N, T, PS, PW, P_dis, P_ch, B, D, F, delta_plus, delta_minus, Ploss):
    """
    Adiciona as restrições de balanço de potência por barra e período.
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        PS, PW: Variáveis de geração renovável
        P_dis, P_ch: Variáveis de bateria
        B (set): Conjunto de baterias
        D (dict): Demanda
        F (dict): Variável de fluxo
        delta_plus, delta_minus: Incidência de linhas
        Ploss: Variável de perdas
    """
    for t in T:
        for i in N:
            inj_ren = PS[(i, t)] + PW[(i, t)]
            inj_batt = 0
            if i in B:
                inj_batt = P_dis[(i, t)] - P_ch[(i, t)]
            
            # Soma fluxos que saem e entram
            sum_out = pulp.lpSum(F[((l_i, l_j), t)] for (l_i, l_j) in delta_plus[i])
            sum_in = pulp.lpSum(F[((l_i, l_j), t)] for (l_i, l_j) in delta_minus[i])
            
            # Perdas nas linhas incidentes
            incident_lines = delta_plus[i] + delta_minus[i]
            sum_loss = 0.5*pulp.lpSum(Ploss[((l_i, l_j), t)] for (l_i, l_j) in incident_lines)      # Ajuste para evitar contagem dupla
            
            prob += (
                inj_ren + inj_batt - D[(i, t)]
                == sum_out - sum_in - sum_loss
            )

# ============================================================
# CONSTRUÇÃO DO MODELO
# ============================================================

# Criar problema de minimização
# prob = criar_modelo()

# Criar variáveis de decisão
#PS, PW, CS, CW = criar_variaveis_renovaveis(N, T)
#P_ch, P_dis, E = criar_variaveis_bateria(B, T)
#theta, F, F_pos, F_neg, Ploss, lam = criar_variaveis_fluxo(N, L, T, K)

# Definir função objetivo
#definir_funcao_objetivo(prob, c_curt, CS, CW, N, T, c_ch, P_ch, c_dis, P_dis, B, c_loss, Ploss, L)

# Adicionar restrições de geração renovável
#adicionar_restricoes_renovaveis(prob, PS, CS, PS_avail, PW, CW, PW_avail, T, GS, GW, N)

# Adicionar restrições de bateria
#adicionar_restricoes_bateria(prob, B, T, E, E0, P_ch, P_ch_max, P_dis, P_dis_max, E_min, E_max, eta_ch, eta_dis, Delta_t)

# Adicionar restrições de fluxo
#adicionar_restricoes_fluxo(prob, theta, F, F_max, N, T, i_ref, b, L)

# Construir incidência
#delta_plus, delta_minus = construir_incidencia(N, L)

# Adicionar restrições de balanço de potência
#adicionar_restricoes_balanço_potencia(prob, N, T, PS, PW, P_dis, P_ch, B, D, F, delta_plus, delta_minus, Ploss)

