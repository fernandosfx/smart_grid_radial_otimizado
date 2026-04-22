# ============================================================
# MODULARIZAÇÃO: Funções para aproximação piecewise de perdas
# ============================================================
import pulp

def adicionar_decomposicao_fluxo(prob, L, T, F, F_pos, F_neg):
    """
    Adiciona as restrições de decomposição de fluxo em positivo e negativo.
    
    F = F_pos - F_neg
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        L (list): Conjunto de linhas
        T (range): Horizonte temporal
        F: Variável de fluxo total
        F_pos: Variável de fluxo positivo
        F_neg: Variável de fluxo negativo
    """
    for t in T:
        for l in L:
            prob += F[(l, t)] == F_pos[(l, t)] - F_neg[(l, t)]

def adicionar_combinacao_convexa(prob, L, T, K, F_pos, F_neg, f, lam):
    """
    Adiciona as restrições de combinação convexa para aproximar o módulo do fluxo.
    
    |F| ≈ F_pos + F_neg = ∑_k f[k,l] * lam[k,l,t]
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        L (list): Conjunto de linhas
        T (range): Horizonte temporal
        K (range): Conjunto de segmentos
        F_pos, F_neg: Variáveis de fluxo positivo e negativo
        f (dict): Valores de fluxo nos pontos de aproximação
        lam: Variáveis de peso da combinação convexa
    """
    for t in T:
        for l in L:
            prob += (
                F_pos[(l, t)] + F_neg[(l, t)]
                == pulp.lpSum(f[(k, l)] * lam[(k, l, t)] for k in K)
            )

def adicionar_convexidade_lambdas(prob, L, T, K, lam):
    """
    Adiciona as restrições de convexidade das variáveis lambda (SOS1).
    
    ∑_k lam[k,l,t] = 1  ∀ l, t
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        L (list): Conjunto de linhas
        T (range): Horizonte temporal
        K (range): Conjunto de segmentos
        lam: Variáveis de peso da combinação convexa
    """
    for t in T:
        for l in L:
            prob += pulp.lpSum(lam[(k, l, t)] for k in K) == 1

def adicionar_aproximacao_perdas(prob, L, T, K, Ploss, p, lam):
    """
    Adiciona as restrições de aproximação das perdas por combinação convexa.
    
    Ploss[l,t] ≈ ∑_k p[k,l] * lam[k,l,t]
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        L (list): Conjunto de linhas
        T (range): Horizonte temporal
        K (range): Conjunto de segmentos
        Ploss: Variável de perdas
        p (dict): Valores de perdas nos pontos de aproximação
        lam: Variáveis de peso da combinação convexa
    """
    for t in T:
        for l in L:
            prob += (
                Ploss[(l, t)]
                == pulp.lpSum(p[(k, l)] * lam[(k, l, t)] for k in K)
            )

# ============================================================
# ADICIONAR RESTRIÇÕES DE APROXIMAÇÃO DE PERDAS
# ============================================================

# Decomposição de fluxo
#adicionar_decomposicao_fluxo(prob, L, T, F, F_pos, F_neg)

# Combinação convexa para módulo do fluxo
#adicionar_combinacao_convexa(prob, L, T, K, F_pos, F_neg, f, lam)

# Convexidade das lambdas (SOS1)
#adicionar_convexidade_lambdas(prob, L, T, K, lam)

# Aproximação das perdas
#adicionar_aproximacao_perdas(prob, L, T, K, Ploss, p, lam)