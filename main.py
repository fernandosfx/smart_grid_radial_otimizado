# ============================================================
# ARQUIVO PRINCIPAL - Orquestração da otimização
# ============================================================

import pandas as pd
import pulp
import numpy as np

# Carregar os scripts modularizados
exec(open("A_definicao_parametros.py").read(), globals())
exec(open("B_condicoes_modelo.py").read(), globals())
exec(open("C_aproximacao_perdas.py").read(), globals())


# ============================================================
# FUNÇÕES DE RESOLUÇÃO E ANÁLISE
# ============================================================

def resolver_modelo(prob, solver=None, log_solucao=True):
    """
    Resolve o modelo de otimização.
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        solver (pulp.Solver): Solver a usar (padrão: CBC)
        log_solucao (bool): Se deve exibir mensagens de log
        
    Returns:
        int: Status da solução (1: ótima, 0: infeasível, -1: unbounded)
    """
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=log_solucao)
    
    status = prob.solve(solver)
    return status


def extrair_resultados_geracao(PS, PW, CS, CW, N, T):
    """
    Extrai os resultados de geração solar e eólica.
    
    Args:
        PS, PW, CS, CW: Variáveis de geração e curtimento
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        
    Returns:
        tuple: (df_PS, df_PW, df_CS, df_CW) com os resultados
    """
    # Geração solar
    df_PS = pd.DataFrame({
        (i, t): pulp.value(PS[(i, t)])
        for i in N for t in T
    }).T
    
    # Geração eólica
    df_PW = pd.DataFrame({
        (i, t): pulp.value(PW[(i, t)])
        for i in N for t in T
    }).T
    
    # Curtimento solar
    df_CS = pd.DataFrame({
        (i, t): pulp.value(CS[(i, t)])
        for i in N for t in T
    }).T
    
    # Curtimento eólico
    df_CW = pd.DataFrame({
        (i, t): pulp.value(CW[(i, t)])
        for i in N for t in T
    }).T
    
    return df_PS, df_PW, df_CS, df_CW


def extrair_resultados_bateria(P_ch, P_dis, E, B, T):
    """
    Extrai os resultados de operação das baterias.
    
    Args:
        P_ch, P_dis, E: Variáveis de bateria
        B (set): Conjunto de baterias
        T (range): Horizonte temporal
        
    Returns:
        tuple: (df_P_ch, df_P_dis, df_E) com os resultados
    """
    # Carga de bateria
    df_P_ch = pd.DataFrame({
        (i, t): pulp.value(P_ch[(i, t)])
        for i in B for t in T
    }).T
    
    # Descarga de bateria
    df_P_dis = pd.DataFrame({
        (i, t): pulp.value(P_dis[(i, t)])
        for i in B for t in T
    }).T
    
    # Estado de carga
    df_E = pd.DataFrame({
        (i, t): pulp.value(E[(i, t)])
        for i in B for t in T
    }).T
    
    return df_P_ch, df_P_dis, df_E


def extrair_resultados_fluxo(F, Ploss, L, T):
    """
    Extrai os resultados de fluxo de potência e perdas.
    
    Args:
        F: Variável de fluxo
        Ploss: Variável de perdas
        L (list): Conjunto de linhas
        T (range): Horizonte temporal
        
    Returns:
        tuple: (df_F, df_Ploss) com os resultados
    """
    # Fluxo nas linhas
    df_F = pd.DataFrame({
        (i, j, t): pulp.value(F[((i, j), t)])
        for (i, j) in L for t in T
    }).T
    
    # Perdas nas linhas
    df_Ploss = pd.DataFrame({
        (i, j, t): pulp.value(Ploss[((i, j), t)])
        for (i, j) in L for t in T
    }).T
    
    return df_F, df_Ploss


def extrair_resultados_tensao(theta, N, T):
    """
    Extrai os resultados de ângulo de tensão.
    
    Args:
        theta: Variável de ângulo
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        
    Returns:
        pd.DataFrame: DataFrame com os ângulos de tensão
    """
    df_theta = pd.DataFrame({
        (i, t): pulp.value(theta[(i, t)])
        for i in N for t in T
    }).T
    
    return df_theta


def calcular_custo_total(prob):
    """
    Calcula o custo total da otimização.
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        
    Returns:
        float: Valor da função objetivo
    """
    return pulp.value(prob.objective)


def exibir_resumo_solucao(prob, N, T, B):
    """
    Exibe um resumo da solução encontrada.
    
    Args:
        prob (pulp.LpProblem): Problema de otimização
        N (list): Conjunto de barras
        T (range): Horizonte temporal
        B (set): Conjunto de baterias
    """
    status = prob.status
    status_msg = "Ótima" if status == 1 else ("Infeasível" if status == -1 else "Unbounded")
    
    print("\n" + "="*60)
    print("RESUMO DA SOLUÇÃO")
    print("="*60)
    print(f"Status: {status_msg}")
    print(f"Custo Total: R$ {calcular_custo_total(prob):,.2f}")
    print(f"Número de barras: {len(N)}")
    print(f"Número de baterias: {len(B)}")
    print(f"Períodos de tempo: {len(list(T))}")
    print("="*60 + "\n")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Resolver o modelo
    status = resolver_modelo(prob)
    
    # Exibir resumo
    exibir_resumo_solucao(prob, N, T, B)
    
    # Extrair resultados
    df_PS, df_PW, df_CS, df_CW = extrair_resultados_geracao(PS, PW, CS, CW, N, T)
    df_P_ch, df_P_dis, df_E = extrair_resultados_bateria(P_ch, P_dis, E, B, T)
    df_F, df_Ploss = extrair_resultados_fluxo(F, Ploss, L, T)
    df_theta = extrair_resultados_tensao(theta, N, T)
    
    print("Modelo resolvido com sucesso!")
    print(f"Custo Total: R$ {calcular_custo_total(prob):,.2f}")
