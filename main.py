# ============================================================
# ARQUIVO PRINCIPAL - Orquestração da otimização
# ============================================================

import pandas as pd
import pulp
import numpy as np

# Carregar os scripts modularizados
exec(open("otimizacao/A_definicao_parametros.py").read(), globals())
exec(open("otimizacao/B_condicoes_modelo.py").read(), globals())
exec(open("otimizacao/C_aproximacao_perdas.py").read(), globals())


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


def salvar_resultados_csv(prob, PS, PW, CS, CW, P_ch, P_dis, E, F, Ploss, theta, N, T, B, L, arquivo_csv="variaveis_otimizadas/variaveis_otimizadas.csv"):
    """
    Salva todos os resultados da otimização em um arquivo CSV.
    
    Args:
        prob: Problema de otimização
        PS, PW, CS, CW: Variáveis de geração renovável
        P_ch, P_dis, E: Variáveis de bateria
        F, Ploss: Variáveis de fluxo e perdas
        theta: Variável de ângulo
        N, T, B, L: Conjuntos
        arquivo_csv: Nome do arquivo de saída
    """
    import pandas as pd
    
    # Lista para armazenar todas as linhas do CSV
    dados = []
    
    # Adicionar informações gerais
    custo_total = calcular_custo_total(prob)
    dados.append({"tempo": "INFO", "entidade": "GERAL", "variavel": "custo_total", "valor": custo_total})
    dados.append({"tempo": "INFO", "entidade": "GERAL", "variavel": "num_barras", "valor": len(N)})
    dados.append({"tempo": "INFO", "entidade": "GERAL", "variavel": "num_baterias", "valor": len(B)})
    dados.append({"tempo": "INFO", "entidade": "GERAL", "variavel": "periodos_tempo", "valor": len(list(T))})
    
    # Geração solar
    for i in N:
        for t in T:
            dados.append({"tempo": t, "entidade": f"barra_{i}", "variavel": "PS", "valor": pulp.value(PS[(i, t)])})
    
    # Geração eólica
    for i in N:
        for t in T:
            dados.append({"tempo": t, "entidade": f"barra_{i}", "variavel": "PW", "valor": pulp.value(PW[(i, t)])})
    
    # Curtailment solar
    for i in N:
        for t in T:
            dados.append({"tempo": t, "entidade": f"barra_{i}", "variavel": "CS", "valor": pulp.value(CS[(i, t)])})
    
    # Curtailment eólico
    for i in N:
        for t in T:
            dados.append({"tempo": t, "entidade": f"barra_{i}", "variavel": "CW", "valor": pulp.value(CW[(i, t)])})
    
    # Carga de bateria
    for i in B:
        for t in T:
            dados.append({"tempo": t, "entidade": f"bateria_{i}", "variavel": "P_ch", "valor": pulp.value(P_ch[(i, t)])})
    
    # Descarga de bateria
    for i in B:
        for t in T:
            dados.append({"tempo": t, "entidade": f"bateria_{i}", "variavel": "P_dis", "valor": pulp.value(P_dis[(i, t)])})
    
    # Estado de carga da bateria
    for i in B:
        for t in T:
            dados.append({"tempo": t, "entidade": f"bateria_{i}", "variavel": "E", "valor": pulp.value(E[(i, t)])})
    
    # Fluxo nas linhas
    for (i, j) in L:
        for t in T:
            dados.append({"tempo": t, "entidade": f"linha_{i}-{j}", "variavel": "F", "valor": pulp.value(F[((i, j), t)])})
    
    # Perdas nas linhas
    for (i, j) in L:
        for t in T:
            dados.append({"tempo": t, "entidade": f"linha_{i}-{j}", "variavel": "Ploss", "valor": pulp.value(Ploss[((i, j), t)])})
    
    # Ângulos de tensão
    for i in N:
        for t in T:
            dados.append({"tempo": t, "entidade": f"barra_{i}", "variavel": "theta", "valor": pulp.value(theta[(i, t)])})
    
    # Criar DataFrame e salvar
    df_resultados = pd.DataFrame(dados)
    df_resultados.to_csv(arquivo_csv, index=False)
    print(f"Resultados salvos em {arquivo_csv}")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Resolver o modelo
    status = resolver_modelo(prob)
    
    # Salvar resultados em CSV
    salvar_resultados_csv(prob, PS, PW, CS, CW, P_ch, P_dis, E, F, Ploss, theta, N, T, B, L)
    
    exec(open("variaveis_otimizadas/otimo_viz.py").read(), globals())

    print("Modelo resolvido e resultados salvos com sucesso!")
