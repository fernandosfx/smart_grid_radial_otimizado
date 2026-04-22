# ============================================================
# ARQUIVO PRINCIPAL - Orquestração da otimização
# ============================================================

import pandas as pd
import pulp
import numpy as np
import os
from pathlib import Path

# Carregar os scripts modularizados
#exec(open("otimizacao/A_definicao_parametros.py").read(), globals())
#exec(open("otimizacao/B_condicoes_modelo.py").read(), globals())
#exec(open("otimizacao/C_aproximacao_perdas.py").read(), globals())


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

def executar_otimizacao(caminho_csv, pasta_saida_figuras="figuras_artigo", caminho_csv_saida=None):
    """
    Executa a otimização completa para um arquivo de microrrede e gera resultados.
    
    Esta função:
    1. Carrega parâmetros do arquivo CSV fornecido
    2. Cria e resolve o modelo de otimização
    3. Salva resultados em CSV
    4. Gera figuras de visualização
    
    Args:
        idx_cenario (int): Índice do cenário a ser otimizado
        caminho_csv (str): Caminho para o arquivo CSV com dados da microrrede
        pasta_saida_figuras (str): Nome da pasta para salvar as figuras (default: "figuras_artigo")
        caminho_csv_saida (str): Caminho completo para salvar o CSV de resultados.
                                 Se None, usa "variaveis_otimizadas/{pasta_saida_figuras}.csv"
    
    Returns:
        dict: Dicionário com informações da execução:
            - "status": status da otimização
            - "arquivo_csv_saida": caminho do arquivo CSV salvo
            - "pasta_figuras": caminho da pasta de figuras
            - "custo_total": valor da função objetivo
    """
    
    print(f"\n{'='*60}")
    print(f"Iniciando otimização para: {"./conjuntos_instancias/dados_microrrede_0.csv"}")
    print(f"{'='*60}\n")
    
    # Passo 1: Carregar parâmetros
    from otimizacao.A_definicao_parametros import definir_parametros_microrrede
    
    print("[1/4] Carregando parâmetros da microrrede...")

    parametros = definir_parametros_microrrede(caminho_csv)
    
    # Desempacotar no escopo global para compatibilidade
    T = parametros["T"]
    Delta_t = parametros["Delta_t"]
    N = parametros["N"]
    GS = parametros["GS"]
    GW = parametros["GW"]
    B = parametros["B"]
    L = parametros["L"]
    F_max = parametros["F_max"]
    b = parametros["b"]
    R = parametros["R"]
    c_loss = parametros["c_loss"]
    D = parametros["D"]
    PS_avail = parametros["PS_avail"]
    PW_avail = parametros["PW_avail"]
    c_curt = parametros["c_curt"]
    eta_ch = parametros["eta_ch"]
    eta_dis = parametros["eta_dis"]
    P_ch_max = parametros["P_ch_max"]
    P_dis_max = parametros["P_dis_max"]
    E_min = parametros["E_min"]
    E_max = parametros["E_max"]
    E0 = parametros["E0"]
    c_ch = parametros["c_ch"]
    c_dis = parametros["c_dis"]
    K = parametros["K"]
    f = parametros["f"]
    p = parametros["p"]
    i_ref = parametros["i_ref"]

    print(f"  ✓ Carregados: {len(N)} barras, {len(B)} baterias, {len(L)} linhas, {len(list(T))} períodos")
    
    # Passo 2: Criar modelo
    from otimizacao.B_condicoes_modelo import (
        criar_modelo,
        criar_variaveis_renovaveis,
        criar_variaveis_bateria,
        criar_variaveis_fluxo,
        definir_funcao_objetivo,
        adicionar_restricoes_renovaveis,
        adicionar_restricoes_bateria,
        adicionar_restricoes_fluxo,
        construir_incidencia,
        adicionar_restricoes_balanço_potencia
    )
    from otimizacao.C_aproximacao_perdas import (
        adicionar_decomposicao_fluxo,
        adicionar_combinacao_convexa,
        adicionar_convexidade_lambdas,
        adicionar_aproximacao_perdas
    )
    print("\n[2/4] Criando e resolvendo modelo...")

    prob = criar_modelo()

    # Criar variáveis de decisão
    PS, PW, CS, CW = criar_variaveis_renovaveis(N, T)
    P_ch, P_dis, E = criar_variaveis_bateria(B, T)
    theta, F, F_pos, F_neg, Ploss, lam = criar_variaveis_fluxo(N, L, T, K)

    # Definir função objetivo
    definir_funcao_objetivo(prob, c_curt, CS, CW, N, T, c_ch, P_ch, c_dis, P_dis, B, c_loss, Ploss, L)

    # Adicionar restrições de geração renovável
    adicionar_restricoes_renovaveis(prob, PS, CS, PS_avail, PW, CW, PW_avail, T, GS, GW)

    # Adicionar restrições de bateria
    adicionar_restricoes_bateria(prob, B, T, E, E0, P_ch, P_ch_max, P_dis, P_dis_max, E_min, E_max, eta_ch, eta_dis, Delta_t)

    # Adicionar restrições de fluxo
    adicionar_restricoes_fluxo(prob, theta, F, F_max, N, T, i_ref, b, L)

    # Construir incidência
    delta_plus, delta_minus = construir_incidencia(N, L)

    # Adicionar restrições de balanço de potência
    adicionar_restricoes_balanço_potencia(prob, N, T, PS, PW, P_dis, P_ch, B, D, F, delta_plus, delta_minus, Ploss)

    # Decomposição de fluxo
    adicionar_decomposicao_fluxo(prob, L, T, F, F_pos, F_neg)

    # Combinação convexa para módulo do fluxo
    adicionar_combinacao_convexa(prob, L, T, K, F_pos, F_neg, f, lam)

    # Convexidade das lambdas (SOS1)
    adicionar_convexidade_lambdas(prob, L, T, K, lam)

    # Aproximação das perdas
    adicionar_aproximacao_perdas(prob, L, T, K, Ploss, p, lam)


    status = resolver_modelo(prob, log_solucao=True)
    
    if status == 1:
        print(f"  ✓ Modelo resolvido com sucesso (status: ótima)")
    else:
        print(f"  ⚠ Modelo resolvido com status: {status}")
    
    # Passo 4: Salvar resultados
    print("\n[3/4] Salvando resultados...")
    
    # Definir caminho de saída do CSV
    if caminho_csv_saida is None:
        os.makedirs("variaveis_otimizadas", exist_ok=True)
        caminho_csv_saida = f"variaveis_otimizadas/variaveis_otimizadas_{pasta_saida_figuras}.csv"
    
    salvar_resultados_csv(prob, PS, PW, CS, CW, P_ch, P_dis, E, F, Ploss, theta, N, T, B, L, caminho_csv_saida)
    print(f"  ✓ Resultados salvos em: {caminho_csv_saida}")
    
    exec(open("variaveis_otimizadas/otimo_viz.py").read(), globals())

    # Passo 5: Gerar figuras
    from variaveis_otimizadas.otimo_viz import gerar_figuras
    print("\n[4/4] Gerando figuras...")
    gerar_figuras(caminho_csv_saida, pasta_saida_figuras)
    
    custo_total = calcular_custo_total(prob)
    print(f"\n{'='*60}")
    print(f"✓ Otimização concluída com sucesso!")
    print(f"  Custo total: {custo_total:.2f}")
    print(f"  Figuras salvas em: figuras_artigo_{pasta_saida_figuras}/")
    print(f"{'='*60}\n")
    
    return {
        "status": status,
        "arquivo_csv_saida": caminho_csv_saida,
        "pasta_figuras": f"figuras_artigo_{pasta_saida_figuras}",
        "custo_total": custo_total
    }


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Executar com parâmetros padrão para compatibilidade
    resultado = executar_otimizacao(
        caminho_csv="./conjuntos_instancias/dados_microrrede_0.csv",
        pasta_saida_figuras="artigo"
    )
    
    print("Modelo resolvido e resultados salvos com sucesso!")
