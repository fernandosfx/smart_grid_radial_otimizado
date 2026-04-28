# Modelo de Otimização para Despacho Econômico em Microrredes com Fontes Renováveis

## 1. Descrição Geral

Este projeto implementa um **modelo de otimização linear para despacho econômico em uma microrrede radial** com geração de energia de fontes renováveis (solar e eólica), armazenamento em baterias e demanda elétrica distribuída. O objetivo é **minimizar custos operacionais** através da alocação ótima de potência, gestão de baterias e curtimento de geração excedente, levando em conta as restrições de fluxo de potência e perdas Joule nas linhas.

O modelo utiliza a **biblioteca PuLP** em Python com o solver **CBC (Coin-or-Branch-and-Cut)** para resolver o problema de programação linear inteira-mista.

---

## 2. Métodos de Otimização

### 2.1 Formulação Matemática

O modelo é formulado como um **Problema de Programação Linear (PL)**:

```
minimize:  ∑ (custos de curtimento + custos de bateria + custos de perdas)
subject to: Equações de balanço de potência
            Limites de fluxo nas linhas
            Dinâmica de armazenamento em baterias
            Limites operacionais de geração e armazenamento
```

### 2.2 Solver Utilizado

- **Biblioteca**: PuLP (Python Linear Programming)
- **Solver**: CBC (Coin-or-Branch-and-Cut)
- **Tipo de Problema**: Programação Linear (LP)

O PuLP oferece uma interface de alto nível para formulação de problemas de otimização, enquanto o solver CBC é responsável pela resolução eficiente do problema LP através de algoritmos de simplex e branch-and-cut.

### 2.3 Aproximação de Perdas Quadráticas (Piecewise Linear)

As perdas de potência quadráticas nas linhas são aproximadas por uma **combinação convexa linear** para manter a linearidade do modelo:

$$P_{loss}(l,t) = R_l \cdot |F_{l,t}|^2 \approx \sum_{k \in K} p_{k,l} \cdot \lambda_{k,l,t}$$

onde:
- $K$ é o conjunto de segmentos da aproximação
- $p_{k,l}$ são os valores de perdas calculados nos pontos de discretização
- $\lambda_{k,l,t}$ são variáveis de peso que formam uma **combinação convexa (SOS1)**

A decomposição é feita como:
$$|F_{l,t}| = F_{l,t}^+ + F_{l,t}^- \approx \sum_{k} f_{k,l} \cdot \lambda_{k,l,t}$$

com a restrição de convexidade:
$$\sum_{k} \lambda_{k,l,t} = 1 \quad \forall l, t$$

---

## 3. Formatos de Dados

### 3.1 Formato de Entrada (CSV)

O modelo suporta **múltiplas instâncias** de dados de microrrede, localizadas na pasta `conjuntos_instancias/`. Cada arquivo **`dados_microrrede_X.csv`** (onde X é o índice da instância, de 0 a 6) contém todos os dados do modelo em formato tabular. Cada linha representa uma entidade ou parâmetro do sistema.

**Estrutura do CSV:**
```
tipo | i | j | t | ativo | valor | F_max | b | R | c_loss | eta_ch | eta_dis | P_ch_max | P_dis_max | E_min | E_max | E0 | c_ch | c_dis
```

**Tipos de Registros:**

| Tipo | Descrição | Campos Utilizados |
|------|-----------|------------------|
| `BARRA` | Define as barras da rede | `i` |
| `LINHA` | Define as linhas de transmissão | `i, j, F_max, b, R, c_loss` |
| `ATIVO` | Define ativos de geração/armazenamento | `i, ativo` (solar/eolica/bateria) |
| `SOLAR` | Disponibilidade solar por período | `i, t, valor` (em MW) |
| `EOLICA` | Disponibilidade eólica por período | `i, t, valor` (em MW) |
| `DEMANDA` | Demanda de carga por barra e período | `i, t, valor` (em MW) |
| `CUSTO_CURT` | Custo de curtimento por barra | `i, valor` (em R$/MWh) |
| `BATERIA` | Parâmetros de bateria | `i, eta_ch, eta_dis, P_ch_max, P_dis_max, E_min, E_max, E0, c_ch, c_dis` |

**Exemplo de Registros:**
```csv
tipo,i,j,t,ativo,valor,F_max,b,R,c_loss,eta_ch,eta_dis,P_ch_max,P_dis_max,E_min,E_max,E0,c_ch,c_dis
BARRA,1,,,,,,,,,,,,,,,,,
BARRA,2,,,,,,,,,,,,,,,,,
LINHA,1,2,,,,10,1.0,0.01,1.0,,,,,,,,,
SOLAR,1,,1,,4,,,,,,,,,,,,,
DEMANDA,2,,1,,5,,,,,,,,,,,,,
BATERIA,3,,,,,,,,,0.95,0.95,3,3,0,10,5,5,5
CUSTO_CURT,1,,,,100,,,,,,,,,,,,,
```

**Descrição dos Parâmetros Principais:**
- `F_max`: Limite de fluxo na linha (MW)
- `b`: Susceptância da linha (p.u.)
- `R`: Resistência da linha (Ω/km)
- `c_loss`: Custo das perdas (R$/MWh)
- `eta_ch`: Eficiência de carga da bateria (0-1)
- `eta_dis`: Eficiência de descarga da bateria (0-1)
- `P_ch_max`: Potência máxima de carga (MW)
- `P_dis_max`: Potência máxima de descarga (MW)
- `E_min`, `E_max`: Limites de energia armazenada (MWh)
- `E0`: Energia inicial no armazenador (MWh)
- `c_ch`: Custo de carga (R$/MWh)
- `c_dis`: Custo de descarga (R$/MWh)

### 3.2 Formato de Saída

Os resultados são exportados como **DataFrames Pandas** com índices multi-nível `(barra, período)`:

#### Variáveis de Geração Renovável:
- **Geração Solar** (`df_PS`): `[(i,t) → MW]`
- **Geração Eólica** (`df_PW`): `[(i,t) → MW]`
- **Curtimento Solar** (`df_CS`): `[(i,t) → MW]`
- **Curtimento Eólico** (`df_CW`): `[(i,t) → MW]`

#### Variáveis de Armazenamento:
- **Potência de Carga** (`df_P_ch`): `[(i,t) → MW]`
- **Potência de Descarga** (`df_P_dis`): `[(i,t) → MW]`
- **Energia Armazenada** (`df_E`): `[(i,t) → MWh]`

#### Variáveis de Fluxo de Potência:
- **Fluxo nas Linhas** (`df_F`): `[(l,t) → MW]`
- **Ângulos de Fase** (`df_theta`): `[(i,t) → rad]`
- **Perdas nas Linhas** (`df_Ploss`): `[(l,t) → MW]`

**Exemplo de Saída:**
```
         1    2    3
(1, 1)  3.5  2.1  0.8
(1, 2)  3.8  2.4  0.7
...
```

---

## 4. Variáveis de Decisão

| Variável | Descrição | Dimensão | Limites |
|----------|-----------|----------|---------|
| $P_S^{i,t}$ | Geração solar na barra $i$, período $t$ | ∀ $i \in N, t \in T$ | $[0, P_{S,(i,t)}^{avail}]$ |
| $P_W^{i,t}$ | Geração eólica na barra $i$, período $t$ | ∀ $i \in N, t \in T$ | $[0, P_{W,(i,t)}^{avail}]$ |
| $C_S^{i,t}$ | Curtimento solar na barra $i$, período $t$ | ∀ $i \in N, t \in T$ | $[0, \infty)$ |
| $C_W^{i,t}$ | Curtimento eólico na barra $i$, período $t$ | ∀ $i \in N, t \in T$ | $[0, \infty)$ |
| $P_{ch}^{i,t}$ | Potência de carga da bateria $i$, período $t$ | ∀ $i \in B, t \in T$ | $[0, P_{ch,max}^i]$ |
| $P_{dis}^{i,t}$ | Potência de descarga da bateria $i$, período $t$ | ∀ $i \in B, t \in T$ | $[0, P_{dis,max}^i]$ |
| $E^{i,t}$ | Energia armazenada na bateria $i$, período $t$ | ∀ $i \in B, t \in T$ | $[E_{min}^i, E_{max}^i]$ |
| $\theta^{i,t}$ | Ângulo de fase na barra $i$, período $t$ | ∀ $i \in N, t \in T$ | $[-\pi, \pi]$ |
| $F^{l,t}$ | Fluxo de potência na linha $l$, período $t$ | ∀ $l \in L, t \in T$ | $[-F_{max}^l, F_{max}^l]$ |
| $P_{loss}^{l,t}$ | Perdas de potência na linha $l$, período $t$ | ∀ $l \in L, t \in T$ | $[0, \infty)$ |
| $\lambda_{k,l,t}$ | Variável de peso para aproximação de perdas | ∀ $k \in K, l \in L, t \in T$ | $[0, 1]$, SOS1 |

---

## 5. Função Objetivo

A função objetivo minimiza o custo total operacional:

$$\text{minimize} \quad Z = \underbrace{\sum_{i \in N} \sum_{t \in T} c_{curt}^i (C_S^{i,t} + C_W^{i,t})}_{\text{Custo de curtimento}} + \underbrace{\sum_{i \in B} \sum_{t \in T} (c_{ch}^i P_{ch}^{i,t} + c_{dis}^i P_{dis}^{i,t})}_{\text{Custo de bateria}} + \underbrace{\sum_{l \in L} \sum_{t \in T} c_{loss}^l P_{loss}^{l,t}}_{\text{Custo de perdas}}$$

---

## 6. Restrições Principais

### 6.1 Balanço de Potência (Equação de Fluxo DC)

$$P_S^{i,t} + P_W^{i,t} + \sum_{j} F_{(j,i),t}^{AC} - \sum_{j} F_{(i,j),t}^{AC} + P_{dis}^{i,t} = D^{i,t} + P_{ch}^{i,t} + C_S^{i,t} + C_W^{i,t} \quad \forall i \in N, t \in T$$

### 6.2 Fluxo DC nas Linhas

$$F^{l,t} = b^l(\theta^{i,t} - \theta^{j,t}) \quad \forall l = (i,j), t \in T$$

### 6.3 Limite de Fluxo nas Linhas

$$|F^{l,t}| \leq F_{max}^l \quad \forall l \in L, t \in T$$

### 6.4 Disponibilidade de Geração

$$P_S^{i,t} \leq P_{S,(i,t)}^{avail}, \quad P_W^{i,t} \leq P_{W,(i,t)}^{avail} \quad \forall i \in N, t \in T$$

### 6.5 Dinâmica de Armazenamento em Baterias

$$E^{i,t} = E^{i,t-1} + \eta_{ch}^i P_{ch}^{i,t} - \frac{1}{\eta_{dis}^i} P_{dis}^{i,t} \quad \forall i \in B, t \in T$$

$$E_{min}^i \leq E^{i,t} \leq E_{max}^i \quad \forall i \in B, t \in T$$

### 6.6 Complementaridade de Carga e Descarga

$$F^{l,t} = F_{pos}^{l,t} - F_{neg}^{l,t}$$

$$F_{pos}^{l,t} + F_{neg}^{l,t} = \sum_{k \in K} f_{k,l} \lambda_{k,l,t}$$

$$P_{loss}^{l,t} = \sum_{k \in K} p_{k,l} \lambda_{k,l,t}$$

$$\sum_{k \in K} \lambda_{k,l,t} = 1 \quad \forall l \in L, t \in T$$

---

## 7. Estrutura de Módulos

O projeto é organizado em **três módulos principais**:

### 7.1 **A_definicao_parametros.py**
Funções para carregamento e processamento de dados:
- `carregar_dados()`: Lê o arquivo CSV e padroniza dados
- `definir_horizonte()`: Define horizonte temporal (24h)
- `extrair_barras()`, `extrair_linhas()`, `extrair_ativos()`: Extração de conjuntos
- `extrair_demanda()`, `extrair_disponibilidade_renovavel()`: Parâmetros de operação

### 7.2 **B_condicoes_modelo.py**
Construção do modelo de otimização:
- `criar_modelo()`: Cria problema PuLP
- `criar_variaveis_renovaveis()`, `criar_variaveis_bateria()`, `criar_variaveis_fluxo()`: Define variáveis
- `definir_funcao_objetivo()`: Função de custo
- Adição de restrições de balanço, fluxo, capacidade

### 7.3 **C_aproximacao_perdas.py**
Implementação da aproximação piecewise:
- `adicionar_decomposicao_fluxo()`: Separa fluxo em positivo/negativo
- `adicionar_combinacao_convexa()`: Cria variáveis lambda
- `adicionar_aproximacao_perdas()`: Restrições de perdas linearizadas

### 7.4 **main.py**
Orquestração:
- Carrega os três módulos
- Resolve o modelo
- Extrai e exibe resultados

### 7.5 **variaveis_otimizadas/otimo_viz.py**
Módulo de visualização:
- `gerar_figuras()`: Cria gráficos automáticos de operação da microrrede
- Geração de figuras para análise de resultados

### 7.6 **conjuntos_instancias/gerador_instancias.py**
Geração de cenários:
- Cria variações das instâncias base
- Modifica disponibilidade de geração renovável
- Gera os 7 cenários pré-configurados

### 7.7 **relatorio_multiplas_instancias.ipynb**
Notebook Jupyter para análise completa:
- Execução automatizada de múltiplas instâncias
- Geração de relatórios consolidados
- Visualizações comparativas entre cenários

## 8. Como Usar

### 8.1 Preparação de Dados

Editar `dados_microrrede.csv` com:
- Topologia da rede (barras e linhas)
- Parâmetros de geração renovável
- Perfis de demanda por período
- Características das baterias

### 8.2 Execução

#### Opção 1: Otimização de uma Instância Específica (via Python)

```python
from main import executar_otimizacao

# Executar otimização para uma instância específica
resultado = executar_otimizacao(
    caminho_csv="conjuntos_instancias/dados_microrrede_0.csv",
    pasta_saida_figuras="figuras_artigo_0"
)

print(f"Status: {resultado['status']}")
print(f"Custo Total: R$ {resultado['custo_total']:.2f}")
```

#### Opção 2: Análise de Múltiplas Instâncias (via Jupyter Notebook)

Para executar todas as instâncias e gerar relatórios consolidados:

```bash
jupyter notebook relatorio_multiplas_instancias.ipynb
```

Execute as células do notebook em sequência. O notebook irá:
- Carregar todas as instâncias
- Resolver o modelo de otimização para cada uma
- Gerar figuras de visualização automática
- Consolidar resultados em `variaveis_otimizadas/variaveis_otimizadas_consolidado.csv`

### 8.3 Saída Esperada

- **Status de resolução** (ótima, infeasível, unbounded)
- **Valor da função objetivo** (custo total em R$)
- **DataFrames com variáveis de decisão** salvos em CSV na pasta `variaveis_otimizadas/`
- **Gráficos automáticos** salvos nas pastas `figuras_artigo_X/` (onde X é o índice da instância)

---

## 9. Dependências

- **Python 3.7+**
- **pandas**: Manipulação de dados
- **PuLP**: Modelagem de otimização
- **numpy**: Operações numéricas
- **matplotlib**: Geração de gráficos
- **seaborn**: Visualizações estatísticas
- **tqdm**: Barras de progresso
- **jupyter**: Para execução do notebook (opcional)
- **Solver**: CBC (Coin-or-Branch-and-Cut) - geralmente incluído com PuLP

**Instalação:**
```bash
pip install pulp pandas numpy matplotlib seaborn tqdm jupyter
```

**Nota:** Para usar o notebook `relatorio_multiplas_instancias.ipynb`, certifique-se de que o Jupyter está instalado (incluído no comando acima).

---

## 10. Notas Técnicas

- A topologia é **radial** (árvore sem ciclos)
- O modelo utiliza **fluxo DC** (aproximação linear de fluxo AC)
- As perdas quadráticas são aproximadas por **segmentos lineares** para manter a convexidade
- Horizonte de planejamento: **24 períodos** (típico: horários)
- O problema é **determinístico** (sem incerteza nos dados)

---



Desenvolvido para disciplina **PO201** do Mestrado em Pesquisa Operacional no Instituto Tecnológico de Aeronáutica.

## 11. Autores e Referências

**Referências:**
- Hart, W. E., et al. (2017). "Pyomo – Optimization Modeling in Python"
- Mitchell, S. (2007). "PuLP: A Linear Programming Toolkit for Python"
- Carpentier, J. (1962). "Optimal power flows"
