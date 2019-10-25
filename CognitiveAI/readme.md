# Qualidade dos Vinhos Portugueses
## Python

---
> **Autor:** Matheus Willian Machado  
> **Data:** Out 10, 2019
---

## Visão Geral

> O presente problema se refere aos dados de vinhos portugueses "Vinho Verde", que possuem variantes de vinho branco e tinto. Devido a questões de privacidade, apenas variáveis físico-químicas (input) e sensoriais (output) estão disponíveis (por exemplo, não há dados sobre tipo de uva, marca do vinho, preço de venda, etc).
>
> (CognitveAI)

---

## Objetivo

> [Descrição dos dados](https://drive.google.com/open?id=1-oG5-kBt9xQ3Li4PEexpiA9_7RZhRM1f "dataset").
>
> Criar um modelo para estimar a qualidade do vinho. Informação sobre os atributos.
>
> Variáveis input (baseado em testes físico-químicos):
> 1. **Type:** Vinho Tinto (_Red_) e Vinho Branco (_White_)
> 1. **Fixed acidity:** Maioria dos ácidos envolvidos no vinho ou fixa ou não volátil (não evapora rapidamente).
> 1. **Volatile acidity:** Quantidade de ácido acético no vinho, que em níveis muito altos pode levar a um sabor desagradável de vinagre.
> 1. **Citric acid:** Encontrado em pequenas quantidades, o ácido cítrico pode adicionar 'frescura' e sabor aos vinhos.
> 1. **Residual sugar:** Quantidade de açúcar restante após a fermentação é interrompida, é raro encontrar vinhos com menos de 1 grama / litro e vinhos com mais de 45 gramas / litro são considerados doces.
> 1. **Chlorides:** Quantidade de sal no vinho.
> 1. **Free sulfur dioxide:** Forma livre de SO2 existe em equilíbrio entre o SO2 molecular (como um gás dissolvido) e o íon bissulfito; impede o crescimento microbiano e a oxidação do vinho.
> 1. **Total sulfur dioxide:** Quantidade de formas livres e ligadas de S02; em baixas concentrações, o SO2 é principalmente indetectável no vinho, mas em concentrações livres de SO2 acima de 50 ppm, o SO2 se torna evidente no nariz e no sabor do vinho.
> 1. **Density:** Densidade do líquido é próxima à da água, dependendo da porcentagem de álcool e açúcar.
> 1. **pH:** Descreve como um vinho é ácido ou básico em uma escala de 0 (muito ácido) a 14 (muito básico); a maioria dos vinhos tem entre 3-4 na escala de pH
> 1. **Sulphates:** aditivo para vinho que pode contribuir para os níveis de gás dióxido de enxofre (S02), que atua como antimicrobiano e antioxidante.
> 1. **Alcohol:** Percentual de teor alcoólico do vinho.
>
> Variável output (baseado em dado sensorial)
> 1. **Quality:** pontuação entre 0 e 10.
>
> (CognitiveAI e [Udacity](https://s3.amazonaws.com/udacity-hosted-downloads/ud651/wineQualityInfo.txt "Udacity"))

---

## Introdução

![vinhos](_img/CognitiveAI_12_0.png)

Os vinhos verdes recebem esse nome não por sua coloração, que por sinal não é verde, mas sim por serem produzidos exclusivamente em uma região específica no noroeste de Portugal, conhecida como "Os Jardins de Portugal".
Alguns enófilos apontam que esse vinho recebe esse nome por causa da sua alta acidez, remetendo ao perfil ácido das uvas produzidas na região.

Os vinhos verdes podem ser brancos, rosados, tintos e espumantes. Aqui avaliaremos dois tipos: os brancos (_White_) e os tintos (_Red_). Tentaremos classificar um vinho quanto a sua qualidade baseado nos atributos dados.

Fonte: [blog](https://blog.famigliavalduga.com.br/afinal-o-que-e-vinho-verde/ "Vinho Verde")

---

## Análise Exploratória

### Bibliotecas


```python
import time # Tarefas relacionadas a tempo

import numpy as np # Operações com vetores e matrizes
import pandas as pd # Manipulação e análise de dados
import seaborn as sns # Visualização de dados
import matplotlib.pyplot as plt # Visualização de dados

from sklearn.preprocessing import StandardScaler
  # Normalizador de valores baseado em desvio padrão 
from sklearn.feature_selection import SelectKBest, f_classif
  # Seleção de features e métrica ANOVA F-value

# Modelos de Classificação
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Ferramentas Auxiliares
from sklearn.pipeline import Pipeline # Agrupador de partes do modelo
from sklearn.metrics import confusion_matrix, classification_report
  # Funções de métricas para avaliação do modelo
from sklearn.model_selection import GridSearchCV
  # Hiperparametrização e Validação Cruzada
from sklearn.model_selection import train_test_split, StratifiedKFold
  # Divisor de dataset

# Ignorar avisos
import warnings
warnings.filterwarnings("ignore")
```

### Funções


```python
def KBestTable(sel, df, features):
  '''Função para ranquear os atributos de um dataset.

  Args:
    sel(object): Objeto SelectKBest parametrizado.
    df(Dataframe): Dataframe com as colunas a serem ranqueadas.
    features(list): lista de variáveis a que participarão do ranqueamento.
    
  Returns:
      Series: Série com os campos e suas pontuações.
  '''
  names = df[features].columns.values[sel.get_support()]
  scores = pd.Series(sel.scores_, names).sort_values(ascending=False)
  return scores

help(KBestTable)
```

    Help on function KBestTable in module __main__:
    
    KBestTable(sel, df, features)
        Função para ranquear os atributos de um dataset.
        
        Args:
          sel(object): Objeto SelectKBest parametrizado.
          df(Dataframe): Dataframe com as colunas a serem ranqueadas.
          features(list): lista de variáveis a que participarão do ranqueamento.
          
        Returns:
            Series: Série com os campos e suas pontuações.
    



```python
def clfs_train_test(clf, param, x_train, x_test, y_train, y_test):
    ''' Função para treinamento e teste de modelos com hiperparametrização.
    Args:
      clf(Object): Classificadores
      prm(Dict): Parâmetros para tuning
      x_train(Dataframe): conjunto de dados para treino
      x_test(Dataframe): conjunto de dados para teste
      Y_train(Series): conjunto de rótulos para treino
      Y_test(Series): conjunto de rótulos para teste
    
    Returns:
      Dict: Dicionário com as informações do melhor modelo.
      Object: Melhor modelo.
    '''

    skb = {'sel__k': ['all'] + list(range(3,len(x.columns),3))}
    param.update(skb)
  
    cv = StratifiedKFold(n_splits=5, random_state=seed)
    pipe = Pipeline(steps=[('sel', SelectKBest()), ('clf', clf)]) 
      # Montagem do pipeline com suas etapas
    
    comeco = time.time() #Marcação da hora de início dos treinos
    model = GridSearchCV(pipe, param_grid=param, cv=cv, scoring='f1_weighted') 
      # GridSearch para treino e teste das combinações de parâmetros
    model.fit(x_train, y_train) # Treino
    fim = time.time() # Marcação da hora de fim dos treinos
    tempo_treino= fim-comeco # Tempo em segundos para execução do treino
                     
    best_parameters = model.best_params_
      # Melhores parâmetros para o respectivo classificador e conjunto de dados
    best_score = model.best_score_ # Maior score (f1 score) obtido
    best_model = model.best_estimator_ # Otimização do melhor modelo

    comeco = time.time() # Marcação da hora de início dos testes               
    pred = best_model.predict(x_test) # Teste
    fim = time.time() # Marcação da hora de fim dos treinos
    tempo_teste = fim-comeco # Tempo em segundos para execução do teste
    
    # Print dos principais resultados da função                 
    print(clf.__class__.__name__) # Nome do classificador
    print("--------------------\n")
    print("Tempo de treino: {}".format(tempo_treino))
    print('A melhor combinação de parâmetros:')
    print(best_parameters)
    print("Maior F1-score: {}".format(best_score))
    print("F1-score de Teste: {}".format(model.score(x_test, y_test)))
    print('Reporte de classificação:')
    print(classification_report(y_test, pred))
    print("\n\n")
    
    #Lista com os principais resultados
    resultado = {'nome':clf.__class__.__name__,
                 'tempo_treino':tempo_treino,
                 'best_score': best_score,
                 'score_treino':model.score(x_test, y_test)
                }
    
    return resultado, best_model

help(clfs_train_test)
```

    Help on function clfs_train_test in module __main__:
    
    clfs_train_test(clf, param, x_train, x_test, y_train, y_test)
        Função para treinamento e teste de modelos com hiperparametrização.
        Args:
          clf(Object): Classificadores
          prm(Dict): Parâmetros para tuning
          x_train(Dataframe): conjunto de dados para treino
          x_test(Dataframe): conjunto de dados para teste
          Y_train(Series): conjunto de rótulos para treino
          Y_test(Series): conjunto de rótulos para teste
        
        Returns:
          Dict: Dicionário com as informações do melhor modelo.
          Object: Melhor modelo.
    


## Dataset


```python
df = pd.read_csv('/content/sample_data/winequality.csv', sep=";")
  # Transformando csv para dataset
df.head() # Primeiras linhas do dataset
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>White</td>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>White</td>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>White</td>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>White</td>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>White</td>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.info() # Informações do conjunto de dados
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6497 entries, 0 to 6496
    Data columns (total 13 columns):
    type                    6497 non-null object
    fixed acidity           6497 non-null float64
    volatile acidity        6497 non-null float64
    citric acid             6497 non-null float64
    residual sugar          6497 non-null float64
    chlorides               6497 non-null float64
    free sulfur dioxide     6497 non-null float64
    total sulfur dioxide    6497 non-null float64
    density                 6497 non-null float64
    pH                      6497 non-null float64
    sulphates               6497 non-null float64
    alcohol                 6497 non-null object
    quality                 6497 non-null int64
    dtypes: float64(10), int64(1), object(2)
    memory usage: 660.0+ KB


Primeiro, vamos dar uma olhada no conjunto de dados (_dataset_). Vemos que o mesmo possui aproximadamente 6500 amostras de vinhos, com 13 características descritas acima. Sendo 10 delas números com decimais (float), 2 campos contendo texto (object) e 1 contendo apenas números inteiros (int64). Aparentemente, não há valores faltantes e, devido ao baixo consumo de memória, não será realizado otimização de tipo de dados para esse dataset.


```python
df.describe(include='all') # Descrição das colunas
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6497</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497</td>
      <td>6497.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>112</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>White</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4898</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>367</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>7.215307</td>
      <td>0.339666</td>
      <td>0.318633</td>
      <td>5.443235</td>
      <td>0.056034</td>
      <td>30.525319</td>
      <td>115.744574</td>
      <td>1.710882</td>
      <td>3.218501</td>
      <td>0.531268</td>
      <td>NaN</td>
      <td>5.818378</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>1.296434</td>
      <td>0.164636</td>
      <td>0.145318</td>
      <td>4.757804</td>
      <td>0.035034</td>
      <td>17.749400</td>
      <td>56.521855</td>
      <td>7.636088</td>
      <td>0.160787</td>
      <td>0.148806</td>
      <td>NaN</td>
      <td>0.873255</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.987110</td>
      <td>2.720000</td>
      <td>0.220000</td>
      <td>NaN</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>6.400000</td>
      <td>0.230000</td>
      <td>0.250000</td>
      <td>1.800000</td>
      <td>0.038000</td>
      <td>17.000000</td>
      <td>77.000000</td>
      <td>0.992340</td>
      <td>3.110000</td>
      <td>0.430000</td>
      <td>NaN</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>7.000000</td>
      <td>0.290000</td>
      <td>0.310000</td>
      <td>3.000000</td>
      <td>0.047000</td>
      <td>29.000000</td>
      <td>118.000000</td>
      <td>0.994890</td>
      <td>3.210000</td>
      <td>0.510000</td>
      <td>NaN</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>7.700000</td>
      <td>0.400000</td>
      <td>0.390000</td>
      <td>8.100000</td>
      <td>0.065000</td>
      <td>41.000000</td>
      <td>156.000000</td>
      <td>0.996990</td>
      <td>3.320000</td>
      <td>0.600000</td>
      <td>NaN</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.660000</td>
      <td>65.800000</td>
      <td>0.611000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>103.898000</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
# df.alcohol = df.alcohol.astype('float64') # Erro de tipo
alcohol_fix = lambda x: x if x.count('.') < 2 else '.'.join(x.split('.')[:2])
  # Função para ajustar o erro (e.g. '128.933.333.333.333' -> 128.933)
df.alcohol = df.alcohol.apply(alcohol_fix) # Aplicação do ajuste
df.alcohol = df.alcohol.astype('float64') # Transformando alcohol para float
df.type = df.type.astype('category').cat.codes
  # Transformando type para categoria (e.g. White -> 1)
df[['type', 'alcohol']].describe(include='all') # Checando transformações
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6497.000000</td>
      <td>6497.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.753886</td>
      <td>12.157179</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.430779</td>
      <td>33.946284</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>9.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>10.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>11.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>973.333000</td>
    </tr>
  </tbody>
</table>
</div>



Vemos que a coluna "type" possui apenas dois valores _White_ e _Red_, como esperado. Sendo assim, transformaremos essa coluna para booleano utilizando o tipo _category_ para uma maior liberdade. Nota-se também que a grande maioria das amostras é do tipo _White_ (>75%).  
Para os campos numéricos foram apresentadas as médias (mean), desvio padrão (std), mínimos, máximos e os quartis.  
O campo "alcohol" trata-se do teor alcoólico da amostra e, portanto, espera-se que esse campo seja númerico. Durante a transformação, foram observados valores que impediam o processo (e.g. '128.933.333.333.333'). Foi criada uma função para ajuste desses valores e na sequência o campo foi transformado para float64.


```python
print('Porcentagem de valores com teor alcoólico acima de 100%: {}%'.\
      format(round(len(df[df.alcohol > 100].index)/len(df)*100, 2)))
  # Porcentagem de valores de teor alcoólico acima de 100%
```

    Porcentagem de valores com teor alcoólico acima de 100%: 0.62%



```python
df.drop(df[df.alcohol > 100].index, inplace=True) # Remoção dos valores acima
```


```python
print('Porcentagem de valores de densidade acima de 10: {}%'.\
      format(round(len(df[df.density > 10].index)/len(df)*100, 2)))
  # Porcentagem de valores de densidade acima de 10
```

    Porcentagem de valores de densidade acima de 10: 2.11%



```python
df.drop(df[df.density > 10].index, inplace=True) # Remoção dos valores acima
```

Observando o detalhamento das colunas "alcohol" e "density" nota-se que existem valores elevados o suficiente para não fazerem sentido de acordo com a descrição do campo. Porcentagens alcoólicas acima de 100% (0,62% dos casos) e valores de densidades muito alto (2,09% dos casos) foram considerados inconsistentes. Assim, os valores elencados serão removidos do dataset para não influenciarem negativamente no modelo.


```python
any(df.duplicated()) # Busca por duplicados
```




    True




```python
df[df.duplicated(keep=False)]\
  .sort_values(['type','fixed acidity','volatile acidity'])\
  .head(10) # Exemplos de registros duplicados
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5040</th>
      <td>0</td>
      <td>5.2</td>
      <td>0.34</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.050</td>
      <td>27.0</td>
      <td>63.0</td>
      <td>0.99160</td>
      <td>3.68</td>
      <td>0.79</td>
      <td>14.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5042</th>
      <td>0</td>
      <td>5.2</td>
      <td>0.34</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.050</td>
      <td>27.0</td>
      <td>63.0</td>
      <td>0.99160</td>
      <td>3.68</td>
      <td>0.79</td>
      <td>14.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5029</th>
      <td>0</td>
      <td>5.6</td>
      <td>0.50</td>
      <td>0.09</td>
      <td>2.3</td>
      <td>0.049</td>
      <td>17.0</td>
      <td>99.0</td>
      <td>0.99370</td>
      <td>3.63</td>
      <td>0.63</td>
      <td>13.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5030</th>
      <td>0</td>
      <td>5.6</td>
      <td>0.50</td>
      <td>0.09</td>
      <td>2.3</td>
      <td>0.049</td>
      <td>17.0</td>
      <td>99.0</td>
      <td>0.99370</td>
      <td>3.63</td>
      <td>0.63</td>
      <td>13.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6386</th>
      <td>0</td>
      <td>5.6</td>
      <td>0.54</td>
      <td>0.04</td>
      <td>1.7</td>
      <td>0.049</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>0.99420</td>
      <td>3.72</td>
      <td>0.58</td>
      <td>11.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6389</th>
      <td>0</td>
      <td>5.6</td>
      <td>0.54</td>
      <td>0.04</td>
      <td>1.7</td>
      <td>0.049</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>0.99420</td>
      <td>3.72</td>
      <td>0.58</td>
      <td>11.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5894</th>
      <td>0</td>
      <td>5.6</td>
      <td>0.66</td>
      <td>0.00</td>
      <td>2.2</td>
      <td>0.087</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.99378</td>
      <td>3.71</td>
      <td>0.63</td>
      <td>12.8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5895</th>
      <td>0</td>
      <td>5.6</td>
      <td>0.66</td>
      <td>0.00</td>
      <td>2.2</td>
      <td>0.087</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.99378</td>
      <td>3.71</td>
      <td>0.63</td>
      <td>12.8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5727</th>
      <td>0</td>
      <td>5.9</td>
      <td>0.61</td>
      <td>0.08</td>
      <td>2.1</td>
      <td>0.071</td>
      <td>16.0</td>
      <td>24.0</td>
      <td>0.99376</td>
      <td>3.56</td>
      <td>0.77</td>
      <td>11.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5729</th>
      <td>0</td>
      <td>5.9</td>
      <td>0.61</td>
      <td>0.08</td>
      <td>2.1</td>
      <td>0.071</td>
      <td>16.0</td>
      <td>24.0</td>
      <td>0.99376</td>
      <td>3.56</td>
      <td>0.77</td>
      <td>11.1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop_duplicates(inplace=True) # Remoção de duplicados
```

Registros duplicados foram analisados e removidos.


```python
scaler = StandardScaler() # Normalizador de valores
numerics = df.drop(['type', 'quality'], axis=1)
  # Novo dataset apenas com colunas numéricas
numerics[numerics.columns] = scaler.fit_transform(numerics)
  # Normalização do novo dataset
numerics.boxplot(figsize=(10,10), vert=False) #Gráficos boxplot para cada coluna
plt.show()
```


![png](_img/CognitiveAI_24_0.png)



```python
numerics[numerics.chlorides > 9] # Outliers em chlorides
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4979</th>
      <td>0.508822</td>
      <td>0.515943</td>
      <td>2.633161</td>
      <td>-0.709130</td>
      <td>11.014813</td>
      <td>-0.450610</td>
      <td>-0.833588</td>
      <td>1.072417</td>
      <td>-0.599170</td>
      <td>5.027200</td>
      <td>-0.984648</td>
    </tr>
    <tr>
      <th>4981</th>
      <td>0.104493</td>
      <td>1.944232</td>
      <td>-0.384591</td>
      <td>-0.732592</td>
      <td>9.312241</td>
      <td>-0.788624</td>
      <td>-1.118196</td>
      <td>0.892863</td>
      <td>-0.412007</td>
      <td>4.087202</td>
      <td>-0.984648</td>
    </tr>
    <tr>
      <th>5004</th>
      <td>0.508822</td>
      <td>0.396919</td>
      <td>2.495991</td>
      <td>-0.756054</td>
      <td>11.095888</td>
      <td>-0.675953</td>
      <td>-0.798012</td>
      <td>1.036506</td>
      <td>-0.911107</td>
      <td>5.228629</td>
      <td>-1.069229</td>
    </tr>
    <tr>
      <th>5049</th>
      <td>1.640945</td>
      <td>1.051551</td>
      <td>4.690719</td>
      <td>-0.357200</td>
      <td>14.960456</td>
      <td>0.112748</td>
      <td>-0.798012</td>
      <td>1.862451</td>
      <td>-3.032281</td>
      <td>9.861477</td>
      <td>-0.984648</td>
    </tr>
    <tr>
      <th>5156</th>
      <td>0.427956</td>
      <td>0.396919</td>
      <td>3.044673</td>
      <td>-0.732592</td>
      <td>14.987481</td>
      <td>-1.239310</td>
      <td>-1.224923</td>
      <td>0.856953</td>
      <td>-1.035882</td>
      <td>4.892915</td>
      <td>-0.984648</td>
    </tr>
    <tr>
      <th>5349</th>
      <td>0.994018</td>
      <td>0.158871</td>
      <td>1.467212</td>
      <td>-0.732592</td>
      <td>9.636540</td>
      <td>-1.182974</td>
      <td>-1.562895</td>
      <td>1.251970</td>
      <td>-1.035882</td>
      <td>3.550060</td>
      <td>-1.238390</td>
    </tr>
    <tr>
      <th>5590</th>
      <td>1.155749</td>
      <td>0.873015</td>
      <td>1.330041</td>
      <td>-0.685668</td>
      <td>9.879765</td>
      <td>-0.788624</td>
      <td>-0.922528</td>
      <td>1.251970</td>
      <td>-1.223045</td>
      <td>4.288630</td>
      <td>-1.322970</td>
    </tr>
    <tr>
      <th>5652</th>
      <td>0.508822</td>
      <td>0.813503</td>
      <td>2.495991</td>
      <td>-0.756054</td>
      <td>9.690590</td>
      <td>-0.901296</td>
      <td>-1.456167</td>
      <td>0.770767</td>
      <td>-0.848720</td>
      <td>3.550060</td>
      <td>-1.238390</td>
    </tr>
    <tr>
      <th>5949</th>
      <td>1.074883</td>
      <td>0.694479</td>
      <td>1.878723</td>
      <td>-0.826440</td>
      <td>9.663565</td>
      <td>-0.788624</td>
      <td>-1.224923</td>
      <td>0.935956</td>
      <td>-1.223045</td>
      <td>5.430057</td>
      <td>-1.153809</td>
    </tr>
    <tr>
      <th>6158</th>
      <td>1.155749</td>
      <td>1.735940</td>
      <td>2.495991</td>
      <td>-0.732592</td>
      <td>9.366291</td>
      <td>-0.619617</td>
      <td>-1.029256</td>
      <td>0.684582</td>
      <td>-1.285432</td>
      <td>4.154345</td>
      <td>-1.069229</td>
    </tr>
    <tr>
      <th>6217</th>
      <td>1.560079</td>
      <td>2.479840</td>
      <td>2.495991</td>
      <td>-0.756054</td>
      <td>9.663565</td>
      <td>-0.675953</td>
      <td>-0.886952</td>
      <td>0.756403</td>
      <td>-2.034082</td>
      <td>5.362914</td>
      <td>-1.238390</td>
    </tr>
    <tr>
      <th>6268</th>
      <td>1.236615</td>
      <td>2.598864</td>
      <td>1.330041</td>
      <td>-0.756054</td>
      <td>9.690590</td>
      <td>-1.013967</td>
      <td>-0.851376</td>
      <td>0.652262</td>
      <td>-1.410207</td>
      <td>4.288630</td>
      <td>-1.153809</td>
    </tr>
  </tbody>
</table>
</div>

```python
numerics[numerics.sulphates > 9] # Outliers em chlorides
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4984</th>
      <td>1.155749</td>
      <td>0.873015</td>
      <td>-0.247420</td>
      <td>-0.709130</td>
      <td>1.447979</td>
      <td>-0.563281</td>
      <td>0.393782</td>
      <td>1.000595</td>
      <td>-1.846919</td>
      <td>9.525763</td>
      <td>-0.561746</td>
    </tr>
    <tr>
      <th>4990</th>
      <td>1.155749</td>
      <td>0.873015</td>
      <td>-0.178835</td>
      <td>-0.685668</td>
      <td>1.447979</td>
      <td>-0.619617</td>
      <td>0.340418</td>
      <td>1.000595</td>
      <td>-1.846919</td>
      <td>9.727191</td>
      <td>-0.646326</td>
    </tr>
    <tr>
      <th>5049</th>
      <td>1.640945</td>
      <td>1.051551</td>
      <td>4.690719</td>
      <td>-0.357200</td>
      <td>14.960456</td>
      <td>0.112748</td>
      <td>-0.798012</td>
      <td>1.862451</td>
      <td>-3.032281</td>
      <td>9.861477</td>
      <td>-0.984648</td>
    </tr>
  </tbody>
</table>
</div>

```python
numerics[numerics['citric acid'] > 9] # Outlier em free sulfur dioxide
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>745</th>
      <td>0.185359</td>
      <td>-0.852834</td>
      <td>9.217347</td>
      <td>-0.662206</td>
      <td>-0.930217</td>
      <td>0.225419</td>
      <td>-0.015341</td>
      <td>-0.992446</td>
      <td>0.211867</td>
      <td>0.125781</td>
      <td>1.383607</td>
    </tr>
  </tbody>
</table>
</div>

```python
numerics[numerics['free sulfur dioxide'] > 9] # Outlier em free sulfur dioxide
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4745</th>
      <td>-0.865898</td>
      <td>-0.495761</td>
      <td>-0.453176</td>
      <td>-0.47451</td>
      <td>-0.254593</td>
      <td>14.591032</td>
      <td>5.801326</td>
      <td>-0.457377</td>
      <td>1.334842</td>
      <td>0.730066</td>
      <td>-0.054262</td>
    </tr>
  </tbody>
</table>
</div>

```python
outliers = [745,4745,4984,4979,4981,4990,5004,5049,
            5156,5349,5590,5652,5949,6158,6217,6268] # Outliers identificados
df.drop(outliers, inplace=True) # Remoção dos outliers
print("{} Outliers removidos!". format(len(outliers)))
```

    16 Outliers removidos!


De acordo com o site [r-statistics](https://www.r-statistics.com/), um _outlier_ é uma observação numericamente distante do restante dos dados. É possível identificá-los como bolinhas pretas no gráfico boxplot, localizados fora das cercas, "bigodes". [Saiba mais](https://www.r-statistics.com/2011/01/how-to-label-all-the-outliers-in-a-boxplot/).  
Devido a grande quantidade de outliers identificados, de acordo com a técnica boxplot, serão considerados apenas aqueles que estiverem acima de 9 variações. Estes pontos serão removidos para não impactarem negativamente no modelo.


```python
df.info() # Informações após as trasnformações
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5171 entries, 0 to 6496
    Data columns (total 13 columns):
    type                    5171 non-null int8
    fixed acidity           5171 non-null float64
    volatile acidity        5171 non-null float64
    citric acid             5171 non-null float64
    residual sugar          5171 non-null float64
    chlorides               5171 non-null float64
    free sulfur dioxide     5171 non-null float64
    total sulfur dioxide    5171 non-null float64
    density                 5171 non-null float64
    pH                      5171 non-null float64
    sulphates               5171 non-null float64
    alcohol                 5171 non-null float64
    quality                 5171 non-null int64
    dtypes: float64(11), int64(1), int8(1)
    memory usage: 530.2 KB


Após várias transformações, o dataset final ficou com 5171 amostras e todos os 13 campos numéricos, sendo eles inteiros ou flutuantes.

### Análise Univariada


```python
sns.set() # Setando estilo do seaborn para os gráficos
df.hist(figsize=(10,10)) # Gráficos de Histogramas para cada coluna
plt.show()
```


![png](_img/CognitiveAI_34_0.png)


Acima vemos a distribuição de todas as colunas. Observamos que na coluna "type" a quantidade de amostras para o vinho branco é quase 4x maior em comparação com o vinho vermelho, mostrando o desbalanceamento entre as classes. As colunas "density", "alcohol" e "pH" estão bem distribuídas. Vemos [caudas longas](https://pt.m.wikipedia.org/wiki/Cauda_longa) nas colunas "chlorides" e "residual sugar". Em "quality" não conseguimos ver muito bem a distribuição para cada classe. Nos demais gráficos vemos as distribuições sendo empurradas para a esquerda por outliers.


```python
df.quality.value_counts().sort_index() # contagem de amostras por qualidade
```




    3      27
    4     203
    5    1688
    6    2262
    7     839
    8     147
    9       5
    Name: quality, dtype: int64




```python
df.quality.value_counts().sort_index().plot.bar() # Gráfico de barras
plt.show()
```


![png](_img/CognitiveAI_37_0.png)


Com o campo 'quality', foram contadas quantas amostras por categoria de qualidade. A maioria das amostras está entre os níveis 5, 6 e 7. Há poucas amostras para os demais níveis. Tendo isto em vista, espera-se que o modelo performe melhor para os níveis 5-7 e, devido a baixa quantidade de explares, pode não performar muito bem para os níveis 3-4 e 8-9


```python
# Cógigo utilizado para auxiliar nas transformações de campos abaixo
# a = df.copy()
# df = a.copy()
# df[df.isin([np.nan, np.inf, -np.inf]).any(1)]
```

```python
# Aplicação de função logarítmica
df.chlorides = df.chlorides.apply(np.log) 
df['residual sugar'] = df['residual sugar'].apply(np.log)

# Aplicação de função quadrática
df['citric acid'] = df['citric acid'].apply(np.sqrt)
df['fixed acidity'] = df['fixed acidity'].apply(np.sqrt)
df['free sulfur dioxide'] = df['free sulfur dioxide'].apply(np.sqrt)
df.sulphates = df.sulphates.apply(np.sqrt)
df['volatile acidity'] = df['volatile acidity'].apply(np.sqrt)
df['fixed acidity'] = df['fixed acidity'].apply(np.sqrt)
df['total sulfur dioxide'] = df['total sulfur dioxide'].apply(np.sqrt)
df['volatile acidity'] = df['volatile acidity'].apply(np.sqrt)

# Histogramas
df.hist(figsize=(10,10), color='green')
plt.show()
```


![png](_img/CognitiveAI_41_0.png)


Para os campos que apresentaram aspecto de cauda longa aplicou-se função logarítmica e função quadrática para os demais, afim de melhorar a visualização e distribuição dos dados.

### Análise Bivariada


```python
corr_matrix = df.drop('type', axis=1).corr().abs() # Matriz de correlação
plt.figure(figsize=(10,10)) # Dimensionamento da figura
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm') # Mapa de calor + Matriz
plt.show()
```


![png](_img/CognitiveAI_44_0.png)


Utilizamos uma matriz de correlação para auxiliar no estudo da influência de uma variável nas outras (correlação).  
Omitiu-se o campo "type" por se tratar de um categoria, ainda que seu tipo seja numérico. Nota-se correlações fracas (0.3-0.5), moderadas (0.5-0.7) e fortes (0.7-0.9). Entretanto não foram encontradas correlações muito fortes (0.9-1.0).
 [Saiba mais](https://pt.wikipedia.org/wiki/Coeficiente_de_correlação_de_Pearson).  
Vale ressaltar que, de acordo com a matriz, os campos que mais influenciam a qualidade do vinho são "alcohol", "density" e "chlorides".



```python
# correlations = ['type', 'volatile acidity', 'citric acid', 'pH', 'sulphates']
# sns.pairplot(df.drop(correlations, axis=1), kind="reg")
sns.pairplot(df, kind="reg") # Gráficos combinados
plt.show()
```


![png](_img/CognitiveAI_46_0.png)


Acima temos um resumo dos histogramas e gráficos bivariados, onde é possível visualizar o sentido das correlações. Complementando a influência das variáveis na qualidade, "alcohol" influencia moderada e positivamente a qualidade do vinho, ou seja, quanto maior o teor alcoólico maior tende a ser a qualidade do vinho; em contra partida "density" e "chlorides" influenciam fraca e negativamente, o que nos leva a crer que vinhos com mais densidade ou mais sal tendem a perder em qualidade.

---

## Modelagem


```python
seed = 0 # Semente aleatória
x = df.drop('quality', axis=1) # Variáveis de entrada
y = df.quality # Variáveis de saída
```

Na fase de modelagem, primeiro vamos definir uma semente aleatória para garantir reproducibilidade do processo. Na sequência, iremos separar as colunas em variáveis de entrada e variável de saída.


```python
sel = SelectKBest(f_classif, k = 'all').fit(x, y) #Seletor de melhores variáveis
KBestTable(sel, x, x.columns) # Tabela de melhores variáveis
```




    alcohol                 304.375080
    density                 141.537230
    chlorides                80.904321
    volatile acidity         78.164984
    free sulfur dioxide      19.047457
    citric acid              18.333549
    type                     15.191395
    fixed acidity             7.993347
    residual sugar            6.427303
    total sulfur dioxide      6.382442
    sulphates                 4.805110
    pH                        2.876678
    dtype: float64



Utilizou-se o módulo SelectKBest para ordenar as colunas por importância em relação a qualidade e, como esperado, as três mais importante são "alcohol", "density" e "chlorides". A métrica utilizada para isso foi a f_classif baseada na função F da ANOVA. [Saiba mais](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/f-statistic-value-test/).  
Também foi adicionado o módulo SelectKBest ao modelo, tornando-o uma sequência de passos (pipeline). Este módulo permite treinarmos o modelo com um número diferente de variáveis, já que uma maior quantidade de colunas não necessariamente melhora as previsões, podendo até confundi-lo em alguns casos.


```python
# Modelos
clf1 = GaussianNB()
clf2 = DecisionTreeClassifier(random_state=seed)
clf3 = RandomForestClassifier(random_state=seed)
clf4 = MLPClassifier(random_state=seed)
clf5 = KNeighborsClassifier()
clf6 = SVC(random_state=seed)
clf7 = LogisticRegression()

# Parâmetros por modelo
prm1 = {}
prm2 = {'clf__max_depth':[None, 5, 10],'clf__min_samples_leaf':[1,20,50]}
prm3 = {'clf__n_estimators':[10,50,100],'clf__max_depth':[None, 5, 10],'clf__min_samples_leaf':[1,20,50]}
prm4 = {'clf__learning_rate_init':[0.001, 0.01], 'clf__max_iter':[200,500]}
prm5 = {'clf__n_neighbors': [3,5,10], 'clf__weights':['uniform', 'distance']}
prm6 = {'clf__C':[1,0.5], 'clf__shrinking':[False, True]}
prm7 = {'clf__solver':['sag'],'clf__multi_class':['multinomial'], 'clf__max_iter': [3,7]}

# Divisão Estratificada
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=seed)

#
summary = []
models = {}

#Looping para aplicar a função para todos os classificadores e parâmetros e armazenar os resultados no dicionário resultados
for clf, prm in [[clf1,prm1], [clf2,prm2],[clf3,prm3],[clf4,prm4],
                 [clf5,prm5],[clf6,prm6],[clf7,prm7]]:
  results, model = clfs_train_test(clf, prm, x_train, x_test, y_train, y_test)
  summary.append(results)
  models[results['nome']] = model
```

    GaussianNB
    --------------------
    
    Tempo de treino: 0.3286120891571045
    A melhor combinação de parâmetros:
    {'sel__k': 6}
    Maior F1-score: 0.47295358500266055
    F1-score de Teste: 0.48488531813265123
    Reporte de classificação:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.17      0.10      0.13        41
               5       0.55      0.64      0.59       338
               6       0.53      0.47      0.50       453
               7       0.38      0.46      0.42       168
               8       0.00      0.00      0.00        29
               9       0.00      0.00      0.00         1
    
        accuracy                           0.50      1035
       macro avg       0.23      0.24      0.23      1035
    weighted avg       0.48      0.50      0.48      1035
    
    
    
    
    DecisionTreeClassifier
    --------------------
    
    Tempo de treino: 3.4856069087982178
    A melhor combinação de parâmetros:
    {'clf__max_depth': 5, 'clf__min_samples_leaf': 50, 'sel__k': 9}
    Maior F1-score: 0.5062948385051201
    F1-score de Teste: 0.4698899322539814
    Reporte de classificação:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.00      0.00      0.00        41
               5       0.55      0.56      0.55       338
               6       0.49      0.66      0.56       453
               7       0.43      0.20      0.27       168
               8       0.00      0.00      0.00        29
               9       0.00      0.00      0.00         1
    
        accuracy                           0.50      1035
       macro avg       0.21      0.20      0.20      1035
    weighted avg       0.46      0.50      0.47      1035
    
    
    
    
    RandomForestClassifier
    --------------------
    
    Tempo de treino: 114.52161431312561
    A melhor combinação de parâmetros:
    {'clf__max_depth': None, 'clf__min_samples_leaf': 1, 'clf__n_estimators': 100, 'sel__k': 9}
    Maior F1-score: 0.5402294947943228
    F1-score de Teste: 0.5180594140779924
    Reporte de classificação:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.25      0.02      0.04        41
               5       0.59      0.63      0.61       338
               6       0.53      0.65      0.59       453
               7       0.50      0.31      0.38       168
               8       0.00      0.00      0.00        29
               9       0.00      0.00      0.00         1
    
        accuracy                           0.54      1035
       macro avg       0.27      0.23      0.23      1035
    weighted avg       0.51      0.54      0.52      1035
    
    
    
    
    MLPClassifier
    --------------------
    
    Tempo de treino: 120.31397151947021
    A melhor combinação de parâmetros:
    {'clf__learning_rate_init': 0.001, 'clf__max_iter': 200, 'sel__k': 'all'}
    Maior F1-score: 0.5078604581738485
    F1-score de Teste: 0.53347875900172
    Reporte de classificação:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.67      0.05      0.09        41
               5       0.59      0.69      0.64       338
               6       0.56      0.62      0.59       453
               7       0.45      0.37      0.41       168
               8       0.00      0.00      0.00        29
               9       0.00      0.00      0.00         1
    
        accuracy                           0.56      1035
       macro avg       0.32      0.25      0.25      1035
    weighted avg       0.54      0.56      0.53      1035
    
    
    
    
    KNeighborsClassifier
    --------------------
    
    Tempo de treino: 3.361093759536743
    A melhor combinação de parâmetros:
    {'clf__n_neighbors': 10, 'clf__weights': 'distance', 'sel__k': 'all'}
    Maior F1-score: 0.5039502968942376
    F1-score de Teste: 0.48734685702335473
    Reporte de classificação:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.00      0.00      0.00        41
               5       0.56      0.56      0.56       338
               6       0.51      0.64      0.57       453
               7       0.42      0.30      0.35       168
               8       0.00      0.00      0.00        29
               9       0.00      0.00      0.00         1
    
        accuracy                           0.51      1035
       macro avg       0.21      0.21      0.21      1035
    weighted avg       0.47      0.51      0.49      1035
    
    
    
    
    SVC
    --------------------
    
    Tempo de treino: 58.264386892318726
    A melhor combinação de parâmetros:
    {'clf__C': 1, 'clf__shrinking': False, 'sel__k': 'all'}
    Maior F1-score: 0.4895379714371229
    F1-score de Teste: 0.49412302192508517
    Reporte de classificação:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.00      0.00      0.00        41
               5       0.59      0.63      0.61       338
               6       0.52      0.73      0.60       453
               7       0.53      0.11      0.19       168
               8       0.00      0.00      0.00        29
               9       0.00      0.00      0.00         1
    
        accuracy                           0.54      1035
       macro avg       0.23      0.21      0.20      1035
    weighted avg       0.51      0.54      0.49      1035
    
    
    
    
    LogisticRegression
    --------------------
    
    Tempo de treino: 0.892695426940918
    A melhor combinação de parâmetros:
    {'clf__max_iter': 7, 'clf__multi_class': 'multinomial', 'clf__solver': 'sag', 'sel__k': 'all'}
    Maior F1-score: 0.44552472047172803
    F1-score de Teste: 0.47064050910913174
    Reporte de classificação:
                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         5
               4       0.00      0.00      0.00        41
               5       0.61      0.52      0.56       338
               6       0.50      0.79      0.61       453
               7       0.44      0.07      0.12       168
               8       0.00      0.00      0.00        29
               9       0.00      0.00      0.00         1
    
        accuracy                           0.53      1035
       macro avg       0.22      0.20      0.19      1035
    weighted avg       0.49      0.53      0.47      1035
    
    
    
    


Definiu-se o problema como classificação. Já que não há opções de valores entre as classes, não podemos classificar um vinho como 5,5 por exemplo.  
Para o modelo, foram testados GaussianNB, DecisionTree, RandomForest, MLP, KNN, SVC e LogisticRegression. Cada um com algumas opções de parâmetros. Todas as divisões de dataset foram feitas de modo estratificado, ou seja, toda divisão utiliza uma proporção de cada nível de qualidade para que todos os treinos e testes sejam realizados com todas as classes.



```python
ordenado = ['nome', 'best_score', 'score_treino', 'tempo_treino']
pd.DataFrame(summary, columns=ordenado)\
  .sort_values('score_treino', ascending=False)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nome</th>
      <th>best_score</th>
      <th>score_treino</th>
      <th>tempo_treino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>MLPClassifier</td>
      <td>0.507860</td>
      <td>0.533479</td>
      <td>120.313972</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier</td>
      <td>0.540229</td>
      <td>0.518059</td>
      <td>114.521614</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SVC</td>
      <td>0.489538</td>
      <td>0.494123</td>
      <td>58.264387</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNeighborsClassifier</td>
      <td>0.503950</td>
      <td>0.487347</td>
      <td>3.361094</td>
    </tr>
    <tr>
      <th>0</th>
      <td>GaussianNB</td>
      <td>0.472954</td>
      <td>0.484885</td>
      <td>0.328612</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LogisticRegression</td>
      <td>0.445525</td>
      <td>0.470641</td>
      <td>0.892695</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTreeClassifier</td>
      <td>0.506295</td>
      <td>0.469890</td>
      <td>3.485607</td>
    </tr>
  </tbody>
</table>
</div>



Acima temos uma tabela que sumariza os resultados. O modelo MLP teve o melhor resultado seguido pelo RandomForest. Coincidentemente, ambos os modelos tiveram os maiores tempos de treinamento. Devido a baixa diferença entre eles, definiu-se MLP (com os parâmetros indicados) como o melhor modelo dentre os testados, ainda que RandomForest tenha se saído melhor na etapa de treino.

---

## Conclusão

1. **Quais variáveis impactam na qualidade do vinho?**  
De acordo com a matriz de correlação, os gráficos bivariados e o módulo SelectKBest, as variáveis que mais impactam na qualidade do vinho são: alcohol, density e chlorides.
1. **Como foi a definição da sua estratégia de modelagem?**  
Para modelagem, definiu-se um processo de duas etapas: seleção de variáveis (SelectKBest) e modelo de ML.
Para a segunda etapa, foram testados 7 modelos de ML com diferentes opções de parâmetros. Utilizou-se técnicas de hiper-parametrização e validação cruzada para definir o melhor modelo, com a melhor combinação de parâmetros e número de variáveis, de modo a maximizar a métrica F-1.
1. **Como foi definida a função de custo utilizada?**  
A função de custo utilizada foi a métrica F-1, pois considerou-se importante não somente a quantidade de acertos do modelo, mas também que ele seja capaz de identificar bem cada um dos níveis de qualidade. Sendo a métrica citada uma média harmônica entre precisão e sensibilidade, que oferece um ótimo balanceamento e, consequentemente, uma boa métrica para este fim.
1. **Qual foi o critério utilizado na seleção do modelo final?**  
Separou-se um conjunto de dados de teste para simular um ambiente com dados desconhecidos. Ainda que o RandomForest tenha apresentado melhor pontuação para os dados de treino, escolheu-se MLP como modelo final por este ter se saído melhor com os dados de teste, sem que o tempo de treinamento tenha apresentado diferença significativa em relação ao outro modelo.
1. **Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?**  
Para validação do modelo, utilizou-se duas técnicas: validação cruzada e divisão estratificada. Devido ao baixo número de amostras dos níveis de qualidade 3, 4, 8 e 9 em relação aos demais, torna-se importante um processo que, mesmo aleatório, consiga selecionar ao menos uma amostra de cada classe tanto para treino quanto para teste. Deste modo, evita-se a obtenção de altas pontuações por acaso da divisão selecionar apenas amostras com as qualidade que o modelo esteja performando muito bem. Em adição, os modelos foram testados com uma amostra estratificada omitida na etapa de treinamento, simulando assim um caso real com dados novos.
1. **Quais evidências você possui de que seu modelo é suficientemente bom?**  
Ainda que os dados e modelos tenham passado por todos esses diversos processos, não se considera que o modelo final seja suficientemente bom. Sendo assim, restam as sugestões abaixo:
  + Obter mais dados das classes 3,4,8 e 9;
  + Criar modelos diferentes para vinhos tintos e vinhos brancos;
  + Testar modelos mais complexos como xgboost, redes neurais mais profundas e modelos combinados;
  + Testar mais parâmetros;
  + Realizar mais engenharia de feature para criação/transformação/combinação de possíveis variáveis com mais impacto na qualidade do vinho.

---

## Referências

+ <https://github.com/MwillianM/EronFraudIdentification>
+ <http://google.github.io/styleguide/pyguide.html>
+ <http://www.uvibra.com.br/legislacao_portaria229.htm>
+ <https://unsplash.com/>
+ <https://blog.famigliavalduga.com.br/afinal-o-que-e-vinho-verde/>
+ <https://drive.google.com/open?id=1-oG5-kBt9xQ3Li4PEexpiA9_7RZhRM1f>
+ <https://s3.amazonaws.com/udacity-hosted-downloads/ud651/wineQualityInfo.txt>
+ <https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/f-statistic-value-test/>

---
