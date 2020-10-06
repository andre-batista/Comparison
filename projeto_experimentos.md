# Projeto de Pesquisa - Revisão do Método Iterativo de Born

## Pressupostos gerais

* Objetos perfeitamente dielétricos.
* Problema bidimensional
* Algoritmo para problema direto: Method of Moments - CG-FFT

## Perguntas da pesquisa

### Qual é o melhor valor do parâmetro de Tikhonov para uma estratégia fixa?
  1. Escolher valor de acordo com uma análise da variação do valor máximo de contraste
    * Discretização por pulso (Método do Subdomínio)
    * Indicador de qualidade: Desvio Médio Percentual da Permissividade Relativa.
    * Fixar uma quantidade razoável de ruído, densidade de objetos, tamanho de objetos, raio de observação e fontes/receptores.
    * Fixar um tipo de objeto: polígonos aleatórios.
    * Intervalo de contrastes: 1 a 16.
    * Intervalo de valores do parâmetro: 1e-15 a 1e-1, multiplicando por 10.
    * Análise: (i) verificar que valor de parâmetro é melhor pra cada valor de contraste e (ii) ajustar uma curva de valor de parâmetro em função do valor de contraste.
  2. Testar regra de definição do valor do parâmetro:
    * Rodar em cinco níveis de máximo contraste para analisar os outros indicadores de qualidade.
    * Testar impacto dos fatores de máximo contraste, máximo tamanho de objeto e máxima densidade (análise fatorial) - se impacta e de quanto pode ser a perda;
    * Testar impacto dos fatores de máximo contraste, ruído, forma dos objetos (análise fatorial) - se impacta e o quanto impacta;

### Existe alguma diferença em performance entre a estratégia fixa e as outras disponíveis?
  1. Comparar os diferentes métodos de escolha (fixo, Princípio de Mozorov, Lavarello 2010, L-curve) em três níveis de máximo contraste
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.
  2. Comparar os diferentes métodos quando o padrão são superfícies
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.
  3. Comparar os diferentes métodos em três níveis de ruídos
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.

### Existe alguma diferença em performance entre o Método de Tikhonov com parâmetro fixo em relação aos outros métodos de regularização (Landweber e CG)?
  0. Definir uma parametrização padrão pro Método de Landweber.
  1. Comparar os diferentes métodos de regularização (Tikhonov, Landweber e CG) em três níveis de máximo contraste
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.
  2. Comparar os diferentes métodos quando o padrão são superfícies
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.
  3. Comparar os diferentes métodos em três níveis de ruídos
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.

### Existe alguma diferença em performance entre a discretização de subdomínio e os métodos de Colocação e Galerkin?
  1. Investigar possíveis diferenças na escolha da resolução e das funções de base do Método da Colocação
    * Escolher uma malha de elementos conforme com três níveis de máximo contraste para as duas funções de base.
    * Escolher função de base com três níveis de contraste máximo, três níveis de ruído e duas formas de padrões de imagem (polígonos aleatórios e superfíes).
  2. Repetir o passo anterior para o Método do Galerkin.
  3. Comparar os diferentes métodos de discretização (Subdomínio, Colocação e Galerkin) em três níveis de máximo contraste
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.
  4. Comparar os diferentes métodos quando o padrão são superfícies
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.
  5. Comparar os diferentes métodos em três níveis de ruídos
    * Salvar todos os indicadores.
    * Níveis básicos dos outros fatores.

## Limitações do estudo

* Investigar o comportamento com objetos com condutividade pode ser interessante.
* Muitas comparações são deixadas de lado pois são muitos fatores que podem ser levado em conta (ex.: impacto do raio de observação, do número de fontes/receptores).
