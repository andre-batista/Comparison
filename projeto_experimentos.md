# Projeto de Pesquisa - Revisão do Método Iterativo de Born

## Pressupostos gerais

* Objetos perfeitamente dielétricos.
* Problema bidimensional
* Algoritmo para problema direto: Method of Moments - CG-FFT

## Perguntas da pesquisa

### Qual é o melhor valor do parâmetro de Tikhonov para uma estratégia fixa?

1. Estudar qual o $\alpha$ com o menor erro ao se variar o valor de contraste.
2. Estudar qual o $\alpha$ com o menor erro ao variar o tamanho dos objetos.
3. Estudar qual o $\alpha$ com o menor erro ao variar a magnitude do campo incidente.

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
* Eu não estou colocando o DBIM em análise. Basicamente, a diferença entre os dois é que o BIM é mais robusto a ruído e o DBIM converge mais rápido. Só que essas conclusões foram tiradas lá em 1990 quando computador não tinha memória e sem rigor estatístico nenhum (fez três estudos casos só). Poderia ser interessante incluir aqui, mas eu acho que vai ficar grande demais se entrar com o DBIM.
