# REVISÃO DO MÉTODO ITERATIVO DE BORN

## Artigos originais do BIM e do DBIM

1. Wang and Chew, 1989

  - Emprega uma solução equivalente à da série de Neumann;
  - Se propõe a ser um método geral
  - Pensado na situação onde as aproximações de Born e Rytov não são suficientes (contrastes não tão fracos);
  - Bons resultados para frequências altas e baixas considerando imagens contínuas ou discontínuas (sempre dielétricos perfeitos);
  - Resolve as equações com uma só frequência, oito fontes.
  - Naquela altura, o Distorted Born Approximation tinha sido proposto [9, 10] pra problemas com uma dimensão.
  - Eles citam uma técnica de Point-Matching (que pode ser um método de colocação) pra um problema específico tri-dimensional.
  - Ao citar um trabalho de Transformação da Pseudoinversa, no qual a fonte de corrente é reconstruída e o contraste é reconstruído a partir disso, eles comentam que "o sucesso do método depende da solução de um problema de fonte inverso que tem dificuldades de unicidade por causa de fontes não-radiantes". Que são essas fontes não radiantes? Daí é citado um livro [15] e dois artigos [13, 14] que abordam isso. Então referências bem concretas sobre a questão da não-unicidade. Ele também fala posteriormente de instabilidade da solução do problema e da má-condição da matriz do operador citando dois livros. Ele afirma que os dados de entrada não dão informação suficiente sobre "altas frequências da solução" (o que talvez pode ser questão de altos contrastes, onde o comprimento de onda seja mais curto). Ele diz que "os dados de entrada implicam somente que a solução tem que estar dentro de uma classe específica de soluções, mas não dão orientação pra qual solução única deve ser feita dentro daquela classe". O que são essas classes? (O método de regularização coloca uma restrição que permite selecionar uma solução única, é uma restrição arbitrária e que não tem a ver com os dados medidosm as informações a priori) Ele também fala de ondas espalhadas evanescentes que se tornam exponencialmente pequenas nos receptores. Isso torna o provlema mais mal-condicionado.
  - Eles chamam o algoritmo deles de um método de Newton modificado.
  - O problema direto é resolvido pelo Método dos Momentos (Point-Matching = Collocation). E o procedimento inverso é feito por um método de regularização que é o de Tikhonov.
  - 5 a 12 iterações nos experimentos.
  - Ele considera ondas planas e fontes de linha.
  - A grande diferença em relação ao DBIM, é que, aqui, a função de Green vai ser constante ao longo do processo.
  - O chute inicial é a aproximação de Born.
  - O critério de parada é uma diferença de menos de 5% nos dados do campo espalhado.
  - A discretização de função de base é feita pro subproblema inverso.
  - A seleção do parâmetro de regularização é responsável por filtrar componentes de alta frequência que são instáveis mas não deve filtrar demais porque algumas frequências podem ser muito úteis.
  - O artigo não define necessariamente um valor do parâmetro e regularização, mas pontua que: (i) é melhor considerar um parâmetro como desconhecido e (ii) escolher com base em simulações. Sobre o intervalo, ele diz que será pequeno quando a solução for muito sensível à escolha do parâmetro, e assim,  significa que a quantidade de dados não é suficiente para a esperada precisão.
  - Ele chega a sugerir o parâmetro como algo de 10^-10 a 10^-15. Mas aconselha que, qualquer algoritmo que seja montado a partir desse procedimento deve ser testado em simulações onde a aproximação de Born é válida.
  - Ele diz que o método dos momentos precisa de uma densidade de malha de 100/lambda^2 pra ter bons resultados.
  - Experimento 1: epsilon_r=11, f=10MHz, diametro=.1*lambda, 4a8 fontes, 26 a 36 receptores, malha=121(11x11) a 361(19x19) nós, 5 iterações
  - Experiment 2: f=100MHz, epsilon_r=1.8, diametro=lambda, 8 iteracoes,
  - Experimento 3: f=200MHz, diametro=2*lambda, 8 fontes, 12 iteracoes
  - O erro entre os mapas exatos e recuperados eram menores que 1% em cada ponto. E ele justifica que a propriedade da restrição empregada no problema inverso coincide com a distribuição original (ele quer dizer que essa estratégia do Tikhonov é muito interessante para distribuições contínuas).
  - Experimento 4: quadrado, algoritmo se comporta como um filtro passa-baixa, f=100MHz, epsilon_r=1.6, 8 iterações
  - Experimento 5: f=10MHz, diametro=.1*lambda, epsilon_r=11, 5 iteracoes,
  - Experimento 6: distribuição contínua assimétrica, 6 iterações
  - O erro que ele define chama MSE (mean squared error) e basicamente ele integra o erro ao quadrado, divide pela integração do original ao quadrado e tira a raiz.

2. Chew and Wang, 1990

  - O DBIM tem convergência mais rápida e o BIM é mais robusto a ruído.
  - Engraçado que agora ele chama o método anterior de BIM sendo que no artigo anterior eles nem usaram esse nome.
  - Resolve o mesmo problema mas a função de Green é atualizada em cada iteração (o meio de fundo não é constante).
  - Chute inicial: aproximação de Born com brackground homogêneo e vácuo -> roda o método direto (outra vez método dos momentos) -> Calcular a função de Green usando o background heterogêneo sendo o mapa recuperado até então -> resolve o problema inverso -> Critério de parada: RRE (relative residual error) menor que um valor or maior que o anterior.
  - O RRE é o somatório do módulo do erro nos dados de espalhamento dividido pelo somatório do módulo dos dados.
  - Experimento 1: distribuição não completamente contínua e não simétrica, f=100MHz, epsilon_r=1.8, diametro=lambda, 25 iteracoes, pouca diferença entre a 4a e a 25a iteraçao, critério: RRE < 1e-5
  - Experimento 2: distribuição senoidal, f=100MHz, epsilon_r=1.8, diametro=lambda; DBIM converge em 4 e BIM converge em 6.
  - Experimento 3: dois quadrados, diametro=.25*lambda, distancia=.25*lambda, f=100MHz, epsilon_r=1.8; DBIM convergindo em 4 e BIM convergindo em 6.
  - Experimento 4: distribuição senoidal com 25dB de SNR, diametro=lambda, epsilon_r=1.8, DBIM não consegue convergir monotonicamente enquanto BIM consegue. Os caras usam um tipo de filtro para tirar ruídos da imagem final.
  - A explicação do motivo do ruído atrapalhar o DBIM é que, como ele resolve a equação integral subtraindo o campo espalhado dado como entrada pelo estimado pela interação, então chega uma hora que os ruídos prevalecem sobre as informações sobre o objeto. Aí não compensa mais. Mas por que ele faz essa subtração? Na real, ele resolve pra achar o delta_contraste que é somado ao contraste atual.
