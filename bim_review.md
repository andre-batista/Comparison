# REVISÃO DO MÉTODO ITERATIVO DE BORN

## Artigos originais do BIM e do DBIM

1. Wang, Y. M., and Weng Cho Chew. "An iterative solution of the two‐dimensional electromagnetic inverse scattering problem." International Journal of Imaging Systems and Technology 1.1 (1989): 100-108.

  - No resumo, ele afirma que emprega uma solução equivalente à da série de Neumann, embora não fale sobre isso ao longo do artigo. Também diz que se propõe a descrever um método geral, i.e., que pode ter diferentes implementações. Esse método geral tem o objetivo de resolver os problemas quando as aproximações de Born e de Rytov não dão conta, i.e., quando os espalhadores não são fracos.
  - Na revisão bibliográfica, chamou a atenção uma referência que implementa uma aproximação chamada *"Distorted Born Approximation"* [9, 10] pra problemas com uma dimensão. Daí pode ter sido inspirado o DBIM no ano seguinte. Chamou atenção também um trabalho que emprega Point-Matching (Método de Colocação) pra um problema específico tri-dimensional. O que mais chamou a atenção foi o da Transformação da Pseudoinversa, no qual a fonte de corrente é reconstruída e o contraste é reconstruído a partir disso. Desse trabalho, eles comentam que *"o sucesso do método depende da solução de um problema de fonte inverso que tem dificuldades de unicidade por causa de fontes não-radiantes"*. 
  - Em relação à questão de má-posição do problema, os autores citam um livro [15] e dois artigos [13, 14] que discorrem sobre a não-unicidade da solução do problema. São então **boas referências quando eu for falar dos motivos pelos quais esse problema é mal-posto**. Inclusive, eles chegam a dizer que não há informações suficientes que sejam capazes de recuperar *"altas frequências da solução"*, o que provavelmente significa discontinuídades dos objetos (arestas). Também diz que ondas espalhadas evanescentes se tornam exponencialmente pequenas conforme a distância dos receptores, o que contribui mais pra má-posição do problema. Ou seja, ondas evanescentes melhores captadas poderiam contribuir para a solução.
  - Em relação ao método de regularização que eles empregam pro subproblema inverso, que é o de Tikhonov, eles enxergam que o problema, por si só, não é capaz de definir uma solução única com as informações que tem, mas dá pra achar uma *"classe de soluções"* dentre outras na qual a regularização é uma implementação de uma restrição arbitrária que vai decidir por uma solução. Então tem a ver com decisões *a priori*. A seleção do parâmetro de regularização é responsável por filtrar componentes de alta frequência que são instáveis mas não deve filtrar demais porque algumas frequências podem ser muito úteis (arestas de objetos). O artigo não define necessariamente um valor do parâmetro e regularização, mas pontua que: (i) é melhor considerar um parâmetro como desconhecido e (ii) escolher com base em simulações. Sobre o intervalo, ele diz que será pequeno quando a solução for muito sensível à escolha do parâmetro, e assim,  significa que a quantidade de dados não é suficiente para a esperada precisão. Ele chega a sugerir o parâmetro como algo de 10^-10 a 10^-15. Mas aconselha que, qualquer algoritmo que seja montado a partir desse procedimento deve ser testado em simulações onde a aproximação de Born é válida.
  - Na explicação do método, ele não chamam o método proposto de BIM, mas de um *método de Newton modificado*.
  - O problema direto é resolvido pelo Método dos Momentos (Point-Matching, Collocation). E o procedimento inverso é feito por um método de regularização que é o de Tikhonov com discretização por pulso (subdomínio). A discretização é feita considerando a densidade indicada pro Método dos Momentos (100/lambda^2) e conforme a quantidade de memória da época. Isso vai influenciar tamanho do domínio e dos objetos nos experimentos. O chute inicial é a aproximação de Born. A grande diferença em relação ao DBIM, é que, aqui, a função de Green vai ser constante ao longo do processo. 
  - Considerações gerais sobre os experimentos: (i) 5 a 12 iterações; (ii) 4 a 8 fontes; (iii) 26 a 36 receptores; e (iv) resolução de 11x11 a 19x19. Isso tudo é bem restrito às condições dos computadores da época. O critério de parada é uma diferença de menos de 5% nos dados do campo espalhado.
  - O erro que ele define chama MSE (mean squared error) e basicamente ele integra o erro ao quadrado, divide pela integração do original ao quadrado e tira a raiz.
  - Experimentos:
    1. Distribuição senoidal, contraste = 10; f = 10MHz; Diâmetro = .1\*lambda; 5 iterações.
    2. Distribuição senoidal; f = 100MHz; contraste = 0.8; Diâmetro = lambda; 8 iterações.
    3. Distribuição senoidal; f = 200MHz, Diâmetro = 2\*lambda; 8 fontes; 12 iterações.  
      - O erro entre os mapas exatos e recuperados nos três primeiros experimentos eram menores que 1% em cada ponto. E ele justifica que a propriedade da restrição empregada no problema inverso coincide com a distribuição original (ele quer dizer que essa estratégia do Tikhonov é muito interessante para distribuições contínuas).
    4. Quadrado, f = 100MHz; contraste = 0.6; 8 iterações.
      - O algoritmo se comporta como um filtro passa-baixa.
    5. Quadrado, f = 10MHz; Diâmetro = .1\*lambda; contraste = 10; 5 iterações.
    6. Distribuição contínua assimétrica, 6 iterações

2. Chew, Weng Cho, and Yi-Ming Wang. "Reconstruction of two-dimensional permittivity distribution using the distorted Born iterative method." IEEE transactions on medical imaging 9.2 (1990): 218-225.

  - O DBIM tem convergência mais rápida e o BIM é mais robusto a ruído;
  - Engraçado que agora ele chama o método anterior de BIM sendo que no artigo anterior eles nem usaram esse nome;
  - Resolve o mesmo problema mas a função de Green é atualizada em cada iteração;
  - Processo:
    1. Chute inicial: aproximação de Born com brackground homogêneo e vácuo;
    2. Roda o método direto (outra vez Método dos Momentos);
    3. Calcular a função de Green usando o background heterogêneo sendo o mapa recuperado até então;
    4. Resolve o problema inverso não pra achar o contraste atual, mas para um *delta* que será adicionado ao contraste atual. Por isso, a variável conhecida não é exatamente o campo espalhado de entrada, mas o mesmo subrtraído pelo achado no passo anterior;
    5. Critério de parada: RRE (relative residual error) menor que um valor or maior que o anterior. O RRE é o somatório do módulo do erro nos dados de espalhamento dividido pelo somatório do módulo dos dados.
  - Experimento 1:
    1. Distribuição não completamente contínua e não simétrica; f = 100MHz; contraste = 0.8; Diâmetro = lambda; 25 iterações.
      * Pouca diferença entre a 4a e a 25a iteração pois o critério RRE foi definido como < 1e-5
    2. Distribuição senoidal; f = 100MHz; contraste = 0.8; Diâmetro = lambda.
      * DBIM converge em 4 e BIM converge em 6.
    3. Dois quadrados; Diâmetro = .25\*lambda; Distância = .25\*lambda; f = 100MHz; contraste = 0.8.
      * DBIM convergindo em 4 e BIM convergindo em 6.
    4. Distribuição senoidal com 25dB de SNR; Diâmetro = lambda, contraste = 0.8.
      * DBIM não consegue convergir monotonicamente enquanto BIM consegue. Os caras usam um tipo de filtro para tirar ruídos da imagem final.
      * A explicação do motivo do ruído atrapalhar o DBIM é que, como ele resolve a equação integral subtraindo o campo espalhado dado como entrada pelo estimado pela interação, então chega uma hora que os ruídos prevalecem sobre as informações sobre o objeto. Aí não compensa mais. Mas por que ele faz essa subtração? Na real, ele resolve pra achar o delta_contraste que é somado ao contraste atual.