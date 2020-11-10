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

## Citaram o BIM

### Até 120 citações

1. Joachimowicz, Nadine, Christian Pichot, and Jean-Paul Hugonin. "Inverse scattering: An iterative numerical method for electromagnetic imaging." IEEE Transactions on Antennas and Propagation 39.12 (1991): 1742-1753.
    * Cita o BIM (entre outros) dizendo que a convergência depende do contraste do objeto.
    * Usa o regularizador de Tikhonov e tem uma expressão fechada para o valor do parâmetro. Só que essa expressão tem uma constante definida arbitrariamente e de acordo com a convergência também.

2. Colton, David, Joe Coyle, and Peter Monk. "Recent developments in inverse acoustic scattering theory." Siam Review 42.3 (2000): 369-414.
    * Cita o BIM sem comentar o método.
    * Fala sobre a regularização de Tikhonov e o Princípio de Discrepância de Mozorov

3. Kleinman, R. E., and P. M. Van den Berg. "A modified gradient method for two-dimensional problems in tomography." Journal of Computational and Applied Mathematics 42.1 (1992): 17-35.
    * Fala muito sobre a série de Born como solução para o problema, citando os trabalhos do BIM e do DBIM. É um trabalho de mesma época que se propõe a ser alternativa a isso.
    * Não tem regularizador de Tikhonov.

4. Franchois, Ann, and Christian Pichot. "Microwave imaging-complex permittivity reconstruction with a Levenberg-Marquardt method." IEEE Transactions on Antennas and Propagation 45.2 (1997): 203-215.
   * É um método equivalente ao DBIM.
   * Define duas formas de definir o valor do parâmetro regularizador de Tikhonov: uma forma empírica (com esquema de atualização) e o GCV (Generalized Cross Validation).

5. Song, Lin-Ping, Chun Yu, and Qing Huo Liu. "Through-wall imaging (TWI) by radar: 2-D tomographic results and analyses." IEEE Transactions on Geoscience and Remote Sensing 43.12 (2005): 2793-2798.
    * Apenas cita o BIM e o DBIM. A proposta é um CSI pra um caso de multi-camadas.

6. Ernst, Jacques R., et al. "Application of a new 2D time-domain full-waveform inversion scheme to crosshole radar data." Geophysics 72.5 (2007): J53-J64.
    * Ligeira citação sobre o BIM e o DBIM. A proposta é um esquema no domínio do tempo.

7. Chew, Weng Cho, and J. H. Lin. "A frequency-hopping approach for microwave imaging of large inhomogeneous bodies." IEEE Microwave and Guided Wave Letters 5.12 (1995): 439-441.
    * É o DBIM e não fala como escolheu o parâmetro de regularização. Artigo muito curto.

8. Harada, Haruyuki, et al. "Conjugate gradient method applied to inverse scattering problem." IEEE Transactions on Antennas and Propagation 43.8 (1995): 784-792.
    * Apenas cita BIM e DBIM como revisão de literatura. É um CG, por isso, não tem regularização de Tikhonov.

9. Ernst, Jacques R., et al. "Full-waveform inversion of crosshole radar data based on 2-D finite-difference time-domain solutions of Maxwell's equations." IEEE transactions on geoscience and remote sensing 45.9 (2007): 2807-2828.
    * Apenas cita como revisão de literatura. O método não tem regularização de Tikhonov.

10. Bulyshev, Alexander E., et al. "Computational modeling of three-dimensional microwave tomography of breast cancer." IEEE Transactions on Biomedical Engineering 48.9 (2001): 1053-1056.
    * Trabalho bem resumido. Cita o BIM e o DBIM apenas como revisão de literatura e implementa um CG.

11. Yao, Yuqi, et al. "Frequency-domain optical imaging of absorption and scattering distributions by a Born iterative method." JOSA A 14.1 (1997): 325-342.
    * É um BIM com algumas diferenças no modelo eletromagnético. Mas não fala como escolhe o parâmetro de Tikhonov.

12. Bolomey, Jean‐Charles, and Christian Pichot. "Microwave tomography: from theory to practical imaging systems." International Journal of Imaging Systems and Technology 2.2 (1990): 144-156.
    * É um trabalho de revisão que cita o BIM mas não comenta sobre como escolher o parâmetro de regularização.

13. Otto, Gregory P., and Weng Cho Chew. "Microwave inverse scattering/spl minus/local shape function imaging for improved resolution of strong scatterers." IEEE Transactions on Microwave Theory and Techniques 42.1 (1994): 137-141.
    * Nesse artigo ele fala sobre a regularização de Tikhonov, citando o DBIM e diz que um valor bom pode variar de 0.001 a 10. Mas, o interessante, é que ele sugere que o valor diminua com a convergência do método.

14. Data, Profile Inversion Using Time Domain. "Nonlinear two-dimensional velocity profile inversion using time domain data." IEEE Transactions on Geoscience and Remote Sensing 30.1 (1992): 147.
    * É muito interessante a explicação do parâmetro de regularização nesse artigo. A ideia é que alpha*I é uma correção nos pequenos autovalores de AA*. Pequenos autovalores de AA* são responsáveis por bruscas variações no espaco. A solução x está no espaço spanned por esses autovalores. Então a solução fica com contornos bem suaves quando esses pequenos autovalores que são bem oscilatórios são corrigidos. A ideia é que, no começo das iterações essa correção seja grande, mas que no final, isso vai diminuindo. Conforme esse valor diminui, o subespaço no qual a solução está se expande. Se no começo, esse parâmetro é muito pequeno, isso significa que o subespaço de soluções é de grande dimensão, e nisso, a solução inicial pode estar bem longe da solução verdadeira e a convergência pode levar até à instabilidade ou soluções finais bem longes da solução verdadeira.
    * Em um experimento, ela fala que o valor do parâmetro é reduzido de modo que sua magnitude no final de nove iterações é duas ordens de magnitude menor do que no começo.
    * Não fala explicitamente o valor dos parâmetros.

15. Li, Fenghua, Qing Huo Liu, and Lin-Ping Song. "Three-dimensional reconstruction of objects buried in layered media using Born and distorted Born iterative methods." IEEE Geoscience and Remote Sensing Letters 1.2 (2004): 107-111.
    * Desenvolveu o BIM e o DBIM para um caso um pouco mais específico de reconstrução. Disse que o valor bom do parâmetro de regularização é 0.1 (várias simulações).

16. Chew, Weng Cho, and Q-H. Liu. "Inversion of induction tool measurements using the distorted Born iterative method and CG-FFHT." IEEE Transactions on Geoscience and Remote Sensing 32.4 (1994): 878-884.
    * Tem uma aplicação específica. Definiu o parâmetro de regularização em 10^-20 (várias simulações provavelmente).

17. M. Moghaddam and W. C. Chew, "Study of some practical issues in inversion with the Born iterative method using time-domain data," in IEEE Transactions on Antennas and Propagation, vol. 41, no. 2, pp. 177-184, Feb. 1993, doi: 10.1109/8.214608.
    * BIM continuou convergindo para objetos grandes (8.5*lambda) e altos contrastes
    * Não foi significativamente afetado com até 10% de ruído de fundo.
    * Como ela utilizou multiplas frequências, então não conseguiu recuperar o sigma só pela parte imaginária do contraste (mas dá pra dibrar isso). Aí ela pegou a equação integral e separou em duas: uma da permissividade e outra da condutividade (porque o contraste é a soma dessas duas partes). Assim ela consegue escrever a solução de Tikhonov pra múltiplas frequências e com condutividade. Nisso, são dois parâmetros de regularização.
    * Ela observa que, conforme a condutividade cresce, a reconstrução do epsilon vai ficando mais pobre. Isso porque as correntes de condução começam a dominar sobre as de deslocamento. Com o campo decaindo muito, não chega nada nos receptores. Aí a reconstrução do epsilon fica fraca.
    * Não fala sobre escolha dos parâmetros de regularização.
    * Em cada situação, um cenário somente. Avaliar com poucos cenários pode ser meio que incompleto. E também não é bem quantificado o erro. Isso é a brecha pro meu artigo: analisar a média diferentes indicadores de qualidade com mais rigor estatístico.

18. Moghaddam, Mahta, Weng Cho Chew, and M. Oristaglio. "Comparison of the born iterative method and tarantola's method for an electromagnetic time‐domain inverse problem." International Journal of Imaging Systems and Technology 3.4 (1991): 318-333.
    * Várias conclusões interessantes na comparação entre esses dois métodos.

### Outros trabalhos relevantes sem tantas citações

1. Nie Zaiping, Yang Feng, Zhao Yanwen and Zhang Yerong, "Variational Born iteration method and its applications to hybrid inversion," in IEEE Transactions on Geoscience and Remote Sensing, vol. 38, no. 4, pp. 1709-1715, July 2000, doi: 10.1109/36.851969.
    * Esse é o trabalho inicial do VBIM. A princípio parece meio esquisito de entender porque parece ter uma aplicação específica. Mas deve ser a ideia de ignorar a atualização da função de Green no DBIM.
    * Aqui se sugere um valor para o regularizador de Tikhonov: 0.02*(AA*)

2. W. Zhang and Q. H. Liu, "Three-Dimensional Scattering and Inverse Scattering from Objects With Simultaneous Permittivity and Permeability Contrasts," in IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 1, pp. 429-439, Jan. 2015, doi: 10.1109/TGRS.2014.2322954.
    * Explica o VBIM. A ideia é a mesma do DBIM, só que ignora-se a atualização da função de Green. Acaba que funciona como um meio termo entre o BIM e o DBIM.
    * Não fala qual valor do parâmetro de regularização. Mas é possível que seja do trabalho original.
    * É uma referência para as discretizações da integral em 3D. Mas, possivelmente é igual ao do Pastorino (2010).

3. S. Caorsi, M. Donelli, D. Franceschini and A. Massa, "A new methodology based on an iterative multiscaling for microwave imaging," in IEEE Transactions on Microwave Theory and Techniques, vol. 51, no. 4, pp. 1162-1173, April 2003, doi: 10.1109/TMTT.2003.809677.
    * Aqui é a explicação do Multiscaling: em uma iteração tem um número fixo de elementos. Só que aí esses elementos podem mudar de tamanho para serem menores onde a resolução precisa ser maior. Então, não muda a quantidade de elementos, mas distribui de uma maneira melhor. Depois vai aumentando o número de elementos.

4. Donelli, Massimo, et al. "An integrated multiscaling strategy based on a particle swarm algorithm for inverse scattering problems." IEEE Transactions on Geoscience and Remote Sensing 44.2 (2006): 298-312.
    * Destaco aqui a seguinte afirmação do trabalho: *"The choice of the value of the regular- ization parameter is a crucial and nontrivial problem. But, un- like linear inverse scattering problems for which well-developed mathematical methods and efficient numerical algorithms are already available, the scientific literature does not provide any simple rule for the optimal choice of the regularization coeffi- cient when nonlinear problems are dealt with [25]. Thus, the choice of the regularization parameter has to be properly tuned for every problem in hand on the basis of a calibration process."*
    * Além disso, os autores destacam que uma saída para esse problema de definir o valor do parâmetro de regularização foi proposta por Abubakar et al. onde os resíduos das equações é multiplicado por um termo atualizado ao longo das iterações e calculado pela variação total do mapa.
    * Mas a ideia principal deles é a multiresolução.

### Trabalhos recentes

1. Nithya, N., R. Sivani Priya, and M. S. K. Manikandan. "Performance Analysis of Brain Imaging Using Enriched CGLS and MRNSD in Microwave Tomography." Evolution in Computational Intelligence. Springer, Singapore 191-199.
    * Cita apenas para fins de revisão de literatura.

2. Chen, Yanjin, et al. "Fast Multiparametric Electromagnetic Full-Wave Inversion via Solving Contracting Scattering Data Equations Optimized by the 3-D MRF Model." IEEE Transactions on Microwave Theory and Techniques (2020).
    * Ideia principal: resolver o problema não-linear pelo VBIM, depois classificar pontos de background e pontos de objetos pelo MRF, eliminar pontos de background e ir para próxima iteração com o problema reduzido.
    * Comentário interessante sobre aplicação de deep learning no problema: pode ser utilizado para acelerar iterações, mas sempre é necessário fazer um treinamento para cada modelo. No caso de problemas eletricamente largos (provavelmente está falando de altos contrastes), fica mais lento e custoso, o que é complicado para problemas 3D.
    * MRF é encontrado em abordagens bayesianas de EISPs
    * Não fala o valor do parâmetro regularizador de Tikhonov, mas está citado nos trabalhos anteriores.

3. Akdogan, Riza Erhan, and Yasemin Altuncu. "Reconstruction of 3D Objects Buried Under Into Half-Space by Using Variational Born Iterative Method." 2019 23rd International Conference on Applied Electromagnetics and Communications (ICECOM). IEEE.
    * A ideia principal é mais a aplicação do VBIM pra esse tipo de caso específico. Trabalho resumido.
    * Sem referência para o valor do parâmetro de regularização.

4. Wang, Jianwen, et al. "Simulation of 3-D Electromagnetic Scattering and Inverse Scattering by Arbitrary Anisotropic Dielectric Objects Embedded in Layered Arbitrary Anisotropic Media." IEEE TRANSACTIONS ON ANTENNAS AND PROPAGATION 68.8 (2020): 6473-6478.
    * Usa um VBIM. No método, não é nada de novo. A questão é aplicação em meios de camada e anisotrópico.

5. Qin, Yingying, et al. "Microwave breast imaging with prior ultrasound information." IEEE Open Journal of Antennas and Propagation (2020).
    * A ideia principal é: trazer dados de ultrasom que separam as regiões de cada tecido. Aí você adiciona isso num DBIM ou CSI.
    * Não deu informações sobre valor de regularizador de Tikhonov no DBIM.

8. Wang, Shoudong, and Ru-Shan Wu. "Multi-Frequency Contrast Source Inversion for Reflection Seismic Data." COMMUNICATIONS IN COMPUTATIONAL PHYSICS 28.1 (2020): 207-227.
    * A ideia principal é: um CG com múltiplas frequências e com uma certa adaptação ao caso que é um tipo de GPR.

9. Burfeindt, Matthew J., and Hatim F. Alqadah. "Qualitative inverse scattering for sparse-aperture data collections using a phase-delay frequency variation constraint." IEEE Transactions on Antennas and Propagation (2020).
    * A ideia principal é uma formulação de Linear Sampling (qualitativo) pra casos do problema em que "os dados são espacialmente esparsos" (falta de fontes ou receptores). Pra isso, um tipo de restrição é adicionada relacionado fase, frequência e comprimento do caminho entre transmissores e pixels da imagem.
    * Na função objetivo do problema tem um termo para minimizar a norma das variáveis, bem parecido com o regularizador de Tikhnov. No caso, eles determinam o valor pela Curva-L.

10. Xu, Kuiwen, et al. "Deep Learning-Based Inversion Methods for Solving Inverse Scattering Problems with Phaseless Data." IEEE Transactions on Antennas and Propagation (2020).
    * É um trabalho de rede neural que o Chen assina como último autor. Parece que o algoritmo chega a utilizar alguns esquemas de inversão. Apresenta resultados bons. Os experimentos parecem ser de até 0.5 de contraste.
    * O primeiro experimento é com círculos somente. Talvez seja depende da geometria.

11. Chu, Yanqing, et al. "Multiplicatively Regularized Iterative Updated Background Inversion Method for Inverse Scattering Problems." IEEE Geoscience and Remote Sensing Letters (2020).
    * Além da regularização de Tikhonov, é citado também a minimização da variação total e esparcidade.
    * A ideia de regularização multiplicativa serve para diminiuir o espaço de busca da solução. Teoricamente, a ideia é obter uma função objetivo que é a multiplicação da soma de resíduos com a variação total.
    * Na parte sintética, os resultados são apenas três casos: austria, breast e 
through-wall.
    * Experimentação com até 2.0 de contraste.

## Citações ao DBIM

### Até 160 citações

1. Van Den Berg, Peter M., and Ralph E. Kleinman. "A contrast source inversion method." Inverse problems 13.6 (1997): 1607.
    * É o paper original do CSI.
    * Tem algumas formulinhas para solução incial.
    * Função objetivo é a soma ponderada com os pesos 

2. Hawrysz, Daniel J., and Eva M. Sevick-Muraca. "Developments toward diagnostic breast cancer imaging using near-infrared optical measurements and fluorescent contrast agents1." Neoplasia 2.5 (2000): 388-417.
    * Quando fala da regularização de Tikhonov, é dito que o parâmetro pode ser determinado arbitrariamente ou pelo método de Levenberg-Marquardt. Mas esse último é pra isso mesmo?
    * São citados dois trabalhos de metodologia estatística de escolha do parâmetro [53] e [28]

2. Eppstein, Margaret J., et al. "Biomedical optical tomography using dynamic parameterization and Bayesian conditioning on photon migration measurements." Applied Optics 38.10 (1999): 2138-2150.
    * Tem uma modelagem física diferente. Inclui outros tipos de informação sobre o modelo: fluorescência, quantum.
    * Não vi a parte que pode falar de regularizador de Tikhonov. Até cita o BIM e o DBIM, mas apenas como referências na literatura.

2. Pogue, Brian W., et al. "Spatially variant regularization improves diffuse optical tomography." Applied optics 38.13 (1999): 2950-2961.
    * A ideia aqui é bem legal: uma variação espacial do parâmetro de regularização. Porque o número de valores é o mesmo de pixels. Em alguns lugares pode ser mais interessante ser baixo e outros ser pequeno.

3. Abubakar, Aria, Peter M. Van den Berg, and Jordi J. Mallorqui. "Imaging of biomedical data using a multiplicative regularized contrast source inversion method." IEEE Transactions on Microwave Theory and Techniques 50.7 (2002): 1761-1771.
    * É o artigo original do MR-CSI. Conforme dito anteriormente, ele multiplica a soma ponderada de resíduos pela regularização de variação total. Essa função de regularização de variação total é igual a 1 no ponto ótimo. Ele começa grande porque depende do erro. Mas a cada iteração vai diminuindo.
    * A complexidade computacional é igual a resolver dois problemas diretos usando o CG a cada iteração

4. Habashy, Tarek M., and Aria Abubakar. "A general framework for constraint minimization for the inversion of electromagnetic measurements." Progress in electromagnetics Research 46 (2004): 265-312.
    * Assumindo que é possível fazer uma parametrização dos objetos na imagem, o trabalho traz uma forma geral de algoritmos de Newton que promete ser melhor do que métodos do gradiente (se bem que usualmente o método de Newton depende de derivada).
    * Essa abordagem geral do método de Newton conta com: um tipo de busca em linha (unidimensional?), transformação não-linear sobre parâmetros pra que eles estejam dentre de um limite físico; termo de penalização; forma de determinar termos na função objetivo.

5. Ammari, Habib, et al. "MUSIC-type electromagnetic imaging of a collection of small three-dimensional inclusions." SIAM Journal on Scientific Computing 29.2 (2007): 674-709.

6. Semenov, Serguei Y., and Douglas R. Corfield. "Microwave tomography for brain imaging: Feasibility assessment for stroke detection." International Journal of Antennas and Propagation 2008 (2008).
    * É mais estudo de caso.

7. Wang, Zhisong, Jian Li, and Renbiao Wu. "Time-delay-and time-reversal-based robust capon beamformers for ultrasound imaging." IEEE transactions on medical imaging 24.10 (2005): 1308-1322.
    * Só cita o DBIM de referência mesmo.

8. Abubakar, Aria, and Peter M. van den Berg. "Iterative forward and inverse algorithms based on domain integral equations for three-dimensional electric and magnetic objects." Journal of computational physics 195.1 (2004): 236-262.
    * É aplicação do MR-CSI pra um problema 3D onde condutividade, permissividade e permeabilidade podem variar.
    * Cita o DBIM só de referência mesmo.

9. Semenov, Serguei Y., et al. "Microwave-tomographic imaging of the high dielectric-contrast objects using different image-reconstruction approaches." IEEE transactions on Microwave Theory and Techniques 53.7 (2005): 2284-2294.
    * É mais um estudo de caso mesmo pra situações de alto contraste. Só cita de passagem o DBIM.

10. Sakamoto, Takuya, and Toru Sato. "A target shape estimation algorithm for pulse radar systems based on boundary scattering transform." IEICE transactions on communications 87.5 (2004): 1357-1365.
    * É um problema de identificar contorno de espalhadores. Tem uma metodologia bem diferente. Apenas cita de passagem o DBIM.

11. Cui, Tie Jun, et al. "Inverse scattering of two-dimensional dielectric objects buried in a lossy earth using the distorted Born iterative method." IEEE Transactions on Geoscience and Remote Sensing 39.2 (2001): 339-346.
    * Definiu o parâmetro como 10^{-10}
    * Poucas experimentações

12. Isernia, Tommaso, Vito Pascazio, and Rocco Pierri. "On the local minima in a tomographic imaging technique." IEEE Transactions on Geoscience and Remote Sensing 39.7 (2001): 1596-1607.
    * Cita como um exemplo de método que não cai em falsa solução sob certas condições.

13. Chandra, Rohit, et al. "On the opportunities and challenges in microwave medical sensing and imaging." IEEE transactions on biomedical engineering 62.7 (2015): 1667-1682.
    * Artigo de revisão bem amplo sobre métodos. Por isso não vai a fundo nos métodos.

14. Zakaria, Amer, Colin Gilmore, and Joe LoVetri. "Finite-element contrast source inversion method for microwave imaging." Inverse Problems 26.11 (2010): 115010.
    * Basicamente, aqui, a equação diferencial que é discretizada com o método dos elementos finitos. Tem resultados muito parecidos com os do CSI normal.

15. Ye, Xiuzhu, and Xudong Chen. "Subspace-based distorted-Born iterative method for solving inverse scattering problems." IEEE Transactions on Antennas and Propagation 65.12 (2017): 7224-7232.
    * Basicamente, a ideia do SOM de representar a função de Green em termos da decomposição SVD é acoplada ao DBIM. Isso traz mais convergência.
    * Ele continua empregando o regularizador de Tikhonov, embora a questão de escolher o número de valores singulares principais também é determinante para a regularização do problema. O de Tikhonov é mais sensível. Neste trabalho, foi escolhido de forma empírica. E o número de valores singulares não precisa ser determinado ao ótimo valor, mas pode ser escolhido dentro de um intervalo amplo de valores.
    * Experimentação só com o Austria. O Chen é fissurado nisso.
    * Seria interessante ler com mais atenção para saber implementar.

### Trabalhos mais recentes

1. Huang, Xingguo. "Integral Equation Methods With Multiple Scattering and Gaussian Beams in Inhomogeneous Background Media for Solving Nonlinear Inverse Scattering Problems."
    * É um DBIM numa formulação na qual não consegui entender.
    * Parece que utiliza o regularizador de Tikhonov, mas não explica como é definido.

2. Ye, Xiuzhu, et al. "An Inhomogeneous Background Imaging Method Based on Generative Adversarial Network." IEEE Transactions on Microwave Theory and Techniques (2020).
    * Ideia: deep learning network para background inomogeneo; dividido em duas etapas: (i) roda um método primeiro (tipo o S-DBIM) e (ii) aplica o resultado na rede pra reconstrução.
    * Tem que dar essa ajuda se não a rede não guenta tanta não-linearidade.

3. Khoshdel, Vahab, et al. "Full 3D Microwave Breast Imaging Using a Deep-Learning Technique." Journal of Imaging 6.8 (2020): 80.
    * Um trabalho que aplica deep learning num modelo pra mama e acopla com outros algoritmos.

4. Elkattan, Mohamed, and Aladin Kamel. "Characterization of electromagnetic parameters through inversion using metaheuristic technique." Inverse Problems in Science and Engineering (2020): 1-19.
    * Aplicação de SA com um problema de duas camadas.

5. Zhang, Lu, et al. "Learning-based Quantitative Microwave Imaging with A Hybrid Input Scheme." IEEE Sensors Journal (2020).
    * Três etapas: (i) resolve qualitativamente com direct sampling method; (ii) resolve quantitativamente com Backpropagation; e (iii) termina de reconstruir com a rede neural.
    * Cara, dá a entender que ele usa 4000 imagens de treino que são 4000 formas dos dígitos de 0 a 9 e depois avalia com 1000 que também são esses dígitos, só que colocados de forma de diferente. Se for isso, isso não está muito especificado no padrão das imagens?
    * Atá, depois eles fazem testes com dois casos diferentes (único valor de contraste);

## Comparação de algoritmos

Artigo: *Beiranvand, Vahid, Warren Hare, and Yves Lucet. "Best practices for comparing optimization algorithms." Optimization and Engineering 18.4 (2017): 815-848.*
  * Foca apenas na comparação de algoritmos que não são multi-objetivos nem contém paralelização.
  * Embora os problemas vão ser falados como um de otimização contínua e irrestrita, transpor os conceitos para outros casos não vai ser problema.
  * Passos propostos pelo artigo:
    1. Deixar claro a razão para o benchmarking
        - As razões para se fazer um benchmarking podem ser: (i) ajudar a escolher um melhor algoritmo para um determinado problema; (ii) mostrar o valor de um novo algoritmo comparado a um mais clássico na literatura; (iii) comparar uma nova versão de algoritmo com as anteriores; (iv) avaliar a performance de um algoritmo quando diferentes opções de configurações são disponíveis.
        - Mas é muito importante prestar atenção no contexto: no caso (i), todos os testes tem de ser exemplos do mesmo problema; no caso (ii), tem que se pensar os testes levando exatamente em consideração o que difere um algoritmo do outro, tipo, o outro algoritmo tem que ser pra resolver o mesmo problema.
        - Outra coisa ~e se perguntar qual aspecto do algoritmo é mais importante: ser rápido mesmo podendo retornar soluções infactíveis? Resolver cada problema ou a média da performance? O objeto é encontrar um otimizador global ou um local?
    2. Selecionar o conjunto de testes
        - Um teste é uma entrada (que pode ser composta de uma função a ser otimizada com um conjunto de restrições, região factível e pontos de começo). Um conjunto de entradas formam um conjunto de teste. Um benchmarking tem significado quando os algoritmos são avaliados no mesmo conjunto de teste com os mesmos medidores de performance.
        - Três tipos de casos: problemas do mundo real (que as vezes podem ser poucos, mas podem ser a melhor opção em estudos bem aplicados), problemas pré-gerados e problemas gerados aleatoriamente (os dois últimos podem ser muitos, mas que talvez não vão dizer muito sobre a realidade).
        - Cinco tipos de deficiências em conjuntos de testes: (i) poucos problemas; (ii) pouca variação em dificuldade ou dificuldade demais na qual não é possível tirar informações úteis; (iii) problemas sem soluções conhecidas (em problemas reais pode ser inevitável); (iv) viés no ponto de partida dos algoritmos; (v) estruturas escondidas (números arredondados).
        - Prestar atenção: (i) conjunto de teste com poucos problemas é estudo de caso ou prova de conceito, mas não benchmarking; (ii) pelo menos um grupo de problemas fáceis e um grupo de problemas difíceis; (iii) pelo menos uma porção ter soluções conhecidas; (iv) todos algoritmos devem receber a mesma quantidade de informação de entrada e garantir que não há um pressuposto que é respeitado somente por um dos algoritmos.
    3. Realizar os experimentos
        - Dois fatores são preponderantes na execução dos experimentos: fatores ambientais (processador, memória, SO, implementação etc) e fatores algorítmicos (tudo que independe da implementação e onde está rodando. Idealmente, nós queremos apenas a segunda situação. Mas é natural esperar que fatores ambientais influenciem. De qualquer maneira, as coisas têm que ser feitas para que apenas os fatores algorítmicos sejam preponderantes.
        - São três categorias de medida de desempenho: (i) eficiência (número de avaliações, tempo de execução, uso de memória); (ii) confiabilidade (taxa de sucesso, número de restrições violadas e percentual de soluções globais encontradas) e (iii) qualidade da saída (resultado do custo-fixo da solução, tempo até alcançar um alvo fixo, precisão computacional).
          1. Eficiência pode ser analisada como: (a) número de avaliações (que não necessariamente pode ser questão da função-objetivo mas também alguma sub-rotina específica); (b) tempo de execução e (c) outras medidas que podem ficar a critério do tipo de algoritmo ou do tipo de problema.
          2. Confiabilidade e robustez: manter um bom desempenho sobre um amplo número de problemas. Isso pode involver coisas como a taxa de sucesso (quantidade de vezes que um algoritmo atinge uma determinada tolerância dado um tempo-limite), a qual pode ter influência em casos onde o problema tem mínimos locais ou não. Além disso, algoritmos não-determinísticos devem ter atenção sobre: quantas vezes o algoritmo vai ser repetido e se vai ser feito uma média ou o pior valor dos casos. Múltiplos pontos de começo pode ser relevante para estudar a sensibilidade.
          3. Qualidade da saída do algoritmo pode ser feita de duas formas: (a) quando se conhece a solução ótima, e aí podemos comparar tanto do ponto de vista de distância de solução quando do ponto de vista de erro da avaliação da função objetivo. Isso pode ser expresso de diversas maneiras, inclusive as violações nas restrições; (b) quando não se conhece a solução ótima e você pode comparar com uma solução utópica ou uma avaliação utópica.
          4. Ajuste de parâmetros e critérios de parada: deve haver um esforço para que os algoritmos devam ter o mesmo critério de parada. Do contrário, isso deve ser bem descatado e conclusões devem levem isso em consideração. Já no caso de algoritmos que dependem de muitos parâmetros, pode-se projetar experimentos para definir esses parâmetros e usar outras técnicas já conhecidas.
    4. Analisar e reportar os resultados:
        * Além de analisar os resultados estudando as médias, outras formas de reportar os resultados são:
          1. Tabelas: cuidar para não ficar densas. Resumir o que for possível.
          2. Gráficos no geral: histogramas, boxplots, plots de trajetória, convergência e tempo de execução por tamanho do problema.
          3. Performance profile: probabilidade que um algoritmo resolve um problema (eixo y) dado um limite de tempo (eixo x) [IMPLEMENTAR ISSO POSTERIORMENTE]
          4. Accuracy profile: proporção de problemas (eixo y) que o algoritmo é capaz de obter uma solução dentro de uma acurácia de tantos por cento da melhor solução (eixo-x). **[IMPLEMENTAR ISSO POSTERIORMENTE]**
          5. Data profiles: a porcentagem de problemas que podem ser resolvidos (considerando um valor de tolerância fixo) dentro de um limite de k avaliações. **[IMPLEMENTAR ISSO POSTERIORMENTE]**

