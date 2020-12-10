Nesta pasta vocês vão encontrar três tipos de experimentos:

# imsa_pso

Neste arquivo, vocês vão encontrar os resultados de um experimento que estava devendo que era deixar o algoritmo do IMSA-PSO ser rodado com o número de iterações do artigo original e o uso da informação *a priori* dos limites inferiores e superiores de contraste. Estas duas características foram consideradas no artigo original e eu ainda não tinha considerado nos experimentos. Além disso, eu considerei somente uma e duas iterações de escala (redução da imagem). No artigo eram seis, mas eu fiz até duas por enquanto por questões de tempo (pra rodar 30 execuções onde cada iteração de escala tem 20 mil iterações do PSO demoraria muito).

Quando vocês forem olhar, vocês vão que os resultados não foram tão ruins. Pode ser que funcione se eu colocar mais iterações.

# mopso

Aqui eu fiz alguns testes com versões multi-objetivo do problema. Eu escrevi diferentes divisões de um problema bi-objetivo e executei uma vez o PSO.

Lá no arquivo vai estar descrito a implementação do PSO multiobjetivo. Mas, de uma maneira geral, o que eu fiz foi, ao invés de armazenar uma única solução global, eu armazenei um arquivo onde eu ia atualizando igual no NSGA-II (usando *crowding distance*) e, na hora de atualizar as velocidades, fazia escolhas aleatórias.

Basicamente, o que é testado em cada experimento é:

1. Combinação de objetivos: resíduo da equação de dados & resíduo da de estados;
2. Combinação de objetivos: resíduo da equação de dados & funcional de Tikhonov;
3. Combinação de objetivos: resíduo da equação de dados + de estados & funcional de Tikhonov
4. Combinação de objetivos: resíduo da equação de dados + funcional de Tikhonov & resíduo da equação de estados;
5. Mesmo de 2 só que, ao invés de aplicar o critério de *crowding distance*, eu aplico um critério de substituição em caso da descoberta de soluções menos extremas do ponto de vista do funcional de Tikhonov.

De um modo geral, não deu muito certo. Talvez é necessário mudar a representação de soluções e a estrutura de vizinhança para depois pensar essa questão dos objetivos.

# wrm

Nesses experimentos, eu mudo a representação das soluções, o que significa basicamente trocar a forma de discretização das funções de contraste e de campo. Basicamente, dentro do contexto do Método dos Resíduos Ponderados (WRM), eu utilizo o método da Colocação em relação às funções peso (discretização do espaço de campo espalhado) e uso diferentes combinações de discretização e função mono-objetivo:

1. Contraste: cossenos; Campo elétrico: exponenciais complexas e coeficientes complexos; Minimização: resíduos dos dados;
2. Contraste: cossenos; Campo elétrico: exponenciais complexas e coeficientes reais; Minimização: resíduos dos dados;
3. Mesmo de 1, só que com minimização dos resíduos de dados + regularização de Tikhonov
4. Contraste: bilinear; Campo elétrico: exponenciais complexas e coeficientes complexos; Minimização: resíduos dos dados + regularização de Tikhonov;

De uma maneira geral, até
