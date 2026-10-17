// Gera relatorio/Relatorio_Final.docx a partir do texto abaixo + tabelas (CSV) e figuras (PNG)
// produzidas por `python -m analysis.make_figures`. Trechos sem resultado viram [A PREENCHER].
// Uso: node analysis/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ImageRun, AlignmentType, BorderStyle, ShadingType, Footer, PageNumber, LevelFormat, PageBreak,
} = require("docx");

const REL = path.resolve(__dirname, "..", "relatorio");
const W = 9026; // largura útil A4 com margens de 1" (DXA)
let figN = 0;
let tabN = 0;

// ---------- helpers ----------
const run = (t, o = {}) => new TextRun({ text: t, ...o });
const p = (t, o = {}) => new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED, children: [run(t, o)] });
const rich = (parts) => new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED, children: parts });
const eq = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 }, children: [run(t, { italics: true })] });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [run(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [run(t)] });
const bullets = (items) => items.map((t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60 }, children: [run(t)] }));
const todo = (t) => new Paragraph({ spacing: { after: 120 }, children: [run(`[A PREENCHER — ${t}]`, { italics: true, highlight: "yellow" })] });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run(t, { italics: true, size: 20 })] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(headers, rows, widths, title) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, head) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "E8EEF7", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [run(String(t), { bold: head, size: 20 })] })],
  });
  const out = [];
  if (title) out.push(caption(`Tabela ${++tabN} — ${title}`));
  out.push(new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }))],
  }));
  out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  return out;
}

const fmt = (x) => Number(x).toFixed(2).replace(".", ",");

function csvRows(file) {
  const f = path.join(REL, "tabelas", file);
  if (!fs.existsSync(f)) return null;
  const lines = fs.readFileSync(f, "utf8").trim().split(/\r?\n/);
  if (lines.length < 2) return null;
  return lines.slice(1).map((l) => l.split(","));
}

// tabela média ± desvio (colunas: tag, n_seeds, mean, std)
function resultTable(file, title, labelHeader = "Configuração") {
  const rows = csvRows(file);
  if (!rows) return [todo(`tabela ${file} (execute os experimentos e \`python -m analysis.make_figures\`)`)];
  return table([labelHeader, "Sementes", "Média", "Desvio padrão"],
    rows.map((r) => [r[0], r[1], fmt(r[2]), fmt(r[3])]), [3626, 1400, 2000, 2000], title);
}

function figure(file, text, maxW = 520) {
  const f = path.join(REL, "figuras", file);
  if (!fs.existsSync(f)) return [todo(`figura ${file}`)];
  const b = fs.readFileSync(f);
  const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
      children: [new ImageRun({ type: "png", data: b, transformation: { width: maxW, height: Math.round((maxW * h) / w) },
        altText: { title: text, description: text, name: file } })] }),
    caption(`Figura ${++figN} — ${text}`),
  ];
}

const discussionPrompts = [
  "As hipóteses iniciais foram confirmadas?",
  "O comportamento dos agentes faz sentido com base na teoria?",
  "Que limitações experimentais foram observadas?",
  "Quais melhorias poderiam ser feitas (hiperparâmetros, arquitetura da rede, técnica de exploração etc.)?",
];
const conclusionPrompts = [
  "O que foi comprovado ou refutado?",
  "Que lições o experimento trouxe sobre o aprendizado por reforço?",
  "Que próximos passos podem ser investigados?",
];

const T = {
 "r1": {
  "frozenlake": [
   "Como a recompensa é 0 ou 1, o retorno médio de avaliação é a taxa de sucesso da política gulosa em 1 000 episódios. Com α = 0,05 e 0,1, os dois algoritmos ficam entre 0,70 e 0,75 (Q-learning: 0,74 e 0,70; SARSA: 0,75 e 0,75), valores muito abaixo de 1 porque o agente só se move na direção desejada em um terço das tentativas. Com α = 0,5 o resultado piora e a variância entre sementes explode (Q-learning 0,57 ± 0,25; SARSA 0,41 ± 0,29). Nos últimos 10% dos episódios de treino, o retorno online é praticamente igual para os dois (0,52), e ambos levam cerca de 9,5 mil episódios (mediana) para atingir 90% do patamar final."
  ],
  "frozenlake_noslip": [
   "Sem escorregar, o Q-learning chega à política ótima (retorno 1,0) em todas as 30 execuções (10 sementes × 3 valores de α). O SARSA também chega com α = 0,1, mas uma semente falhou com α = 0,05 e outra com α = 0,5 (retorno 0 nessas sementes, média 0,90 ± 0,30). Com recompensa esparsa, uma política gulosa que não encontra o objetivo repete ações até o limite de 100 passos. A hipótese de que o SARSA, por incorporar a exploração ao valor de Q, deixa mais estados sem valor positivo é plausível, mas não foi testada."
  ],
  "cliffwalking": [
   "O resultado clássico se reproduz. Na avaliação gulosa, o Q-learning obtém −13 em todas as sementes e para todos os α, o caminho ótimo de 13 passos rente ao penhasco; o SARSA obtém −17, o caminho mais longo e seguro pela linha superior do grid (mapas de calor abaixo). Durante o treino, nos últimos 10% dos episódios, o Q-learning tem retorno online de −30,7 ± 6,5 contra −19,6 ± 1,0 do SARSA, pois cai no penhasco quando a exploração ε-greedy o empurra para a borda. Com α = 0,5, uma semente do SARSA terminou com política gulosa em ciclo (−1 000, o teto de passos da avaliação), o que explica a média −115,5 ± 294,8. Os dois algoritmos convergem rápido (mediana de 187 episódios para o Q-learning e 153 para o SARSA, para chegar a 90% do patamar final)."
  ],
  "taxi": [
   "O Q-learning chega à mesma política em todas as sementes e para todos os α (retorno 7,86 com desvio de 0,007). O SARSA fica ligeiramente abaixo com α pequeno (7,83 com α = 0,05; 7,77 com α = 0,1) e falha com α = 0,5 (−29,9 ± 38,8; 8 das 10 sementes com retorno negativo). O retorno online final é ≈ 5,2 para os dois, mais baixo que o da avaliação por causa de ε = 0,05. O Q-learning precisou de menos episódios que o SARSA para atingir 90% do patamar final (mediana de 2,2 mil contra 2,9 mil)."
  ],
  "blackjack": [
   "Todos os retornos de avaliação ficam entre −0,03 e −0,07, como esperado para um jogo com vantagem do cassino, e o retorno online nos últimos 10% dos episódios é −0,08 para os dois algoritmos. Com 1 000 episódios de avaliação, o erro padrão de cada média é de cerca de 0,03 (o retorno de um episódio tem desvio próximo de 1), portanto as diferenças entre Q-learning e SARSA no mesmo α não são distinguíveis do ruído. A única tendência consistente nos dois algoritmos é que α = 0,5 é pior (−0,07 e −0,065) que α = 0,05 (−0,032 e −0,034)."
  ],
  "disc": [
   "Hipóteses. H1 foi confirmada: no CliffWalking o Q-learning aprende o caminho ótimo (−13), mas tem retorno online pior durante o treino (−30,7 contra −19,6 do SARSA), e o SARSA aprende o caminho seguro (−17). H2 foi confirmada em parte: no FrozenLake escorregadio a taxa de sucesso não passa de ≈ 0,75 e a variância é maior que na versão sem escorregar, onde o Q-learning acerta em todas as execuções; o SARSA teve duas falhas isoladas nessa versão. H3 só foi confirmada na parte do α alto: α = 0,5 foi o pior e o mais variável em quatro dos cinco ambientes (FrozenLake, CliffWalking e Taxi, no SARSA, e Blackjack). A parte \"α pequeno converge mais devagar\" não foi medida (comparamos apenas o resultado final, não a velocidade por α). H4 é consistente com os dados (retornos levemente negativos).",
   "Teoria. O comportamento observado segue o esperado: o Q-learning é off-policy e aprende o valor da política gulosa, por isso encontra o caminho ótimo mesmo explorando com ε-greedy; o SARSA é on-policy e aprende o valor da política que realmente executa, com a exploração incluída, e por isso evita a borda do penhasco. O mesmo mecanismo pode explicar por que o SARSA é mais sensível a α alto em Taxi e CliffWalking, mas isso não foi verificado isoladamente.",
   "Limitações. São 10 sementes por configuração e nenhum teste estatístico formal; as falhas isoladas de uma semente (por exemplo, FrozenLake sem escorregar e CliffWalking com α = 0,5) não permitem separar efeito do algoritmo de acaso. A avaliação com 1 000 episódios tem ruído de cerca de ±0,03 no Blackjack e a política em ciclo no CliffWalking é limitada a 1 000 passos. O decaimento de ε (linear em 50% dos episódios, até 0,05) e α constante foram os mesmos em todos os ambientes, sem ajuste por ambiente.",
   "Melhorias. Usar mais sementes e intervalos de confiança; decair α e ε até valores menores; avaliar o SARSA com ε reduzido gradualmente (para aproximar o valor da política gulosa); investigar separadamente as falhas do SARSA com α alto; ajustar o orçamento de episódios por ambiente."
  ],
  "concl": [
   "Foi comprovado que o Q-learning(0) e o SARSA(0) tabulares resolvem os quatro ambientes com a mesma política de comportamento, e que a diferença entre eles (off-policy × on-policy) aparece de forma nítida no CliffWalking. Foi refutada a ideia de que o α alto seja uma escolha segura: α = 0,5 degrada ou torna instável o resultado na maioria dos ambientes.",
   "A lição principal é que a métrica importa: o retorno durante o treino e o retorno da política gulosa final contam histórias diferentes (CliffWalking), e as conclusões sobre algoritmos exigem várias sementes, porque falhas isoladas são comuns.",
   "Próximos passos: medir a velocidade de convergência por α, testar decaimento de α, repetir com mais sementes e comparar com Expected SARSA e Double Q-learning."
  ]
 },
 "r2": {
  "analysis": [
   "O agente aleatório obteve retorno médio de 23,7 (desvio 12,3), dentro da faixa prevista em H1 (20 a 25 passos). A política heurística sem ruído atingiu o limite de 500 passos nos 100 episódios (desvio 0), confirmando H2. Com o ActionWrapper, o retorno médio cai de forma monótona: 490,9 (ε = 0,1), 386,1 (ε = 0,3) e 141,6 (ε = 0,5), com desvios de 35,7, 123,6 e 96,6; portanto H3 também se confirma. O desvio cresce de ε = 0 até ε = 0,3, quando alguns episódios falham cedo e outros chegam a 500, e diminui em ε = 0,5, quando quase todos terminam cedo. Mesmo com ε = 0,5 o retorno (141,6) é cerca de seis vezes o do agente aleatório: como a ação sorteada coincide com a da política em metade das vezes, a fração de ações realmente trocadas é ε/2. Os vídeos de um episódio do agente aleatório e da política heurística estão em runs/cartpole_demo/videos (não versionados)."
  ],
  "disc": [
   "Hipóteses. H1, H2 e H3 foram confirmadas (23,7; 500; e queda monótona do retorno com ε). A degradação não é linear: com ε = 0,1 (cerca de 5% das ações trocadas) a perda é de apenas 2%; com ε = 0,3 é de 23%; com ε = 0,5 é de 72%. Isso é coerente com a teoria: o CartPole é instável e tolera erros isolados, mas, quando os erros se acumulam, o pêndulo sai da região em que a política consegue recuperá-lo.",
   "Relação com o aprendizado por reforço. O experimento mostra o custo de desempenho da exploração ε-greedy durante o treino, e por que o desempenho de treino (com exploração) e o de avaliação (sem exploração) devem ser reportados separadamente, como é feito na Parte III.",
   "Limitações. São 100 episódios por condição, uma única política heurística e ruído apenas na ação; não há aprendizado nesta parte. Os episódios usam as mesmas sementes de reinício em todas as condições, o que torna a comparação pareada nos estados iniciais.",
   "Melhorias. Varrer ε em uma grade mais fina, calcular intervalos de confiança, comparar com ε decrescente e com ruído de ação correlacionado no tempo."
  ],
  "concl": [
   "Foi comprovado que a API do Gymnasium permite compor ambiente, agente e wrappers com pouco código, e que uma pequena perturbação de ação (ε ≤ 0,1) quase não afeta uma boa política no CartPole, enquanto ε ≥ 0,3 já a degrada de forma importante. Nenhuma hipótese foi refutada.",
   "A lição é que a exploração tem custo imediato e que a avaliação de um agente deve ser feita sem exploração. Próximo passo: substituir a política heurística por um agente que aprende, o que é feito na Parte III."
  ]
 },
 "r3": {
  "cartpole": [
   "Retorno de avaliação final (100 episódios gulosos): DQN 347,2 ± 160,2 e Double DQN 495,5 ± 9,1 (o máximo é 500). Por semente, o DQN obteve 500; 134,8; 495,2; 172,5 e 433,6, e o Double DQN 477,4; 500; 500; 500 e 500: duas sementes do DQN terminaram abaixo de 175, enquanto a pior do Double DQN ficou em 477. O DQN chegou primeiro a avaliações ≥ 475 (entre 35 e 60 mil passos contra 50 a 60 mil), mas não manteve. A média das 5 últimas avaliações periódicas é 376 ± 131 para o DQN e 416 ± 61 para o Double DQN, ou seja, o Double DQN tem menor variância, o que vai no sentido de H1. A curva de treino (média móvel de 20 episódios, com ε = 0,05) oscila muito nos dois algoritmos e termina em ordem inversa à da avaliação final (DQN ≈ 340, Double DQN ≈ 220), o que mostra que o desempenho de treino com exploração residual e o da política gulosa final são medidas diferentes. Com 5 sementes e desvios dessa magnitude, não é possível afirmar diferença significativa."
  ],
  "cartpole_prof": [
   "Com lr = 1e-4 (valor do notebook do professor; as demais configurações são as do CartPole principal) o aprendizado é mais lento: avaliações ≥ 475 só aparecem entre 70 e 100 mil passos no DQN, e entre 75 e 100 mil no Double DQN, onde uma semente nunca chega lá. A avaliação final é de 499,6 ± 0,7 para o DQN e 448,7 ± 90,7 para o Double DQN (268,3 na pior semente). A ordem entre os algoritmos se inverte em relação a lr = 1e-3, o que indica que a diferença observada com 5 sementes é da mesma ordem que a variação entre configurações e sementes. Nesta configuração, o lr menor deu o resultado mais estável para o DQN."
  ],
  "mountaincar": [
   "Contrariando H2, o DQN puro resolveu o MountainCar: as 5 sementes de cada algoritmo terminam com retorno médio entre −100 e −118 (o limite do episódio é −200). DQN: −102,6 ± 1,4; Double DQN: −109,0 ± 6,2. O retorno de treino fica em −200 até cerca de 30 mil passos, cai rapidamente até ≈ 100 mil passos e se estabiliza perto de −110 a −120, com leve piora nos últimos passos. O primeiro episódio de treino que chegou ao topo ocorreu entre 9,6 mil e 53,5 mil passos, quase sempre depois de ε ter decaído até 0,05 (aos 10 mil passos); portanto, o primeiro sucesso parece vir das ações gulosas de uma rede ainda pouco treinada, e não da exploração aleatória (explicação plausível, não testada). O DQN foi um pouco melhor que o Double DQN (−108,2 ± 7,7 contra −117,9 ± 7,9 nas 5 últimas avaliações), mas com 5 sementes e faixas sobrepostas a diferença não é conclusiva."
  ],
  "mountaincar_shaped": [
   "Com recompensa modelada por energia, o primeiro sucesso em treino ocorre mais cedo (entre 9 mil e 50 mil passos, mediana ≈ 22 mil para o DQN, contra ≈ 36 mil sem modelagem) e as avaliações ≤ −150 aparecem entre 30 e 60 mil passos, contra 40 a 80 mil sem modelagem. O resultado final, porém, é equivalente: DQN −105,8 ± 9,4 e Double DQN −106,9 ± 4,5, de modo que a modelagem acelerou o início do aprendizado sem melhorar o desempenho final."
  ],
  "disc": [
   "Hipóteses (Classic Control). H1 não pôde ser confirmada nem refutada: no CartPole com lr = 1e-3 o Double DQN foi melhor (495,5 ± 9,1 contra 347,2 ± 160,2) e mais estável, mas com lr = 1e-4 a ordem se inverteu (DQN 499,6 ± 0,7; Double DQN 448,7 ± 90,7), e no MountainCar o DQN foi ligeiramente melhor (−102,6 contra −109,0). As diferenças são da ordem do desvio entre sementes. H2 foi refutada: o DQN puro resolveu o MountainCar em todas as sementes, e a recompensa modelada só acelerou o início do aprendizado.",
   "Teoria. O Double DQN corrige a superestimação causada pelo máximo sobre ações ruidosas. Em CartPole (2 ações) e MountainCar (3 ações), com recompensas simples, essa superestimação tende a ser pequena, de modo que um efeito pequeno do Double DQN é coerente com a teoria; a vantagem costuma ser maior em ambientes com muitas ações e recompensas mais variáveis, como o Atari.",
   "Limitações. São apenas 5 sementes por configuração e nenhum teste estatístico; as curvas de treino incluem exploração residual (ε = 0,05) e oscilam bastante no CartPole; o fim de episódio por truncamento (500 passos) é tratado como terminal no replay buffer; cada algoritmo usa um único conjunto de hiperparâmetros.",
   "Melhorias. Usar 20 sementes ou mais com teste estatístico; guardar o melhor checkpoint em vez do último, já que o desempenho oscila; atualização suave da rede-alvo; tratar o truncamento sem fixar o alvo em zero; Prioritized Replay e Dueling."
  ],
  "concl": [
   "No Classic Control, DQN e Double DQN resolveram o CartPole e o MountainCar nas condições testadas, e não houve diferença sistemática entre eles. Foi refutada a expectativa de que o MountainCar exigisse recompensa modelada.",
   "A lição é que, em ambientes simples, a escolha de lr e a variância entre sementes pesam mais que a regra do alvo, e que reportar média e desvio de várias sementes é indispensável."
  ]
 }
};
const paras = (arr, fallback) => (arr ? arr.map((x) => p(x)) : [todo(fallback)]);

// Textos do Atari e da conclusão geral (resultados chegaram depois dos demais).
const T2 = {
  pong: [
    "Linha de base: um agente aleatório, com o mesmo pré-processamento, obtém −20,60 ± 0,66 (30 episódios). O DQN praticamente não saiu dela: avaliação final de −20,9 ± 0,3 (30 episódios, ε = 0,05) e avaliações periódicas entre −21,0 e −20,2 ao longo de todo o milhão de passos; o retorno médio de treino foi −20,5 nos primeiros 100 mil passos e −20,9 nos últimos 100 mil. O Double DQN começou a melhorar a partir de ≈ 300 mil passos (curva de treino): avaliações periódicas de −19,4 (400 mil), −18,0 (500 e 600 mil), −16,8 (700 e 800 mil) e −15,0 (900 mil, o melhor valor) e avaliação final de −17,7 ± 1,8; o retorno médio de treino subiu de −20,4 (primeiros 100 mil passos) para −15,9 (últimos 100 mil). Os episódios do Double DQN também ficaram bem mais longos (cerca de 1 800 passos por episódio nos últimos 100 mil passos, contra ≈ 790 do DQN), o que indica que ele passou a devolver a bola e a prolongar os pontos. Nenhum dos dois ganhou pontos de forma consistente: ambos ainda perdem a partida. Como há uma única semente por algoritmo, a diferença entre eles (−17,7 contra −20,9) pode refletir em parte a variação entre sementes e não só a regra do alvo."],
  spaceinvaders: [
    "Linha de base: o agente aleatório obtém 147,0 ± 86,1 (30 episódios). Os dois algoritmos aprenderam de forma clara e quase idêntica: avaliação final de 415,8 ± 163,4 para o DQN e 402,7 ± 127,3 para o Double DQN (30 episódios, ε = 0,05), cerca de 2,7 a 2,8 vezes a linha de base, com dispersão entre episódios maior que a diferença entre as médias. O retorno médio de treino subiu de ≈ 154 (DQN) e ≈ 151 (Double DQN) nos primeiros 100 mil passos para ≈ 450 e ≈ 472 nos últimos 100 mil, sem sinal de saturação, ou seja, o treino de 1 milhão de passos (4 milhões de quadros) é curto para este jogo. As avaliações periódicas, com apenas 5 episódios, atingem o pico em 800 mil passos (427 e 429) e depois oscilam, o que reflete o ruído da avaliação. As curvas de treino dos dois algoritmos praticamente se sobrepõem."],
  disc: [
    "Hipóteses (Atari). H3 foi só em parte confirmada. No Pong, o Double DQN saiu de −21 e melhorou (−17,7 na avaliação final), mas o DQN não saiu da linha de base em 1 milhão de passos. No Space Invaders, ao contrário do esperado, o jogo não foi o mais lento: os dois algoritmos mais que dobraram o retorno do agente aleatório (415,8 e 402,7 contra 147,0), enquanto o Pong exigiu mais passos para sair do platô de −21. H1 (Double DQN mais estável e melhor) é sugerida no Pong e ausente no Space Invaders, mas, com uma semente por jogo e algoritmo, não pode ser confirmada.",
    "Teoria. A superestimação de valores que o Double DQN corrige depende do jogo e do estágio do aprendizado; é coerente que o efeito apareça em um jogo (Pong, em que o DQN ficou preso no início, com recompensa muito esparsa) e não em outro (Space Invaders, com recompensas frequentes). Esta explicação é uma hipótese, que este experimento não testa. Os escores também não são comparáveis aos de Mnih et al. (2015) ou van Hasselt et al. (2016), porque aqui o orçamento é de 4 milhões de quadros, o ambiente usa ações pegajosas e a avaliação usa ε = 0,05.",
    "Limitações. Uma única semente por jogo e algoritmo (as diferenças não têm significância estatística); 1 milhão de passos, muito menos que o necessário para a convergência em ambos os jogos; avaliações periódicas com apenas 5 episódios; um único conjunto de hiperparâmetros; o truncamento tratado como terminal (irrelevante no Atari, cujo limite é de 108 mil quadros) e o replay buffer não restaurado em retomadas (não houve retomadas: as quatro execuções foram contínuas).",
    "Melhorias. Repetir com pelo menos 3 sementes por jogo e algoritmo e testar a diferença; treinar por 5 a 10 milhões de passos; usar avaliações periódicas com mais episódios; testar Prioritized Replay, Dueling e Rainbow, apresentados na Aula 03; e ajustar a taxa de aprendizado e o decaimento de ε, que foram mantidos fixos."],
  concl: [
    "No Atari, foi comprovado que o DQN e o Double DQN aprendem Space Invaders diretamente de pixels em 1 milhão de passos (≈ 2,7 a 2,8 vezes o agente aleatório), sem diferença distinguível entre eles. No Pong, só o Double DQN saiu do platô de −21 e, ainda assim, sem vencer pontos de forma consistente; o resultado sugere vantagem do Double DQN, mas não a comprova, porque cada configuração tem uma única semente.",
    "A lição é que o orçamento de passos e o número de sementes limitam o que se pode concluir no Atari, e que a comparação entre algoritmos deveria usar mais sementes e treinos mais longos."],
  general: [
    "Os três conjuntos de experimentos mostram, em ordem crescente de complexidade, como a escolha do algoritmo e da exploração afeta o aprendizado. Com tabelas (Parte I), a diferença entre aprender o valor da política gulosa (Q-learning) e o da política executada (SARSA) aparece de forma nítida e reprodutível. A Parte II mostra o custo imediato da exploração sobre uma política que já funciona. Com aproximação de função (Parte III), a variação entre sementes e entre configurações passa a pesar mais que a regra do alvo: em CartPole e MountainCar não houve diferença sistemática entre DQN e Double DQN, e no Atari, com uma semente, só se pode sugerir uma vantagem do Double DQN no Pong.",
    "O aprendizado transversal é metodológico: separar o desempenho de treino do de avaliação, repetir com várias sementes antes de afirmar diferenças e declarar as limitações de cada experimento. Os próximos passos naturais são repetir o Atari com mais sementes e passos, e testar as extensões do DQN apresentadas na Aula 03."],
};

function atariTable(file, title) {
  const rows = csvRows(file);
  if (!rows) return [todo(`tabela ${file}`)];
  return table(["Algoritmo", "Sementes", "Retorno médio (30 episódios)", "Desvio entre os 30 episódios"],
    rows.map((r) => [r[0], r[1], fmt(r[2]), fmt(r[4])]), [2626, 1400, 2500, 2500], title);
}

// ---------- capa e introdução geral ----------
const cover = [
  new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.CENTER, children: [run("Relatório de Experimentos de Aprendizado por Reforço", { bold: true, size: 44 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [run("Q-learning, SARSA, DQN e Double DQN em ambientes do Gymnasium", { size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [run("Disciplina de Aprendizado por Reforço — PPGTI — 2026/2", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Danielle Magalhães Ballester", { size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Código: github.com/daniballester-ai/reinforcement-learning-lab", { size: 20 })] }),
];

const intro = [
  h1("Introdução geral"),
  p("Este documento reúne três relatórios de experimentos da disciplina, cada um organizado conforme o modelo disponibilizado na turma: Parte I (Aula 02, exercício do slide 56) compara Q-learning(0) e SARSA(0) nos quatro ambientes Toy Text do Gymnasium; Parte II (Aula 02, exemplo do CartPole) explora a API do Gymnasium e o efeito de ruído de exploração sobre uma política fixa; Parte III (Aula 03, exercício do slide 64) compara DQN e Double DQN em CartPole, MountainCar, Pong e Space Invaders."),
  p("Todo o código é de autoria própria, em PyTorch e Gymnasium (Towers et al., 2024), e está disponível no repositório indicado na capa. Os resultados são reproduzíveis pelos comandos do README: cada execução grava arquivos CSV/JSON por semente e as figuras e tabelas deste documento são geradas a partir deles."),
];

// ---------- Parte I: Toy Text ----------
const partI = [
  h1("Parte I — Q-learning(0) e SARSA(0) nos ambientes Toy Text"),
  h2("1. Introdução"),
  p("Os ambientes Toy Text do Gymnasium têm espaços de estados e ações discretos e pequenos, o que permite representar a função de valor de ação Q como uma tabela. O objetivo principal deste experimento é comparar dois algoritmos de diferença temporal com a mesma política de comportamento ε-greedy, Q-learning(0) (off-policy) e SARSA(0) (on-policy), em Blackjack, CliffWalking, FrozenLake e Taxi, e analisar de forma crítica o efeito do ambiente e da taxa de aprendizado α."),
  p("O problema do agente, em cada ambiente, é aprender uma política que maximize o retorno esperado apenas por interação, sem conhecer o modelo de transição. As hipóteses de trabalho são:"),
  ...bullets([
    "H1 — No CliffWalking, o Q-learning aprende o caminho ótimo rente ao penhasco, mas, com exploração ε-greedy durante o treino, obtém retorno online menor que o SARSA, que aprende um caminho mais seguro.",
    "H2 — No FrozenLake escorregadio, ambos obtêm taxa de sucesso bem abaixo de 100% e maior variância entre sementes que na versão determinística; na versão sem escorregar ambos convergem para a política ótima.",
    "H3 — α pequeno (0,05) converge mais devagar e α grande (0,5) produz curvas mais ruidosas; α = 0,1 é um compromisso razoável.",
    "H4 — No Blackjack, ambos aprendem uma política próxima da estratégia básica, com retorno esperado ligeiramente negativo (o cassino mantém a vantagem).",
  ]),
  h2("2. Fundamentação Teórica"),
  h3("2.1. Algoritmos Utilizados"),
  p("Ambos os algoritmos atualizam a estimativa Q(s, a) a cada transição (s, a, r, s′), movendo-a na direção de um alvo de diferença temporal com taxa de aprendizado α e fator de desconto γ (Sutton e Barto, 2018). O sufixo (0) indica que o alvo usa apenas um passo (TD(0)), sem traços de elegibilidade."),
  eq("Q-learning:  Q(s,a) ← Q(s,a) + α [ r + γ · max_a′ Q(s′,a′) − Q(s,a) ]"),
  eq("SARSA:  Q(s,a) ← Q(s,a) + α [ r + γ · Q(s′,a′) − Q(s,a) ]"),
  p("A diferença está no alvo. O Q-learning usa o máximo sobre as ações em s′ e, por isso, aprende o valor da política gulosa independentemente da política de comportamento (off-policy). O SARSA usa a ação a′ efetivamente escolhida pela política de comportamento ε-greedy e, portanto, aprende o valor dessa política, incluindo o custo da exploração (on-policy). Em estados terminais o alvo é apenas r. Como política de comportamento usa-se ε-greedy com ε decaindo linearmente de 1,0 a 0,05 durante a primeira metade dos episódios, com desempate aleatório entre ações de mesmo valor."),
  h3("2.2. Ambientes"),
  ...table(
    ["Ambiente", "Estados", "Ações", "Recompensa", "Término"],
    [
      ["FrozenLake 4×4", "16 posições", "4 (esq., baixo, dir., cima)", "+1 ao chegar ao objetivo; 0 nos demais passos", "Objetivo, buraco ou 100 passos"],
      ["CliffWalking 4×12", "48 posições", "4 (cima, dir., baixo, esq.)", "−1 por passo; −100 ao cair no penhasco (volta ao início)", "Chegada ao objetivo"],
      ["Taxi", "500", "6 (4 movimentos, embarcar, desembarcar)", "−1 por passo; +20 na entrega; −10 em embarque/desembarque ilegal", "Entrega ou 200 passos"],
      ["Blackjack", "(soma do jogador, carta do dealer, ás utilizável)", "2 (parar, pedir)", "+1 vitória, 0 empate, −1 derrota", "Ao parar ou estourar"],
    ],
    [1500, 1800, 1900, 2226, 1600], "Resumo dos ambientes Toy Text utilizados."),
  p("As principais dificuldades de cada ambiente são: no FrozenLake escorregadio, a estocasticidade das transições (o agente só se move na direção desejada com probabilidade 1/3) e a recompensa esparsa; no CliffWalking, a tensão entre o caminho curto, perigoso, e o longo, seguro; no Taxi, o espaço de estados maior e a necessidade de realizar duas subtarefas em sequência; no Blackjack, a variância alta do retorno e a observação em tupla."),
  h2("3. Metodologia Experimental"),
  h3("3.1. Configuração dos Experimentos"),
  ...table(
    ["Ambiente", "Episódios de treino", "γ", "Avaliação"],
    [
      ["FrozenLake (escorregadio)", "20 000", "0,99", "1 000 episódios gulosos"],
      ["FrozenLake (sem escorregar)", "20 000", "0,99", "1 000 episódios gulosos"],
      ["CliffWalking", "1 000", "0,99", "1 000 episódios gulosos"],
      ["Taxi", "10 000", "0,99", "1 000 episódios gulosos"],
      ["Blackjack", "500 000", "1,0", "1 000 episódios gulosos"],
    ],
    [3000, 2200, 1000, 2826], "Orçamento por ambiente."),
  ...bullets([
    "Taxa de aprendizado: α ∈ {0,05; 0,1; 0,5}; α = 0,1 nas curvas principais.",
    "Exploração: ε-greedy, ε de 1,0 a 0,05 com decaimento linear nos primeiros 50% dos episódios.",
    "Repetições: 10 sementes (0 a 9) por combinação de ambiente, algoritmo e α.",
    "Critério de parada: número fixo de episódios (não há parada antecipada).",
    "Ferramentas: Python 3.13, Gymnasium 1.4.0, NumPy, pandas e matplotlib.",
  ]),
  p("Como o CliffWalking-v1 não possui limite de passos nesta versão do Gymnasium, a avaliação gulosa é limitada a 1 000 passos por episódio, de modo que uma política que entre em ciclo seja contabilizada como falha em vez de travar a execução."),
  h3("3.2. Procedimentos de Treinamento e Avaliação"),
  p("O agente é treinado online, atualizando Q a cada passo. Ao fim do treino, a política gulosa em relação a Q é avaliada em 1 000 episódios, sem exploração e sem aprendizado. O desempenho é medido pelo retorno: (i) a curva de aprendizado é a média móvel do retorno por episódio, com média e desvio padrão entre as sementes; (ii) a tabela de resultados traz a média e o desvio padrão, entre sementes, do retorno médio de avaliação. A estabilidade é lida pela largura da faixa de desvio. Para FrozenLake e CliffWalking, os valores máximos de Q e as ações gulosas são visualizados como mapas de calor sobre o grid."),
  h2("4. Resultados"),
  ...["frozenlake", "frozenlake_noslip", "cliffwalking", "taxi", "blackjack"].flatMap((e, i) => {
    const names = { frozenlake: "FrozenLake (escorregadio)", frozenlake_noslip: "FrozenLake (sem escorregar)", cliffwalking: "CliffWalking", taxi: "Taxi", blackjack: "Blackjack" };
    const out = [h3(`4.${i + 1}. ${names[e]}`), ...resultTable(`r1_${e}.csv`, `${names[e]}: retorno médio de avaliação (média e desvio entre sementes).`, "Algoritmo e α"),
      ...figure(`r1_${e}.png`, `${names[e]}: curva de aprendizado, Q-learning × SARSA (α = 0,1).`)];
    if (["frozenlake", "frozenlake_noslip", "cliffwalking"].includes(e)) {
      out.push(...figure(`r1_${e}_q_learning_policy.png`, `${names[e]}: valor máximo e ação gulosa, Q-learning.`, 400));
      out.push(...figure(`r1_${e}_sarsa_policy.png`, `${names[e]}: valor máximo e ação gulosa, SARSA.`, 400));
    }
    out.push(...paras(T.r1[e], `análise de ${names[e]}`));
    return out;
  }),
  h2("5. Discussão"),
  ...paras(T.r1.disc),
  h2("6. Conclusões"),
  ...paras(T.r1.concl),
];

// ---------- Parte II: CartPole ----------
const partII = [
  h1("Parte II — CartPole e a API do Gymnasium"),
  h2("1. Introdução"),
  p("O CartPole-v1 é o ambiente de controle clássico usado como \"Hello World\" do aprendizado por reforço. Este experimento tem dois objetivos: (i) exercitar a API do Gymnasium (ambiente, agente, wrappers e renderização), reproduzindo o exemplo da aula; (ii) medir o efeito de injetar exploração aleatória, por meio de um ActionWrapper ε-aleatório, sobre uma política que já resolve o problema. Não há aprendizado nesta parte; o DQN no CartPole é tratado na Parte III."),
  p("O problema do agente é manter o pêndulo equilibrado pelo maior número de passos. As hipóteses são: H1 — a política aleatória dura em torno de 20 a 25 passos; H2 — a política heurística sem ruído atinge o limite de 500 passos em todos os episódios; H3 — o retorno médio cai de forma monótona e acentuada conforme ε aumenta."),
  h2("2. Fundamentação Teórica"),
  h3("2.1. Algoritmos Utilizados"),
  p("Três agentes sem aprendizado são comparados. O agente aleatório sorteia uma ação uniformemente a cada passo. A política heurística empurra o carrinho na direção em que o pêndulo está inclinando, considerando também a velocidade angular:"),
  eq("a = 1 (empurrar à direita) se θ + 0,5 · θ̇ > 0; caso contrário a = 0 (à esquerda)"),
  p("Por fim, o RandomActionWrapper (um gym.ActionWrapper) substitui, com probabilidade ε, a ação do agente por uma ação sorteada, o mecanismo prático de exploração ε-greedy mostrado na aula."),
  h3("2.2. Ambientes"),
  ...table(
    ["Item", "CartPole-v1"],
    [
      ["Estado", "Posição do carrinho x, velocidade ẋ, ângulo do pêndulo θ e velocidade angular θ̇ (vetor com 4 números reais)"],
      ["Ações", "2 discretas: empurrar o carrinho para a esquerda (0) ou para a direita (1)"],
      ["Recompensa", "+1 a cada passo em que o pêndulo permanece de pé"],
      ["Término", "|θ| > 12° (≈ 0,2095 rad) ou |x| > 2,4 (terminação); truncamento em 500 passos"],
      ["Desafio", "Sistema instável: pequenos erros se amplificam e exigem correção contínua"],
    ],
    [1800, 7226], "Resumo do CartPole-v1."),
  h2("3. Metodologia Experimental"),
  h3("3.1. Configuração dos Experimentos"),
  ...bullets([
    "Condições: agente aleatório; política heurística com ε ∈ {0; 0,1; 0,3; 0,5}.",
    "100 episódios por condição, com reinício determinístico (semente base 0 + índice do episódio).",
    "Sem treinamento, sem fator de desconto e sem critério de parada além do término do episódio.",
    "Ferramentas: Python 3.13, Gymnasium 1.4.0, NumPy e pandas; vídeos gravados com RecordVideo.",
  ]),
  h3("3.2. Procedimentos de Treinamento e Avaliação"),
  p("Não há treinamento. O desempenho é o retorno por episódio (equivalente ao número de passos em pé), resumido por média e desvio padrão em cada condição. Um vídeo do agente aleatório e outro da política heurística foram gravados para inspeção qualitativa."),
  h2("4. Resultados"),
  ...(() => {
    const rows = csvRows("r2_cartpole.csv");
    if (!rows) return [todo("tabela r2_cartpole.csv (execute `python -m analysis.make_figures`)")];
    const names = { random: "Aleatório", "heuristic_eps0.0": "Heurística, ε = 0", "heuristic_eps0.1": "Heurística, ε = 0,1", "heuristic_eps0.3": "Heurística, ε = 0,3", "heuristic_eps0.5": "Heurística, ε = 0,5" };
    const order = ["random", "heuristic_eps0.0", "heuristic_eps0.1", "heuristic_eps0.3", "heuristic_eps0.5"];
    return table(["Condição", "Retorno médio", "Desvio padrão"],
      order.map((k) => { const r = rows.find((x) => x[0] === k); return r ? [names[k], fmt(r[1]), fmt(r[2])] : [names[k], "-", "-"]; }),
      [4026, 2500, 2500], "CartPole-v1: retorno por episódio em 100 episódios por condição.");
  })(),
  ...paras(T.r2.analysis),
  h2("5. Discussão"),
  ...paras(T.r2.disc),
  h2("6. Conclusões"),
  ...paras(T.r2.concl),
];

// ---------- Parte III: DQN e Double DQN ----------
const names3 = { cartpole: "CartPole", cartpole_prof: "CartPole (lr do notebook do professor)", mountaincar: "MountainCar", mountaincar_shaped: "MountainCar (recompensa modelada)", pong: "Pong", spaceinvaders: "Space Invaders" };
const partIII = [
  h1("Parte III — DQN e Double DQN: Classic Control e Atari"),
  h2("1. Introdução"),
  p("Quando o espaço de estados é grande ou contínuo, a tabela Q deixa de ser viável e a função de valor passa a ser aproximada por uma rede neural. O objetivo deste experimento é comparar o DQN (com rede-alvo) e o Double DQN em dois ambientes de controle clássico, CartPole e MountainCar, e em dois jogos do Atari, Pong e Space Invaders, em que a entrada são imagens."),
  p("O agente deve aprender, a partir de interação e recompensas, uma política que equilibre o pêndulo (CartPole), leve o carro ao topo da colina (MountainCar), vença pontos contra a CPU (Pong) e destrua invasores (Space Invaders). As hipóteses são:"),
  ...bullets([
    "H1 — O Double DQN reduz a superestimação dos valores Q e, por isso, é mais estável que o DQN; a diferença é pequena no CartPole e mais visível no Atari.",
    "H2 — No MountainCar, a recompensa esparsa (−1 por passo, sem sinal intermediário) faz o DQN puro falhar dentro do orçamento; uma recompensa modelada por energia resolve o problema.",
    "H3 — No Pong, ambos saem de aproximadamente −21 e melhoram ao longo do treino; o Space Invaders, com recompensas mais variáveis, melhora mais devagar.",
  ]),
  h2("2. Fundamentação Teórica"),
  h3("2.1. Algoritmos Utilizados"),
  p("O DQN (Mnih et al., 2015) aproxima Q(s, a; θ) por uma rede neural e estabiliza o aprendizado com duas técnicas: a repetição de experiências (replay buffer), que amostra transições passadas de forma descorrelacionada, e uma rede-alvo com parâmetros θ⁻ copiados periodicamente da rede online. A perda é o erro quadrático (ou de Huber) entre Q(s, a; θ) e o alvo y:"),
  eq("DQN:  y = r + γ · max_a′ Q(s′, a′; θ⁻)"),
  eq("Double DQN:  y = r + γ · Q(s′, argmax_a′ Q(s′, a′; θ); θ⁻)"),
  p("O máximo do DQN usa a mesma rede para escolher e para avaliar a melhor ação, o que tende a superestimar valores. O Double DQN (van Hasselt et al., 2016) desacopla as duas etapas: a rede online escolhe a ação e a rede-alvo a avalia. Nos dois casos a transição terminal usa apenas y = r."),
  p("Observação sobre o material da aula: no notebook de exemplo, o algoritmo chamado de \"DDQN\" usa uma rede-alvo com atualização suave, mas calcula o alvo com o máximo da própria rede-alvo; isso corresponde, na prática, a um DQN com rede-alvo. Neste trabalho o DQN usa rede-alvo com atualização rígida e o Double DQN implementa o desacoplamento descrito acima, de modo que a diferença entre os dois algoritmos é exatamente a regra do alvo."),
  h3("2.2. Ambientes"),
  ...table(
    ["Ambiente", "Estado", "Ações", "Recompensa", "Término"],
    [
      ["CartPole-v1", "4 números reais (x, ẋ, θ, θ̇)", "2", "+1 por passo de pé", "|θ| > 12°, |x| > 2,4 ou 500 passos"],
      ["MountainCar-v0", "Posição ∈ [−1,2; 0,6] e velocidade ∈ [−0,07; 0,07]", "3 (esq., nada, dir.)", "−1 por passo", "Posição ≥ 0,5 ou 200 passos"],
      ["Pong (ALE/Pong-v5)", "Imagem RGB 210×160", "6 (conjunto mínimo)", "+1 por ponto marcado, −1 por ponto sofrido; retorno em [−21, +21]", "Fim da partida (21 pontos)"],
      ["Space Invaders (ALE/SpaceInvaders-v5)", "Imagem RGB 210×160", "6 (conjunto mínimo)", "Pontos por invasor destruído", "Fim das vidas ou invasores chegam à base"],
    ],
    [1700, 2100, 1300, 2326, 1600], "Resumo dos ambientes."),
  p("Nos jogos do Atari (Arcade Learning Environment; Bellemare et al., 2013), a observação é convertida em escala de cinza 84×84 e quatro quadros consecutivos são empilhados para que a rede perceba movimento; cada ação é repetida por 4 quadros (frame-skip) e inicia-se cada episódio com até 30 ações nulas. As principais dificuldades são: no CartPole, o equilíbrio instável; no MountainCar, a exploração com recompensa esparsa (o carro precisa ganhar impulso oscilando); no Atari, a entrada de alta dimensão, as recompensas atrasadas e o custo computacional."),
  h2("3. Metodologia Experimental"),
  h3("3.1. Configuração dos Experimentos"),
  ...table(
    ["Parâmetro", "Classic Control", "Atari"],
    [
      ["Rede", "MLP 2 × 128 (ReLU)", "CNN da DeepMind: 3 conv. (32, 64, 64) + FC 512"],
      ["Otimizador / taxa de aprendizado", "Adam, 1e-3 (baseline do professor: 1e-4)", "Adam, 1e-4"],
      ["Fator de desconto γ", "0,99", "0,99"],
      ["Perda", "Erro quadrático médio", "Huber"],
      ["Lote / buffer de replay", "64 / 50 mil", "32 / 100 mil"],
      ["Início do aprendizado", "1 000 passos", "20 mil passos"],
      ["Frequência de atualização", "A cada passo", "A cada 4 passos"],
      ["Rede-alvo (cópia rígida)", "A cada 500 passos", "A cada 8 000 passos"],
      ["ε-greedy", "1,0 → 0,05 em 10 mil passos", "1,0 → 0,01 em 250 mil passos"],
      ["Recompensa", "Original", "Limitada a [−1, 1] no treino"],
      ["Orçamento", "CartPole 100 mil passos; MountainCar 200 mil", "1 milhão de passos por execução"],
      ["Sementes", "5 (0 a 4)", "1 (semente 0)"],
      ["Avaliação", "A cada 5 mil passos (10 episódios gulosos); final com 100", "A cada 100 mil passos (5 episódios); final com 30 episódios, ε = 0,05"],
    ],
    [2600, 3100, 3326], "Hiperparâmetros e orçamento."),
  p("As quatro execuções do Atari rodaram em paralelo em uma GPU Tesla T4 do Kaggle, de forma contínua (sem retomadas), com o orçamento integral de 1 milhão de passos cada. A versão do PyTorch da plataforma não foi registrada."),
  p("DQN e Double DQN usam a mesma semente e, portanto, partem de pesos iniciais idênticos, de modo que a comparação isola o efeito da regra do alvo. No MountainCar, foi incluída uma variante com recompensa modelada por energia mecânica (r + γΦ(s′) − Φ(s), com Φ proporcional à energia cinética mais potencial); as métricas continuam sendo calculadas com a recompensa original. Como Φ não é zerada nos estados terminais, a modelagem não é estritamente invariante à política (Ng et al., 1999); o viés é pequeno e a variante é apenas um extra. Ferramentas: Python 3.13, Gymnasium 1.4.0, ale-py, PyTorch, NumPy, pandas e matplotlib. Os ambientes ALE/*-v5 usam ações pegajosas (repeat_action_probability = 0,25) e conjunto mínimo de ações, o protocolo de Machado et al. (2018); somados aos até 30 no-ops iniciais, os escores de Pong e Space Invaders não são diretamente comparáveis aos das tabelas de Mnih et al. (2015) ou van Hasselt et al. (2016)."),
  h3("3.2. Procedimentos de Treinamento e Avaliação"),
  p("Em cada passo o agente escolhe uma ação ε-greedy, armazena a transição no replay buffer e, a partir do início do aprendizado, atualiza a rede online com um lote amostrado do buffer; periodicamente copia os pesos para a rede-alvo. O desempenho é medido pelo retorno por episódio: a curva de treino é a média móvel do retorno com média e desvio entre as sementes, e a avaliação é feita com a política gulosa (ε = 0 no Classic Control e ε = 0,05 no Atari) em ambientes separados do treino. As execuções do Atari salvam checkpoints a cada 50 mil passos e podem ser retomadas."),
  p("Limitações metodológicas declaradas: (i) o Atari usa apenas uma semente por jogo e algoritmo, portanto as diferenças entre DQN e Double DQN não têm significância estatística; (ii) o orçamento de passos é muito menor que o da literatura; (iii) o fim de episódio por truncamento (por exemplo, os 500 passos do CartPole e os 200 do MountainCar) é tratado como terminal no replay buffer, sem bootstrap, o que injeta alvos contraditórios justamente nos estados da política boa e pode contribuir para as quedas tardias observadas no CartPole (hipótese não testada); (iv) ao retomar uma execução a partir de um checkpoint, o replay buffer não é restaurado e é reaquecido com a política atual."),
  h2("4. Resultados"),
  ...Object.keys(names3).flatMap((e, i) => [
    h3(`4.${i + 1}. ${names3[e]}`),
    ...(["pong", "spaceinvaders"].includes(e)
      ? atariTable(`r3_${e}.csv`, `${names3[e]}: retorno de avaliação final (1 semente; desvio entre os 30 episódios de avaliação).`)
      : resultTable(`r3_${e}.csv`, `${names3[e]}: retorno de avaliação final (média e desvio entre sementes).`, "Algoritmo")),
    ...figure(`r3_${e}.png`, `${names3[e]}: curva de aprendizado, DQN × Double DQN.`),
    ...paras(T.r3[e] || T2[e], `análise qualitativa de ${names3[e]}: velocidade de aprendizado, estabilidade, falhas observadas`),
  ]),
  h2("5. Discussão"),
  ...paras(T.r3.disc),
  ...paras(T2.disc),
  h2("6. Conclusões"),
  ...paras(T.r3.concl),
  ...paras(T2.concl),
];

const conclusaoGeral = [
  h1("Conclusão geral"),
  ...paras(T2.general),
  h2("Referências"),
  ...bullets([
    "BELLEMARE, M. G.; NADDAF, Y.; VENESS, J.; BOWLING, M. The Arcade Learning Environment: An Evaluation Platform for General Agents. Journal of Artificial Intelligence Research, v. 47, p. 253-279, 2013. DOI: 10.1613/jair.3912.",
    "MACHADO, M. C.; BELLEMARE, M. G.; TALVITIE, E.; VENESS, J.; HAUSKNECHT, M.; BOWLING, M. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. Journal of Artificial Intelligence Research, v. 61, p. 523-562, 2018. DOI: 10.1613/jair.5699.",
    "MNIH, V. et al. Human-level control through deep reinforcement learning. Nature, v. 518, n. 7540, p. 529-533, 2015. DOI: 10.1038/nature14236.",
    "NG, A. Y.; HARADA, D.; RUSSELL, S. J. Policy invariance under reward transformations: theory and application to reward shaping. In: International Conference on Machine Learning (ICML), 16., 1999. Morgan Kaufmann, p. 278-287. (Sem DOI.)",
    "SUTTON, R. S.; BARTO, A. G. Reinforcement Learning: An Introduction. 2. ed. Cambridge, MA: MIT Press, 2018. ISBN 978-0-262-03924-6. (Livro, sem DOI.)",
    "TOWERS, M. et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032, 2024. DOI: 10.48550/arXiv.2407.17032.",
    "VAN HASSELT, H.; GUEZ, A.; SILVER, D. Deep Reinforcement Learning with Double Q-learning. Proceedings of the AAAI Conference on Artificial Intelligence, v. 30, n. 1, 2016. DOI: 10.1609/aaai.v30i1.10295.",
    "Material da disciplina: Aprendizado por Reforço, Aulas 01 a 03 (2026).",
  ]),
];

const doc = new Document({
  creator: "Danielle Magalhães Ballester",
  title: "Relatório de Experimentos de Aprendizado por Reforço",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, font: "Arial" }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: "Arial" }, paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, font: "Arial" }, paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Página "), new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
    children: [...cover, ...intro, ...partI, ...partII, ...partIII, ...conclusaoGeral],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2] || path.join(REL, "Relatorio_Final.docx");
  fs.writeFileSync(out, buf);
  console.log("gerado", out);
});
