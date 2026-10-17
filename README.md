# Experimentos de Aprendizado por Reforço (PPGTI, 2026/2)

Código dos experimentos da disciplina de Aprendizado por Reforço, com o relatório final em
[`relatorio/Relatorio_Final.docx`](relatorio/Relatorio_Final.docx).

| Parte | Experimento | Onde roda |
|---|---|---|
| I | Q-learning(0) × SARSA(0) em FrozenLake, CliffWalking, Taxi e Blackjack | CPU |
| II | CartPole e a API do Gymnasium (agente aleatório, política heurística, `ActionWrapper` ε-aleatório) | CPU |
| III | DQN × Double DQN em CartPole, MountainCar (CPU) e Pong, Space Invaders (GPU) | CPU / GPU |

O DQN usa rede-alvo com cópia rígida; o Double DQN muda apenas o alvo: a rede online escolhe a ação
e a rede-alvo a avalia (`rl_lab/deep/dqn.py`, função `td_targets`).

## Estrutura

```
rl_lab/
  tabular/agent.py     Q-learning e SARSA tabulares
  deep/                DQN/Double DQN: dqn.py, networks.py, replay.py, train.py (checkpoint e retomada)
  envs/factory.py      criação dos ambientes (pré-processamento do Atari, recompensa modelada do MountainCar)
  configs.py           presets de cada experimento
  run.py               CLI de uma execução; sweeps.py: varreduras com várias sementes
  cartpole_demo.py     Parte II
analysis/              make_figures.py (figuras e tabelas), build_report.js (gera o docx)
notebooks/             atari_gpu.ipynb (Kaggle/Colab)
tests/                 testes unitários (pytest)
```

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # PyTorch só CPU: --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q                                      # os testes devem passar
```

Requer Python 3.11 ou superior (desenvolvido em 3.13) e Gymnasium 1.x com `ale-py` (as ROMs do Atari vêm no pacote).

## Como reproduzir

**Atalho (Windows/PowerShell):** `reproduzir.ps1` encadeia as etapas de CPU (testes, Parte I, Parte II,
Classic Control, linha de base, figuras e relatório). O Atari exige GPU e fica fora (veja abaixo).

```powershell
.\reproduzir.ps1 -Etapa instalar            # cria o venv e instala as dependências
.\reproduzir.ps1 -Etapa tudo -Workers 5     # roda tudo, na ordem; -DryRun só mostra os comandos
.\reproduzir.ps1 -Etapa tabular             # ou uma etapa só: testes, tabular, cartpole, classic,
                                            #   baseline, figuras, relatorio
```

Os comandos individuais, equivalentes às etapas, estão a seguir. Cada execução grava em
`runs/<preset>/<algoritmo>/seed<k>/` e é idempotente (pula o que já terminou).

```bash
# Parte I: 5 ambientes x 2 algoritmos x 3 valores de alfa x 10 sementes
python -m rl_lab.sweeps tabular --workers 5

# Parte II
python -m rl_lab.cartpole_demo

# Parte III, Classic Control: CartPole, MountainCar e variantes, 5 sementes
python -m rl_lab.sweeps classic --workers 6 --shaped

# Parte III, Atari (GPU): uma execução por jogo e algoritmo
python -m rl_lab.run --env pong --algo ddqn --seed 0 --out runs --resume
python -m rl_lab.run --env spaceinvaders --algo dqn --seed 0 --out runs --resume
# (o notebook notebooks/atari_gpu.ipynb dispara as 4 combinações em paralelo na mesma GPU)

# Linha de base aleatória do Atari (mesmo pré-processamento)
python -m analysis.random_baseline

# Figuras, tabelas e relatório
python -m analysis.make_figures
NODE_PATH=<pasta com o pacote npm docx> node analysis/build_report.js
```

Presets disponíveis em `--env`: `frozenlake`, `frozenlake_noslip`, `cliffwalking`, `taxi`, `blackjack`
(com `--algo q_learning|sarsa`) e `cartpole`, `cartpole_prof`, `mountaincar`, `mountaincar_shaped`,
`pong`, `spaceinvaders` (com `--algo dqn|ddqn`). Qualquer campo de `DeepConfig` pode ser sobrescrito com
`--set campo=valor`, por exemplo `--set total_steps=20000` para um teste rápido.

Para o Atari no Kaggle ou no Colab, empacote a pasta `rl_lab` em `rl_lab.zip`, disponibilize-a (Dataset no
Kaggle, ou Drive no Colab) e execute `notebooks/atari_gpu.ipynb`. O treino salva checkpoint a cada 50 mil
passos e `--resume` continua de onde parou.

## Limitações conhecidas

- No Atari há uma semente por jogo e algoritmo, então as diferenças entre DQN e Double DQN não têm
  significância estatística.
- O orçamento de passos é bem menor que o da literatura.
- O fim de episódio por truncamento é tratado como terminal no replay buffer.
- Ao retomar de um checkpoint, o replay buffer não é restaurado e é reaquecido com a política atual.

Os dados brutos (`runs/`), checkpoints e vídeos não são versionados.
