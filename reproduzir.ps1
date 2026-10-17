<#
.SYNOPSIS
  Reproduz os experimentos de CPU (Partes I, II e Classic Control da Parte III) e regenera figuras e relatorio.
  O Atari exige GPU e roda no Kaggle/Colab (notebooks\atari_gpu.ipynb; veja o README).

.EXAMPLE
  .\reproduzir.ps1 -Etapa tudo
  .\reproduzir.ps1 -Etapa tabular -Workers 4
  .\reproduzir.ps1 -Etapa tudo -DryRun      # so mostra os comandos, nao executa
#>
param(
    [ValidateSet('tudo', 'instalar', 'testes', 'tabular', 'cartpole', 'classic', 'baseline', 'figuras', 'relatorio')]
    [string]$Etapa = 'tudo',
    [int]$Workers = 5,                 # processos paralelos das varreduras (ajuste ao numero de nucleos)
    [string]$VenvDir = 'C:\venvs\rl',  # venv fora do Drive, para nao sincronizar milhares de arquivos
    [string]$NodeDir = 'C:\venvs\node',# pacote npm "docx" (so para gerar o relatorio)
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$py = Join-Path $VenvDir 'Scripts\python.exe'

function Run([string]$titulo, [scriptblock]$cmd) {
    Write-Host ''
    Write-Host "=== $titulo ===" -ForegroundColor Cyan
    if ($DryRun) { Write-Host "(dry-run) $cmd"; return }
    $inicio = Get-Date
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Falhou: $titulo (codigo $LASTEXITCODE)" }
    Write-Host ('ok em {0:N0} s' -f ((Get-Date) - $inicio).TotalSeconds)
}

function Instalar {
    if (-not (Test-Path $py)) { Run 'Criar venv' { python -m venv $VenvDir } }
    Run 'Instalar PyTorch (CPU)' { & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu }
    Run 'Instalar dependencias' { & $py -m pip install -r requirements.txt }
}

function VerificarAtari {
    $faltam = @()
    foreach ($d in 'pong\dqn', 'pong\ddqn', 'spaceinvaders\dqn', 'spaceinvaders\ddqn') {
        if (-not (Test-Path "runs\$d\seed0\final.json")) { $faltam += $d }
    }
    if ($faltam.Count -gt 0) {
        Write-Warning ("Resultados do Atari ausentes em runs\: " + ($faltam -join ', '))
        Write-Warning 'Rode notebooks\atari_gpu.ipynb no Kaggle e copie o resultado para runs\ (veja o README).'
    } else {
        Write-Host 'Atari: 4 execucoes encontradas em runs\.' -ForegroundColor Green
    }
}

function GerarRelatorio {
    if (-not $DryRun -and -not (Get-Command node -ErrorAction SilentlyContinue)) {
        throw 'Node.js nao encontrado. Instale o Node.js para gerar o relatorio.'
    }
    if (-not (Test-Path (Join-Path $NodeDir 'node_modules\docx'))) {
        Run 'Instalar pacote npm docx' { npm install --prefix $NodeDir docx@9 }
    }
    $env:NODE_PATH = Join-Path $NodeDir 'node_modules'
    Run 'Gerar relatorio\Relatorio_Final.docx' { node analysis\build_report.js }
}

# ---- exige o venv, exceto na etapa de instalacao ----
if ($Etapa -in 'tudo', 'instalar') {
    if ($Etapa -eq 'instalar' -or -not (Test-Path $py)) { Instalar }
} elseif (-not $DryRun -and -not (Test-Path $py)) {
    throw "Venv nao encontrado em $VenvDir. Rode primeiro: .\reproduzir.ps1 -Etapa instalar"
}

$todas = ($Etapa -eq 'tudo')
if ($todas -or $Etapa -eq 'testes')   { Run 'Testes (esperado: 47 passed)' { & $py -m pytest -q } }
if ($todas -or $Etapa -eq 'tabular')  { Run 'Parte I: Q-learning x SARSA (300 execucoes)' { & $py -m rl_lab.sweeps tabular --workers $Workers } }
if ($todas -or $Etapa -eq 'cartpole') { Run 'Parte II: CartPole e ActionWrapper' { & $py -m rl_lab.cartpole_demo } }
if ($todas -or $Etapa -eq 'classic')  { Run 'Parte III: DQN x Double DQN, Classic Control (40 execucoes)' { & $py -m rl_lab.sweeps classic --workers $Workers --shaped } }
if ($todas -or $Etapa -eq 'baseline') { Run 'Linha de base aleatoria do Atari' { & $py -m analysis.random_baseline } }
if ($todas) { VerificarAtari }
if ($todas -or $Etapa -eq 'figuras')  { Run 'Figuras e tabelas' { & $py -m analysis.make_figures } }
if ($todas -or $Etapa -eq 'relatorio') { GerarRelatorio }

Write-Host ''
Write-Host 'Concluido.' -ForegroundColor Green
