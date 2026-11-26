# Links (or with -Copy copies) skills\gpu-kernel-tuning into the skills directory of every agent found on this PC.
param([switch]$Copy)
$ErrorActionPreference = "Stop"
$src = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "skills\gpu-kernel-tuning"
$targets = @(
    @{ agent = "Claude Code"; home = "$HOME\.claude" },
    @{ agent = "Codex CLI"; home = "$HOME\.codex"; dir = "$HOME\.agents\skills" },
    @{ agent = "Cline"; home = "$HOME\.cline" },
    @{ agent = "Hermes Agent"; home = "$env:LOCALAPPDATA\hermes" },
    @{ agent = "DeepSeek Harness"; home = "$HOME\.dsh" },
    @{ agent = "oh-my-pi"; home = "$HOME\.omp"; dir = "$HOME\.omp\agent\skills" },
    @{ agent = "pi"; home = "$HOME\.pi"; dir = "$HOME\.pi\agent\skills" }
)
foreach ($t in $targets) {
    if (-not (Test-Path $t.home)) { continue }
    $dir = if ($t.dir) { $t.dir } else { Join-Path $t.home "skills" }
    $dst = Join-Path $dir "gpu-kernel-tuning"
    if (Test-Path $dst) { "$($t.agent): $dst exists, left as it is"; continue }
    New-Item -ItemType Directory -Force $dir | Out-Null
    if ($Copy) { Copy-Item $src $dst -Recurse } else { New-Item -ItemType Junction -Path $dst -Target $src | Out-Null }
    "$($t.agent): $dst"
}
