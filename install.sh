#!/usr/bin/env bash
# Links (or with --copy copies) skills/gpu-kernel-tuning into the skills directory of every agent found on this machine.
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)/skills/gpu-kernel-tuning"
copy=0; [ "${1:-}" = "--copy" ] && copy=1
# agent name | its home directory | its skills directory
while IFS='|' read -r agent home dir; do
  [ -d "$home" ] || continue
  dst="$dir/gpu-kernel-tuning"
  if [ -e "$dst" ]; then echo "$agent: $dst exists, left as it is"; continue; fi
  mkdir -p "$dir"
  if [ $copy = 1 ]; then cp -r "$src" "$dst"; else ln -s "$src" "$dst"; fi
  echo "$agent: $dst"
done <<EOF
Claude Code|$HOME/.claude|$HOME/.claude/skills
Codex CLI|$HOME/.codex|$HOME/.agents/skills
Cline|$HOME/.cline|$HOME/.cline/skills
Hermes Agent|$HOME/.hermes|$HOME/.hermes/skills
DeepSeek Harness|$HOME/.dsh|$HOME/.dsh/skills
oh-my-pi|$HOME/.omp|$HOME/.omp/agent/skills
pi|$HOME/.pi|$HOME/.pi/agent/skills
EOF
