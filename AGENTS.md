# AGENTS.md — model-vram-calc

## Project Overview

Single-file static HTML app (`index.html`) for estimating LLM VRAM usage, KV Cache, and cluster concurrency. No build step, no dependencies.

## How to Run

Open `index.html` directly in a browser. No dev server needed.

## Architecture

- **Single entry point**: `index.html` (796 lines) — all HTML, CSS, and JS in one file
- **No framework**: vanilla HTML5 + CSS3 + JavaScript
- **No build tooling**: no package.json, no bundler, no transpiler

## Key Files

| File | Purpose |
|------|---------|
| `index.html` | Entire application |
| `VRAM_CALC.md` | Formula reference (weights, KV Cache, overhead, concurrency) |
| `README.md` | User-facing docs |

## Calculation Logic (JS in `index.html`)

- **Model presets**: 12 models (DeepSeek, GLM, Qwen, Minimax) with hardcoded params
- **GPU presets**: 7 GPUs (H200, H20, B200, RTX 5090/4090, A710E, 真武810E)
- **Quantization map**: FP16/BF16=1, FP8/INT8=0.5, FP4/INT4=0.25
- **KV Cache formula**: `2 × layers × (hiddenDim × kvHeads/attnHeads) × contextLen × bytesPerElem / 1e9`
- **Auto-calc on load**: `calculate()` runs once at init

## Conventions

- Dark theme UI with glassmorphism effects
- Responsive at 4 breakpoints: 960px, 768px, 480px, 374px
- Chinese UI labels, code comments in English

## Git

- Single branch: `main`
- Shallow clone (`.git/shallow` exists)
