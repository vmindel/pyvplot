# Claude Code skill

`pyvplot` ships an [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) for
[Claude Code](https://claude.com/claude-code) that teaches the agent to use the library
correctly. It covers the API contracts, the assay physics and defaults that silently change
results, and common multi-step workflows.

## Install

Claude Code doesn't scan installed Python packages for skills, so copy the skill once into
your personal skills directory:

```bash
pyvplot-install-skills
```

This installs it to `~/.claude/skills/pyvplot/`, where Claude Code can use it in every
project from the next session on.

| command | effect |
|---|---|
| `pyvplot-install-skills` | install to `~/.claude/skills/pyvplot` |
| `pyvplot-install-skills --dest .claude/skills/pyvplot` | install into the current project only |
| `pyvplot-install-skills --force` | overwrite, e.g. after upgrading `pyvplot` |
| `pyvplot-install-skills --print-path` | print the bundled copy's location |

Once it's installed, ask Claude Code for `pyvplot` tasks in plain language — for example:
*"build a cached FragmentStore from this ChEC-seq BAM"*, *"compute a differential V-plot of
mutant vs WT around these moodspy CTCF hits with expected-background normalization"*, or
*"plot heatmaps of TF-sized and nucleosome-sized fragments around these motifs and estimate the
nucleosome repeat length"*. It reads the skill automatically.

## What's inside

The skill is a short router, `SKILL.md`, that sends the agent to one topic file under
`references/`. The agent reads that file only when the task needs it, so the skill costs
almost no context until it's used.

| file | covers |
|---|---|
| `references/store-ingestion.md` | BAM (`read1`/`mate`), BED/`.frag.tsv.gz`/BEDPE, `.npz`/`.parquet` caches, dedup, filtering |
| `references/vplots.md` | `vplot()`, CPM & expected/background normalization, smoothing, `diff()`, `VPlotCollection` |
| `references/matrices-footprints.md` | `to_matrix()`, per-anchor heatmaps, 1-bp `+`/`-` cut footprints, `footprint_score()` |
| `references/assays-coordinates.md` | Assay presets, `+4/-5` Tn5 shift, 5'→3' strand flipping, ChIP-exo, asymmetric windows |
| `references/qc-phasing.md` | Insert-size distribution, FRiP, nucleosome phasogram, NRL estimation, `qc_summary()` |
| `references/bias-export.md` | `KmerBiasTable` bias modeling/correction, control normalization, bedGraph / BigWig / `pymisha` |
| `scripts/check_extract.py` | Independent brute-force verification of `vplot` and `to_matrix` across `+` and `-` strands |

Each reference file highlights footguns — such as not double-shifting pre-shifted `.frag.tsv.gz`
ATAC files, half-open `[min_len, max_len)` V-plot length bins vs. inclusive `len_range`
filters, nucleosome dominance without expected-background normalization, and the opposite
sign conventions of `protection_score` vs. `footprint_depth`.

The test suite checks the skill:

- every routed file exists;
- every reference file is routed from `SKILL.md`;
- every `vp.<name>` it mentions is a real export on `pyvplot`;
- the `pyvplot-install-skills` CLI and Python installer work;
- `scripts/check_extract.py --synthetic` passes with `0 mismatches`.
