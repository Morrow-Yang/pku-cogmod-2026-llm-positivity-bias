# RLHF Positivity Bias in Llama-3.1: Representation or Output Policy?

A cognitive-modeling study of why RLHF-tuned LLMs systematically rate fictional entities more positively than their base counterparts — and whether that shift lives in what the model "believes" or in how it speaks.

**Course:** PKU Spring 2026, *Introduction to Cognitive Modeling* (01603706 / 01630075), instructor Prof. Hang Zhang
**Author:** Mingtian Yang (杨明天), School of Psychological and Cognitive Sciences, PKU
**Submission components:** mini-conference (June 10) + final research report (June 17)

> This repository is a curated **showcase** of the project's deliverables — the manuscript, slides, poster, source code, and dataset. The author's daily working repo lives elsewhere; this one is rebuilt from a sync script at each deadline so the file tree stays clean.

## The story in one paragraph

When asked to rate fictional things on a 1–7 Likert scale (pseudoword entities, made-up concepts), Llama-3.1-8B-Instruct gives an average rating of **5.04** — substantially higher than the base model's **4.45**. RLHF appears to install a positivity bias worth roughly **+0.6 Likert points** on entities the model has no real information about. The natural question: is the model *representing* these entities more positively (motivated cognition), or *speaking* about them more positively (polite output policy)? We answer this with a paired forced-honesty + control-honesty diagnostic, a 2 × 2 × 2 factorial decomposition of the prompt that elicits the bias, a 7-model cognitive-model comparison with full identifiability checks, and hidden-state probes that localize the shift to transformer layers 14–31. The bias is dominated by a single prompt clause (a "negativity-license" instruction telling the model that negative ratings are acceptable); its absence accounts for **β\_L = −1.23**, more than twice any other factor.

## Headline numbers

| | |
|---|---|
| +0.6 Likert | RLHF positivity bias (Llama-3.1-8B-Instruct 5.04 vs base 4.45) |
| −0.39 Likert | Forced-honesty overshoot at M3 below base baseline (95% CI [−0.47, −0.31]) |
| β\_L = −1.23 | Negativity-license clause coefficient (≈ 2× β\_A, 7× β\_H) |
| 15 / 15 | Pseudoword targets showing the effect (p < 10⁻⁵) |
| N = 710 | Unique observations after dedup correction |
| Layers 14–31 | Where probes recover the rating-shift representation |

## Repository layout

```
.
├── manuscript/             Final manuscript (PDF) in English and Chinese, plus the AI-conference cut
│   ├── paper-en.pdf            English course paper (v5, ~44 pp)
│   ├── paper-zh.pdf            Chinese final report (~6 pp; June 17 submission base)
│   └── paper-conf-version.pdf  AI-conference version (~15 pp)
├── slides/                 Pedagogical / teaching slides
├── poster/                 Mini-conference poster (HTML + PNG)
├── code/                   All experiment source code (5 sub-pipelines)
├── data/                   Trial-level results bundled in-repo; large probe data documented for regen
├── references/             Bibliography with BibTeX + arXiv / DOI / publisher links
├── README.md               You are here
├── AI_USAGE.md             AI-tool usage statement (mandatory per course syllabus)
└── LICENSE                 MIT
```

## How to read this repo (suggested order)

1. **`manuscript/paper-en.pdf`** — start here for the full study (English, the canonical version).
2. **`manuscript/paper-zh.pdf`** — Chinese summary; basis for the final 4000-字 report.
3. **`poster/poster.png`** — one-page visual summary.
4. **`slides/teaching-deck.pdf`** — 48 slides walking through the project pedagogically (中文).
5. **`code/` and `data/`** — for anyone who wants to inspect or rerun the analyses.
6. **`AI_USAGE.md`** — required disclosure of AI-tool usage.

## How to reproduce

The work runs on a single GPU machine with Llama-3.1-8B-Instruct (Meta) and Llama-3.1-8B base, both downloaded from Hugging Face. No fine-tuning, no activation steering — only forward passes and (for Study 4) hidden-state extraction.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd code/<sub-pipeline>
bash run.sh   # or invoke individual scripts; see code/README.md
```

Cognitive-model fitting is CPU-bound and finishes in minutes. Hidden-state extraction (Study 4) requires a GPU and ~1 hour on an A100. See [`code/README.md`](code/README.md) for per-pipeline detail and [`data/README.md`](data/README.md) for dataset provenance.

## Acknowledgments

Built on [**OmegaWiki**](https://github.com/cuibinpku/OmegaWiki), the agentic research platform from PKU's DAIR Lab, which provided much of the research workflow tooling (literature ingestion, idea management, experiment scaffolding, paper drafting support). Llama-3.1-8B (base + Instruct) is from Meta. Cognitive-model formulations draw on Yoon et al. (2017, 2020), Frank & Goodman (2012), and Lefebvre et al. (2017); see [`references/README.md`](references/README.md) for the full list.

## License

MIT — see [`LICENSE`](LICENSE).
