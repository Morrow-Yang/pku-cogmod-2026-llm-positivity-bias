# AI Usage Statement

Per the course syllabus (*Introduction to Cognitive Modeling*, Spring 2026, AI Usage Policy): this statement discloses which AI tools were used in this project, for what purposes, and to what extent. All AI-generated outputs were reviewed by the author, who retains full responsibility for the correctness and intellectual integrity of the submitted work.

## Tools used

| Tool | Provider | Role |
|---|---|---|
| **Claude Code** (Anthropic) | Anthropic | Primary AI coding assistant; used through almost every phase of the project — code generation, paper drafting, statistical analysis planning, debugging, literature search, repository organization |
| **OmegaWiki** (built on Claude) | PKU DAIR Lab | Agentic research-workflow framework providing skills for paper ingestion, idea management, experiment scaffolding, draft generation; runs on top of Claude Code |
| **paperreview.ai** (GPT-based) | external service | Independent automated reviewer for the manuscript; used once for an external review pass that produced the 6.7 / 10 verdict and a feedback list, several of which were then addressed |

## Purposes and extent

### Code (extensive)
The experiment code in `code/` was largely drafted with Claude Code's assistance. The author specified experimental designs, hypotheses, and analysis logic in natural language; Claude generated runnable Python. The author then executed, validated outputs, and iterated on errors. Most analysis functions, plotting code, statistical fits (MLE, hierarchical Bayes, bootstrap), and the hidden-state probe pipeline went through this loop. Test code (`test_model*.py`) was also AI-drafted and reviewed.

### Manuscript drafting (extensive)
The English paper drafts went through multiple rounds of Claude-assisted writing and revision. The author provided the research questions, claims to make, results to report, and structural feedback; Claude generated and refined prose. The Chinese summary and pedagogical slides were similarly co-drafted. Section structure, figure placement, and citation choices were author-directed; phrasing-level work was largely AI-assisted.

### Statistical / analytical decisions (mixed)
Choices of which models to fit (the 7-model cognitive-model lineup, the 2 × 2 × 2 factorial decomposition, the forced-honesty + control-honesty diagnostic) were made through dialogue with Claude: the author surfaced research questions; Claude suggested candidate analyses and their tradeoffs; the author selected and approved each. Identifiability checks (parameter recovery, model recovery, hierarchical Bayes) were added on AI recommendation after the initial fits and validated by the author.

### Literature search and citation (mixed)
Initial reading list (Mahowald 2024, Binz & Schulz 2023, Tuckute 2024, Goldstein 2022, etc.) was author-curated. Subsequent literature was discovered via OmegaWiki's `/discover` skill and Claude's web search. The Lehr et al. (2025) replication study and Murthy et al. (2025) precedent were surfaced through AI-assisted search; the author then read both papers and judged their relevance.

### Repository / git operations (extensive)
Git workflow, commit messages, repository organization, the `tools/sync_showcase.sh` that produces this very repo — all author-directed but Claude-implemented.

### External review (one-time)
The manuscript was submitted once to paperreview.ai (a GPT-5.5-based academic-review service) for an independent reviewer-style critique. The 6.7 / 10 verdict and the specific suggestions are quoted accurately; the author selected which to address.

## What the author claims as their own

- The choice of research question (RLHF positivity bias as a cognitive-modeling object)
- The framing as a representation-vs-output-policy distinction
- The design of the forced-honesty + control-honesty diagnostic as the load-bearing causal probe
- All interpretations of results and final claims in the manuscript
- Judgment calls on what to keep, cut, or rephrase across drafts
- The decision to preserve negative findings (ΔBIC = −10 is modest) rather than overclaim
- Full responsibility for the correctness of the experimental procedure, the validity of the statistical claims, and the integrity of the data

The author is preparing to defend each design and methodological choice at the June 10 mini-conference and in any follow-up examination, having reviewed the AI-generated code, analyses, and prose in detail.

## What the author does *not* claim as their own

- The day-to-day code implementation details and most variable-level choices
- Most prose-level phrasing in the English paper
- The OmegaWiki framework itself (third-party; credited in [README.md](README.md) and [`OMEGAWIKI/`](#) in the working repo)
- The Llama-3.1-8B model (Meta)
- Cognitive-model formulations from Yoon, Frank-Goodman, Lefebvre, et al.

---

*Last updated when the showcase was last synced from the working repo. If this statement is out of date, see the sync timestamp on the most recent commit.*
