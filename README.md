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

Built on [**OmegaWiki**](https://github.com/skyllwt/OmegaWiki), the agentic research platform from PKU's DAIR Lab, which provided much of the research workflow tooling (literature ingestion, idea management, experiment scaffolding, paper drafting support). Llama-3.1-8B (base + Instruct) is from Meta. Cognitive-model formulations draw on Yoon et al. (2017, 2020), Frank & Goodman (2012), and Lefebvre et al. (2017); see [`references/README.md`](references/README.md) for the full list.

## License

MIT — see [`LICENSE`](LICENSE).
