# References

Key references for this project, organized by role. Full BibTeX is in [`references.bib`](references.bib) alongside this file in the working repository's `paper/` directory; the entries below give arXiv / DOI / publisher links for direct access.

PDFs are not redistributed here for copyright reasons. The full paper at [`../manuscript/paper-en.pdf`](../manuscript/paper-en.pdf) contains the complete citation list in its bibliography.

## Primary precedent (the paper we extend)

- **Lehr, S. A., Saichaie, K., Anwar, U., et al. (2025).** *Kernels of selfhood: GPT-4o shows humanlike patterns of cognitive dissonance moderated by free choice.* PNAS Nexus.
  - Introduces the pseudoword-rating paradigm we replicate and extend.

- **Murthy, A., Rufino, A. R. T., Yoon, E. J., Cohen, J. D. (2025).** *Cognitive models reveal interpretable value tradeoffs in large language models.* NeurIPS 2025.
  - arXiv: [2506.20666](https://arxiv.org/abs/2506.20666). Direct precedent for fitting Yoon-style cognitive models to LLM behavior; tests the same M3 prompt format we use.

## Cognitive-model formulations

- **Frank, M. C., & Goodman, N. D. (2012).** *Predicting pragmatic reasoning in language games.* *Science*, 336(6084), 998.
  - DOI: [10.1126/science.1218633](https://doi.org/10.1126/science.1218633). Source of the Rational Speech Acts framework that underpins our M2 / M3 polite-speech models.

- **Yoon, E. J., Tessler, M. H., Goodman, N. D., & Frank, M. C. (2017, 2020).** *Polite Speech Emerges From Competing Social Goals.* Open Mind / CogSci.
  - The polite-speech RSA model that our M3 variant is built on.

- **Lefebvre, G., Lebreton, M., Meyniel, F., Bourgeois-Gironde, S., & Palminteri, S. (2017).** *Behavioural and neural characterization of optimistic reinforcement learning.* *Nature Human Behaviour*, 1, 0067.
  - DOI: [10.1038/s41562-017-0067](https://doi.org/10.1038/s41562-017-0067). Optimistic-RL family informing one of our 7 cognitive-model candidates.

- **Sharot, T. (2011).** *The optimism bias.* *Current Biology*, 21(23), R941–R945.
  - DOI: [10.1016/j.cub.2011.10.030](https://doi.org/10.1016/j.cub.2011.10.030). Background framing for the asymmetric-update hypothesis we test against.

## LLM cognitive-modeling background

- **Binz, M., & Schulz, E. (2023).** *Using cognitive psychology to understand GPT-3.* *PNAS*, 120(6), e2218523120.
  - DOI: [10.1073/pnas.2218523120](https://doi.org/10.1073/pnas.2218523120). Methodological template for treating LLMs as behavioral subjects.

- **Mahowald, K., Ivanova, A. A., Blank, I. A., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2024).** *Dissociating language and thought in large language models.* *Trends in Cognitive Sciences*, 28(6), 517–540.
  - DOI: [10.1016/j.tics.2024.01.011](https://doi.org/10.1016/j.tics.2024.01.011). Framing for the representation-vs-output-policy distinction.

- **Tuckute, G., Kanwisher, N., & Fedorenko, E. (2024).** *Language in brains, minds, and machines.* *Annual Review of Neuroscience*, 47, 277–301.
  - DOI: [10.1146/annurev-neuro-120623-101142](https://doi.org/10.1146/annurev-neuro-120623-101142). Survey of probing methods we draw on for Study 4.

- **Goldstein, A., Zada, Z., Buchnik, E., et al. (2022).** *Shared computational principles for language processing in humans and deep language models.* *Nature Neuroscience*, 25(3), 369–380.
  - DOI: [10.1038/s41593-022-01026-4](https://doi.org/10.1038/s41593-022-01026-4). Neural-alignment precedent for layer-wise hidden-state probing.

- **Binz, M., Akata, Z., Bethge, M., & Schulz, E. (2024).** *Turning large language models into cognitive models.* ICLR 2024.
  - arXiv: [2306.03917](https://arxiv.org/abs/2306.03917).

## Models

- **Meta AI (2024).** *The Llama 3 Herd of Models.* arXiv: [2407.21783](https://arxiv.org/abs/2407.21783).
  - Documentation for the Llama-3.1 model family we evaluate.

## Statistical / methodological

- **McElreath, R. (2020).** *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). Chapman & Hall / CRC.
  - Course textbook; framework for the hierarchical Bayesian fits in Study 3.

- **Farrell, S., & Lewandowsky, S. (2018).** *Computational Modeling of Cognition and Behavior.* Cambridge University Press.
  - Course textbook; framework for parameter recovery and model recovery (the identifiability battery).

---

For the complete bibliography as cited in the manuscript, see the `\bibliography` block in `../manuscript/paper-en.pdf`.
