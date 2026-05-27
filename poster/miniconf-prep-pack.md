# Mini-Conference Prep Pack (Bilingual)

Course: PKU Intro to Cognitive Modeling, Spring 2026
Mini-conference: 2026-06-10
Final report: 2026-06-17

---

## 1. 中英术语对照 / English–Chinese Term Map

| English | 中文 | 使用建议 |
|---|---|---|
| Large language model (LLM) | 大语言模型 | 讲解时直接说 LLM |
| Instruction-tuned | 指令微调的 | |
| Post-training | 后训练 | |
| RLHF | 人类反馈强化学习 | 直接说 RLHF |
| SFT / DPO | — | 直接说英文缩写 |
| Motivated cognition | 动机性认知 | Kunda 1990 的术语 |
| Polite speech / politeness mask | 礼貌性表达 / 礼貌面具 | |
| Forced honesty | 强制诚实 / 强制坦白 | |
| Control honesty | 控制诚实 | 中性准确性指令 |
| Below-base overshoot | 低于基线的过冲 | 也可直接说 overshoot |
| Mask sensitivity | 面具敏感性 | $C_5 - C_8$ 的别名 |
| Anti-politeness specific | 反礼貌特异成分 | |
| Chat template | 对话模板 | 直接说 chat template |
| Calibration veto | 校准否决 | |
| Baseline-artifact veto | 基线伪影否决 | |
| Recipe confound | 配方混淆 | M2→M3 的不可分离性 |
| Cluster bootstrap | 聚类自举 | 直接说 cluster bootstrap |
| Tülu-3 mediation chain | Tülu-3 中介链 | |
| Dose-response | 剂量-反应 | |

**策略**: LLM, RLHF, SFT, DPO, prompt, baseline, overshoot, chat template, bootstrap, Likert 这些直接说英文。叙事、连接词、解释用中文。

---

## 2. 30 秒电梯演讲 / 30-Second Elevator Pitch

### 中文版
> 指令微调的 LLM 对虚构/陌生实体给出强正面评价——这究竟是表征层面的"动机性认知"，还是输出层面的"礼貌性表达"？我们设计了一个简单的诊断：给模型加一条 system prompt——"你必须完全诚实,不要礼貌"。如果评分跌到 base model 基线之下（我们叫 overshoot），就说明偏置是在表达层面。在 M3 上,overshoot 出现了（−0.39 Likert, CI 不跨零）。进一步在 Tülu 链上做定位,发现 SFT 和 DPO 安装了可去除的礼貌面具,但 overshoot 这个信号只在 M3 + chat template + 负面表达许可 同时满足时才出现。

### English version
> Instruction-tuned LLMs rate novel/fictional entities strongly positive. Is this motivated cognition (representation-level belief) or polite speech (output-level default)? We add one system prompt — "be completely honest, don't be polite" — and check whether ratings drop below the base-model baseline. On M3 (Llama-3.1-8B-Instruct), they do (−0.39 Likert, CI strictly below 0). Across the Tülu-3 chain, SFT/DPO install a removable politeness mask, but the below-base overshoot signature requires chat-template activation + explicit negative-expression licensing — a conjunctive mechanism.

---

## 3. Poster 逐区讲解 / Section-by-Section Walk-Through (~3 min)

### Introduction (~25 sec)
> 🇨🇳 "LLM-as-cognitive-models 这个方向现在很火——把 LLM 当认知主体来拟合经典模型。但有个根本风险：RLHF 后训练带来的表达层面偏置，可能被误读为内在的认知现象。我们这个工作就是提供一个第一线诊断工具。"
>
> 🇬🇧 "The LLM-as-cognitive-models program fits classical cognitive models to LLM behavior. The foundational risk: post-training artifacts may be misattributed to internal beliefs. We provide a first-line diagnostic."

### Method (~30 sec, point at TikZ chain)
> 🇨🇳 "我们用 Tülu-3 的公开 checkpoint：M0 base → M1 SFT → M2 DPO，这三个是一条因果链；M3 是 Meta 自己的 pipeline，跟 Tülu 链只在 M0 相接——M2→M3 不是因果一步，是 recipe transition。评分用 logprob 在 digit 1-7 上的概率分布提取，不靠生成。"
>
> 🇬🇧 "Tülu-3 provides intermediate checkpoints on the same Llama-3.1-8B base. Within-Tülu (M0→M1→M2) is a clean causal chain. M3 is Meta's parallel pipeline — the M2→M3 step is a recipe comparison, not a causal stage. Ratings via logprob-based Likert extraction."

### Experiments (~70 sec, point at table → fig3 → fig4 → Round-2 table)
> 🇨🇳 "先看 M3 主结果——表里三个 headline delta：总效应 −0.98，anti-politeness 特异成分 −0.56，overshoot −0.39。Control-honesty 把效应拆成 43% 一般指令 + 57% 反礼貌特异。"
>
> "fig3 是 M3 上 forced vs control 散点图，橘色区是 overshoot zone——红色方块在里面，蓝色圆圈不在。"
>
> "fig4 把诊断扩展到整个 Tülu 链：左面板 anti-politeness share 在 M2 最大（−1.01），右面板 overshoot 只在 M3 出现。SFT 和 DPO 被排除为 overshoot 的原因。"
>
> "右下 Round-2 表回答了 reviewer 的关键问题：同样 prompt 在 completion 模式下 overshoot 消失，但 anti-politeness share 还在。Chat template 是 overshoot 的必要成分。"

### Discussion (~30 sec)
> 🇨🇳 "两层机制：Tülu 安装了可去除的礼貌面具（M2 最强）；M3 额外安装了一个依赖 chat template 激活 + 负面许可的'准许过冲'模式。Kunda 的预注册认知模型在所有 7 个 criteria 上失败。结论：拟合认知模型之前，应先做 forced-honesty + control-honesty dissociation。"

### Conclusion (~25 sec)
> 🇨🇳 "方法学建议：LLM-as-cog-models 研究在拟合前做一次诊断。我们 release 代码、prompt、token ID、原始结果。Future work：linear probing、M2→M3 分解、跨模型族复制。"

---

## 4. 预期 Q&A / Anticipated Questions

### Q1. 为什么用 base model 当"无偏基线"？它自己就没有偏置吗？
> **答**: Pragmatic choice，不是 theoretical claim。Base 也有 pretraining prior。但 baseline-artifact veto 证明 prompt 本身不拉低 base（−0.04, CI 跨零），calibration veto 证明 forced-honesty 不会 uniformly 压低所有评分（水保持高、蛀牙保持低、砖保持中性）。所以"用 base 当基线"虽然不完美，替代解释都已排除。
>
> **Answer**: A pragmatic choice. Base models carry pretraining priors too. But the baseline-artifact veto (M0+forced = null) and calibration veto (valence-appropriate ratings preserved) jointly rule out the main alternative explanations.

### Q2. 这跟 sycophancy 有什么区别？
> **答**: Sycophancy 是 user-relative bias（模型迎合用户意见）。我们的 positivity prior 是 user-independent（用户没表达任何意见）。Forced-honesty 不提用户，所以不是在测 sycophancy。两种现象的诊断信号也不同：sycophancy 的 forced-honesty 效应不会 target-specific，但我们观察到 target-specific overshoot。
>
> **Answer**: Sycophancy = user-relative bias (model agrees with user). Our positivity prior = user-independent (no user opinion stated). Different mechanisms, different diagnostic signatures.

### Q3. Forced-honesty 是不是就是个 jailbreak？
> **答**: 这是 control-honesty 设计的核心动机。如果只是 jailbreak，任何强指令都应产生类似下降。Control-honesty 只产生总效应的 43%，且不过 base。量化区分——一个通用指令成分 + 一个反礼貌特异成分，只有后者过 base——是我们 falsify "just a jailbreak" 的关键证据。
>
> **Answer**: The control-honesty condition directly addresses this. Generic instruction-override accounts for only 43% of the total drop and does not overshoot. The anti-politeness-specific 57% is separable and is the only component that drives below-base ratings.

### Q4. 为什么 Tülu chain 特别有价值？
> **答**: 直接对比 M0 vs M3 confounded（5+ 因素同时变化）。Tülu 公开了中间 checkpoint 让我们做 per-stage mediation。结果 SFT+DPO 装不出 overshoot——所以 overshoot 不能简单归因为"指令微调"。
>
> **Answer**: M0-vs-M3 conflates 5+ factors. Tülu's intermediate checkpoints enable per-stage causal mediation within a shared recipe. Finding: SFT and DPO are ruled out as proximate causes of the overshoot.

### Q5. M2→M3 的 recipe confound 具体是什么？
> **答**: M3 不是从 M2 来的——Meta 用自己的 SFT 数据、DPO 偏好、RLHF、safety tuning、不同的 chat template 从 M0 重新训出来。所以 M2→M3 混淆 5+ 因素。Round-2 证明 chat template 是其中一个必要因素——拿掉它 overshoot 消失——把 confound 缩小到 4 个剩余因素。
>
> **Answer**: M3 is Meta's own pipeline from M0. M2→M3 conflates 5+ factors. Round-2 isolates chat-template as one necessary contributor, narrowing the confound to 4 remaining factors.

### Q6. Kunda 模型为什么失败？
> **答**: 3 个结构性原因：(1) 方向不对称——虚构 target 正向、Putin 负向，单方向 motivation parameter 没法 fit；(2) 轨迹非单调——M2 > M3；(3) forced-honesty 与 representation-level 假设冲突——prompt 不应该把 belief 推到 base 以下。
>
> **Answer**: Direction asymmetry (fictional positive, Putin negative); non-monotonic trajectory (M2 > M3); forced-honesty contradicts representation-level mechanism (prompt shouldn't push beliefs below base).

### Q7. Round-2 的 "chat template 必要性" 具体是什么？
> **答**: 同一个 checkpoint、同一个 prompt，chat template 模式 overshoot −0.39；completion 模式 +0.16（不 overshoot）。但 completion 模式的 anti-politeness share 还有 −0.33。所以 chat template 是 overshoot 的必要条件，但面具去除本身不依赖它。
>
> **Answer**: Same checkpoint, same prompt. Chat mode → overshoot (−0.39). Completion mode → no overshoot (+0.16), but anti-politeness share still present (−0.33). Chat template is necessary for the below-base signature specifically.

### Q8. 能推广到其他模型吗？
> **答**: 诊断方法本身是 method-general（只需要 chat-tuned 模型暴露 logprob）。但具体的 decomposition 数字和 recipe-step localization 可能是 pipeline-specific。跨模型族复制是 future work 第一项。选 Llama-3.1-8B 是因为 Tülu 公开了中间 checkpoint。
>
> **Answer**: The diagnostic is method-general. Specific magnitudes and localization may be pipeline-specific. Cross-family replication is the #1 future-work item. We chose Llama-3.1-8B because Tülu provides the only publicly available intermediate checkpoints.

---

## 5. Lightning Talk 幻灯片大纲 / Slide Outline (5 slides, ~5 min)

| Slide | Title | 中文叙述要点 | Key visual |
|---|---|---|---|
| 1 | Title + affiliation | "我们研究指令微调 LLM 的正面性偏置是表达层面还是表征层面的" | 标题 + PKU 校徽 |
| 2 | The phenomenon + question | "M3 对虚构实体评 5.04/7，base 只有 4.45。看起来是 motivated cognition，但也可能是 polite speech。这个区分对 LLM-as-cog-models 有方法论意义。" | 5.04 vs 4.45 对比图 |
| 3 | The diagnostic | "加 forced-honesty → 如果 motivated cognition, 不能过 base；如果 polite speech, 可以过。Control-honesty 拆分 generic vs anti-politeness。" | fig1 (TikZ chain) + prompt text |
| 4 | M3 result | "M3 上 overshoot −0.39; 43% generic + 57% anti-politeness。只有 57% 过 base。" | fig3 (scatter) + decomposition table |
| 5 | Per-stage + Round-2 | "Tülu 的 SFT/DPO 装不出 overshoot; chat template 是必要的; 负面许可也是。两层机制。建议：认知模型拟合前先做这个 dissociation。" | fig4 (trajectory) + table4 (chat vs completion) |

---

## 6. 时间线建议 / Timeline to Mini-Conf

| Date | Task |
|---|---|
| 本周 (5/20–5/25) | 熟读术语表 + 用中文复述 30 秒 pitch 3 遍 |
| 下周 (5/26–6/1) | 对着镜子 / 室友练 3 分钟 poster 讲解 + 口答 8 个 Q&A |
| 6/2–6/3 | 海报送印刷店（用 poster.html → Cmd+P → PDF → Print shop）|
| 6/4–6/6 | 如果有 lightning talk → 准备 5 张 beamer slides |
| 6/8–6/9 | 最终练习；带齐 poster PDF(手机+U盘) + paper PDF 备用 |
| 6/10 | Mini-conference 🎤 |
| 6/17 | 提交 final report (paper/main.pdf, 26 pages) |

---

*Generated by Claude Code from OmegaWiki, 2026-05-20.*
