# MIRROR 的 Rebuttal (1)

> #### Reviewer 1：[2/3]
>
> **Critical Weaknesses:**
>
> 1. The novelty of the method is not sufficiently established. MIRROR builds on the idea of constructing a teacher distribution from reference information and performing self-distillation along the student’s own generation trajectories. The proposed approach seems close to prior work on on-policy self-distillation, such as Zhao et al. (2026). The authors need to clarify what is genuinely different about MIRROR with those prior works. Using an existing paradigm for personalized generation does not make the method novel.该方法的新颖性未得到充分确立。MIRROR基于从参考信息构建教师分布并在学生的生成轨迹上执行自蒸馏的想法。提出的方法似乎接近于关于按策略自蒸馏的先前工作，如Zhao等人（2026年）。作者需要阐明MIRROR与那些先前工作真正不同的地方。使用现有的个性化生成范式并不使该方法具有新颖性。
> 2. The experiments do not directly test whether the proposed method is necessary. For instance, would filtering for high-quality reference samples, or supervising only the reference tokens relevant to content quality, can get similar improvement?实验并未直接测试所提出的方法是否必要。例如，过滤高质量参考样本，或者仅监督与内容质量相关的参考标记，能否获得类似的改进？
> 3. The evidence for improved content quality remains insufficient. The paper claims that MIRROR improves the completeness, and professionalism of generated content. But this evaluation just relies on text similarity metrics and LLM-as-a-judge assessments. Overall text quality scores do not directly reflect improvements in factual accuracy, completeness, or domain expertise. The current experimental results therefore provide insufficient support for the paper’s central claims about the reliability and professional quality of generated content.论文声称MIRROR提高了生成内容的完整性和专业性。但这种评估仅依赖于文本相似性指标和LLM作为评判者的评估。整体文本质量分数并不能直接反映事实准确性、完整性或领域专业知识方面的改进。
>
> **Questions For Authors:**
>
> 1. How does MIRROR differ fundamentally from prior on-policy self-distillation methods? What specific technical innovation does MIRROR introduce beyond applying reference-conditioned self-distillation to personalized generation?MIRROR 与以往的策略级自我精炼方法在本质上有何不同？除了应用基于参考条件的自我精炼技术之外，MIRROR 还引入了哪些具体的技术创新？
> 2. Can the authors demonstrate the necessity of the proposed training objectives through comparisons with simpler alternatives? For example, how would MIRROR-F compare with selective token-level supervision applied directly to the reference responses, without on-policy distributional alignment?作者能否通过与其他更简单的替代方案的比较，来证明所提出的训练目标的必要性呢？例如，MIRROR-F 与直接应用于参考响应的选择性令牌级监督方案相比，会有何差异？在应用令牌级监督方案时，是否需要保持策略分布的一致性呢？
> 3. How do the authors verify the claimed improvements in content quality and reliability? Since the current evaluation depends heavily on automatic metrics and a single LLM judge, could they provide more targeted assessments of factual accuracy, completeness, and domain expertise—or otherwise establish that the gains are genuine improvements in content quality, not merely better alignment with the reference or the judge model’s preferences?作者是如何验证所声称的内容质量和可靠性提升的？由于当前的评估很大程度上依赖于自动化指标和单一的 LLM 评判者，他们能否提供更具体的评估，以判断事实准确性、完整性以及领域专业知识方面的情况？或者至少能证明这些提升确实是内容质量的真正改善，而不仅仅是为了更好地符合参考标准或评判者的偏好。

> #### Reciewer 2:[2/3]重点 Rebuttal
>
> **Critical Weaknesses:**
>
> 1. **The teacher and training setup are unclear.** Eq. (3) uses the current student as teacher, while Figure 2 and the code describe a frozen teacher. The scripts also allow differences in teacher size, input length, and training exposure. These defaults do not establish what produced the tables, but a different teacher or more context could improve performance, while less training could reduce forgetting. The actual configurations and comparisons under comparable conditions are needed.关于教师和训练机制的具体设置尚不清楚。方程(3)中将当前学生视为教师，而图 2 和代码则描述了一个固定的教师角色。脚本还允许教师的大小、输入长度以及训练经验存在差异【已回复】。这些默认设置并未明确说明是什么产生了这些结果，但使用不同的教师或提供更多上下文可能会提升性能，而较少的训练则有助于减少遗忘现象。需要实际配置以及在相似条件下的比较结果才能得出更准确的结论。
> 2. **The added benefit and stability of selective anchoring need clearer evidence.** [OPSD v1](https://arxiv.org/html/2601.18734v1) already studies reference-conditioned on-policy distillation, and [SDFT](https://proceedings.mlr.press/v306/shenfeld26a.html) reports reduced forgetting. MIRROR-F’s consistent gains are promising, but do not show whether selection helps beyond adding reference supervision. A matched comparison with uniform supervision on all reference tokens would clarify this. The main tables also lack uncertainty, and content quality does not improve in every setting: 4B news C&S is 3.125 versus 3.273 for Base. These limits should be reflected in the claims.选择性锚定的附加好处和稳定性需要更清晰的证据。OPSD v1已经研究了基于参考条件的在线策略蒸馏，SDFT报告了减少遗忘。MIRROR-F的一致增益有希望，但并未显示选择是否有助于除了添加参考监督之外。与所有参考标记上均匀监督的匹配比较将阐明这一点。主要表格也缺乏不确定性，内容质量并非在每种设置中都会提高：4B新闻C&S为3.125，而Base为3.273。这些限制应反映在声明中。
>
> **Questions For Authors:**
>
> 1. **Clarification:** Which teacher checkpoint and update rule produced each main result? Please reconcile Eq. (3), Figure 2, and the released implementation, and report the actual context limits, effective batch sizes, and examples or tokens processed for each method. Please also resolve the 150 versus 200/250 training-step descriptions.澄清：哪个教师检查点和更新规则产生了每个主要结果？请对照式（3）、图2和发布的实现，报告实际上下文限制、有效批量大小以及每个方法的示例或标记处理。请还解决150比200/250训练步骤的描述。
> 2. **Validation of the claimed contribution:** Does selective anchoring outperform uniform supervision on all reference tokens under matched conditions? If this comparison has already been conducted, please report the results and their uncertainty.验证所声称的贡献：在匹配条件下，选择性锚定是否优于均匀监督在所有参考标记上？如果此比较已经进行，请报告结果及其不确定性。
> 3. **Validation of the main results:** Please provide the per-run results or uncertainty for the main performance and retention differences. If the actual teacher, input, or training conditions differed across methods, do the conclusions hold in a representative controlled comparison?验证主要结果：请提供每次运行的主要性能和保留差异的结果或不确定性。如果实际教师、输入或训练条件在不同方法之间存在差异，代表性受控比较中的结论是否成立？
>
> Resolving the configuration questions and showing a stable benefit from selective anchoring would move my recommendation toward acceptance.解决配置问题并从选择性锚定中获得稳定的收益将使我的建议更倾向于接受。

> #### Reciewer 3:[2/3]
>
> **Critical Strengths:**
>
> 1. The formulation is simple and practical: on-policy self-distillation is conceptually a good fit for the personalization task.公式简单实用：在个性化任务中，按策略自蒸馏从概念上讲是一个很好的匹配
> 2. The evaluation covers out-of-distribution forgetting, with positive results.评估涵盖了分布外遗忘，结果积极。
>
> **Critical Weaknesses:**
>
> 1. It is unclear where the performance improvements come from. Although MIRROR is simple and principled, it seems to have limited performance gain without the additional token selector and NLL loss in MIRROR-F. But the token selector in MIRROR-F is a rather hard-engineered heuristic with little principled justification behind it. The absence of an ablation of α = 0 or 1 makes this more concerning.性能提升的来源不明确。尽管MIRROR简单且遵循原则，但似乎没有额外的token选择器和MIRROR-F中的NLL损失，性能提升有限。但MIRROR-F中的token选择器是一个相当硬编码的启发式方法，其背后的原则性依据很少。α=0或1的消融实验缺失使这个问题更加令人担忧。
> 2. Experiments are not entirely well controlled. No confidence interval is included. And the tuning procedure for baselines is under-specified; several baselines score below Base in Table 1. Only one LLM family is used, at a rather small scale, and the judge is from the same family.实验控制并不完全良好。没有包括置信区间。基准的调整过程未明确说明；几个基准在表1中的得分低于Base。只使用了一个LLM家族，规模相当小，评委也来自同一家族。
> 3. The OOD experiment could be biased. The OOD experiment is evaluated with a metric that rewards the smallest deviation from Base, which systematically favors the self-distillation-based method like MIRROR.OOD实验可能存在偏差。OOD实验使用了一个奖励与Base最小偏差的指标，系统地偏向基于自蒸馏的方法，如MIRROR。
