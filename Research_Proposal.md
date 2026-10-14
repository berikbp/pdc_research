# Research Proposal

## Characterizing Tokenization-Induced Performance Variation in Heterogeneous CPU–GPU LLM Inference

**Course:** Parallel and Distributed Computing  
**Supervisor:** Mohammed Alaa Ala’anzy  
**Students:** Berik Satybaldy, Yerassyl Kairatuly, Yernazar Kaldybaev, Dias Maksatov  
**Institution:** SDU University

## Background and Motivation

Large Language Model (LLM) inference requires substantial computational power and memory. This makes local deployment difficult on consumer hardware, where GPU memory is much more limited than on data-center accelerators. During autoregressive generation, the key–value (KV) cache grows together with the sequence length, while model parameters and intermediate states must also remain in memory. One way to overcome this limitation is heterogeneous CPU–GPU inference, where computation and memory are shared between the GPU and the larger host memory available to the CPU.

Previous research has shown that CPU–GPU cooperation can significantly affect inference performance. PowerInfer uses a hybrid CPU–GPU design on consumer-grade hardware by placing frequently activated neurons on the GPU while executing less frequently activated neurons on the CPU [1]. Dovetail combines CPU execution with GPU-based speculative decoding for consumer devices [2]. APEX focuses on asynchronous CPU–GPU execution and dynamically schedules work according to estimated execution costs under constrained GPU memory [3]. Other studies have investigated latency-aware memory offloading and the performance characteristics of different CPU–GPU architectures [4, 5]. These works demonstrate that the way computation is distributed between heterogeneous processors can directly influence latency, throughput, and memory efficiency.

Multilingual workloads introduce another source of variation. The same semantic content can require very different numbers of tokens depending on the language and tokenizer. Petrov et al. show that equivalent text in different languages can have substantially different tokenized sequence lengths [6]. This is important because LLMs operate on tokens rather than directly on words or characters. More recent work such as MUTANT shows that improving multilingual tokenization efficiency can lead directly to higher inference throughput [7]. Therefore, language-dependent tokenization can change the amount of computation and memory required for a request.

Although heterogeneous inference and multilingual tokenization have both been studied extensively, the interaction between them is less clear. Existing heterogeneous systems generally make decisions using hardware state, memory pressure, sequence length, or predicted execution cost. At the same time, recent serving systems have started to use language as a routing feature. For example, Yoshimura et al. use language and context length when routing requests across different model instances in a distributed serving environment [8]. However, this is different from deciding how computation should be divided between the CPU and GPU inside a single memory-constrained machine.

## Research Problem and Questions

The main goal of this project is to determine whether multilingual tokenization differences affect the optimal CPU–GPU execution configuration for LLM inference.

Simply showing that Kazakh, Russian, Chinese, or English have different inference times would not be sufficient. If one language produces more tokens than another, a conventional scheduler that already considers sequence length may be able to handle this difference without explicitly considering language. Previous research has already demonstrated that token counts and workload size can be useful for workload-aware scheduling and energy optimization [9, 10]. Therefore, the important question is whether tokenizer-related characteristics provide useful information beyond ordinary sequence length.

The project will investigate two research questions:

**RQ1:** Do multilingual workloads with different tokenization characteristics exhibit different performance-optimal CPU–GPU execution configurations after controlling for tokenized sequence length?

**RQ2:** If such differences exist, can a tokenizer-aware or workload-aware configuration policy outperform both a fixed CPU–GPU configuration and a conventional sequence-length-aware policy?

<!-- End of source PDF page 1; source PDF page 2 begins below. -->

RQ1 is the central research question. If the differences between languages disappear after controlling for token count, then explicit language-aware scheduling may not be necessary. In that case, general workload-aware scheduling would already be sufficient. If the optimal configuration still changes after token count is controlled, this would suggest that additional tokenizer-derived workload characteristics are useful for CPU–GPU resource allocation.

## Methodology

The experiments will use open-source LLMs running on a consumer CPU–GPU system with limited GPU memory. The initial languages will be English, Russian, Kazakh, and Chinese. These languages provide different scripts, morphological structures, and tokenizer behavior while still allowing equivalent prompts to be created across languages.

Two types of workloads will be evaluated. First, semantically equivalent prompts will be translated into all selected languages. These experiments will show the real end-to-end effect of multilingual tokenization and generation behavior. Second, token-count-controlled workloads will be constructed so that the number of input tokens is approximately equal across languages. This second experiment is necessary to determine whether any remaining performance difference is caused by language-related workload characteristics rather than simply by longer sequences.

For each workload, several heterogeneous execution configurations will be tested. These may include GPU-only or GPU-dominant execution, multiple CPU–GPU layer or offload ratios, CPU-dominant execution where practical, different quantization levels, and different context lengths. Every experiment will be repeated several times to reduce measurement noise caused by warm-up effects, caching, background processes, and hardware temperature.

The measurements will include input and output token counts, token fertility, time-to-first-token, prefill latency, decode latency, total inference latency, tokens per second, peak GPU memory usage, host RAM usage, CPU utilization, and GPU utilization. Power or energy consumption may also be recorded if reliable hardware-level measurements are available.

For each workload $w$ and execution configuration $c$, the measured latency can be represented as $L(w, c)$. The latency-optimal configuration is then

$$
c^{*}(w) = \underset{c \in C}{\arg\min}\; L(w, c),
$$

where $C$ is the set of tested CPU–GPU configurations. A similar formulation can be used when maximizing throughput instead of minimizing latency.

The final analysis will compare three strategies: a fixed configuration for all requests, a conventional workload-aware configuration based mainly on sequence length and token count, and, if supported by the experimental results, a tokenizer-aware configuration that uses additional workload characteristics. The purpose of this comparison is to determine whether multilingual tokenization provides information that improves CPU–GPU execution decisions beyond what can already be explained by sequence length.

## Expected Contribution

The first expected contribution is a controlled characterization of multilingual LLM inference on memory-constrained heterogeneous CPU–GPU hardware. This will connect tokenizer behavior with systems-level properties such as latency, throughput, memory pressure, and processor utilization.

The second contribution is an empirical answer to whether multilingual workloads actually change the performance-optimal CPU–GPU configuration after sequence length is controlled. If a stable relationship is found, the project will additionally investigate a lightweight configuration-selection policy and compare it with fixed and length-aware baselines.

A negative result would also be meaningful. If sequence length explains nearly all of the observed variation, the study would show that explicit language-aware scheduling is unnecessary and that general workload-aware scheduling is sufficient.

Although multilingual LLMs provide the workload variation, the project is primarily a parallel-computing study. Its main focus is heterogeneous CPU–GPU execution, memory constraints, workload partitioning, resource utilization, and scheduling.

<!-- End of source PDF page 2; source PDF page 3 begins below. -->

## References

[1] Y. Song, Z. Mi, H. Xie, and H. Chen, “Powerinfer: Fast large language model serving with a consumer-grade gpu,” in *Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles*, ser. SOSP ’24. Association for Computing Machinery, 2024, pp. 590–606. [Online]. Available: https://doi.org/10.1145/3694715.3695964

[2] L. Zhang, Z. Zhang, B. Xu, R. Li, Z. Tian, S. Mei, and D. Li, “Dovetail: A CPU/GPU heterogeneous speculative decoding for LLM inference,” in *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*. Suzhou, China: Association for Computational Linguistics, 2025, pp. 17 382–17 395. [Online]. Available: https://doi.org/10.18653/v1/2025.emnlp-main.879

[3] J. Fan, Y. Zhang, X. Li, and D. S. Nikolopoulos, “APEX: Asynchronous parallel cpu–gpu execution for online llm inference on constrained gpus,” in *2026 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*. IEEE, 2026, pp. 1190–1203. [Online]. Available: https://doi.org/10.1109/IPDPS65963.2026.00099

[4] C. Ma, H. Zhao, Z. Ye, Z. Yang, T. Fu, J. Han, J. Zhang, Y. Luo, X. Wang, Z. Wang, Y. Li, and D. Zhou, “Latency-slo-aware memory offloading for large language model inference,” in *Proceedings of the 40th ACM International Conference on Supercomputing*, ser. ICS ’26. Association for Computing Machinery, 2026, pp. 919–932. [Online]. Available: https://doi.org/10.1145/3797905.3807874

[5] P. Vellaisamy, T. Labonte, S. Chakraborty, M. Turner, S. Sury, and J. P. Shen, “Characterizing and optimizing llm inference workloads on cpu–gpu coupled architectures,” in *2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*. IEEE, 2025, pp. 49–61. [Online]. Available: https://doi.org/10.1109/ISPASS64960.2025.00015

[6] A. Petrov, E. L. Malfa, P. H. S. Torr, and A. Bibi, “Language model tokenizers introduce unfairness between languages,” in *Advances in Neural Information Processing Systems*, vol. 36, 2023, pp. 36 963–36 990. [Online]. Available: https://doi.org/10.52202/075280-1608

[7] S. Rana, A. Kulkarni, A. Menezes, C. Khatri, and S. Agarwal, “MUTANT: A recipe for multilingual tokenizer design,” in *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. San Diego, California, United States: Association for Computational Linguistics, 2026, pp. 46 261–46 277. [Online]. Available: https://doi.org/10.18653/v1/2026.acl-long.2146

[8] T. Yoshimura, V. D. van de Beek, and T. Chiba, “Accuracy is speed: Towards long-context-aware routing for distributed llm serving,” in *Proceedings of the Sixth European Workshop on Machine Learning and Systems*, ser. EuroMLSys ’26. Association for Computing Machinery, 2026, pp. 91–98. [Online]. Available: https://doi.org/10.1145/3805621.3807652

[9] G. Wilkins, S. Keshav, and R. Mortier, “Hybrid heterogeneous clusters can lower the energy consumption of llm inference workloads,” in *Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems*, ser. e-Energy ’24. Association for Computing Machinery, 2024, pp. 506–513. [Online]. Available: https://doi.org/10.1145/3632775.3662830

[10] ——, “Offline energy-optimal llm serving: Workload-based energy models for llm inference on heterogeneous systems,” *ACM SIGENERGY Energy Informatics Review*, vol. 4, no. 5, pp. 113–119, Apr. 2025. [Online]. Available: https://doi.org/10.1145/3727200.3727217

<!-- End of source PDF page 3. -->
