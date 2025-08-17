import 'package:dfc_llama/src/llama.dart';
import 'package:dfc_llama/src/llama_cpp.dart';

// RoPE scaling types
enum LlamaRopeScalingType {
  unspecified(-1),
  none(0),
  linear(1),
  yarn(2),
  maxValue(2);

  const LlamaRopeScalingType(this.value);

  final int value;
}

// ===============================================================

// Pooling types for embeddings
enum LlamaPoolingType {
  unspecified(-1),
  none(0),
  mean(1),
  cls(2),
  last(3),
  rank(4);

  const LlamaPoolingType(this.value);

  final int value;
}

// ===============================================================

// Attention types for embeddings
enum LlamaAttentionType {
  unspecified(-1),
  causal(0),
  nonCausal(1);

  const LlamaAttentionType(this.value);

  final int value;
}

// ===============================================================

// ContextParams holds configuration settings for the Llama model context
class ContextParams {
  ContextParams();

  // controls the context window size
  // 2,048 – 8,192 is reasonable; this handles a substantial number of messages without high RAM use.
  int nCtx = 2048;

  // The maximum number of tokens to be processed at once at the API/application level.
  // In chat AI, this means how many tokens you send to the model to decode in a single batch.
  // Common defaults for n_batch are 32, 64, 128, or even 512 for larger systems
  // Use moderate values (n_batch=32–128, n_ubatch=8–64) for most chat AI setups, tuning for your hardware and prompt needs.

  int nBatch = 64;

  // Divides n_batch into smaller chunks that fit the compute buffer; helps manage memory, especially on GPUs
  // Typical values: Usually set lower than n_batch; 8, 16, 32, or 64 are common, depending on
  // your hardware. n_ubatch must always be less than or equal to n_batch.
  // Setting n_ubatch high can speed things up but uses more device memory; set lower if you hit
  // out-of-memory errors on your hardware.
  // For chat AI on consumer hardware: Try n_batch=64 and n_ubatch=16 or 32. Increase n_batch for longer prompts if your hardware allows.
  // Always ensure: n_ubatch ≤ n_batch.
  int nUbatch = 16;

  // Max number of sequences (i.e. distinct states for recurrent models)
  // a value of 1 means strictly single-threaded generation,
  // while higher values (e.g., 4, 8, 16) could allow for parallel processing—at the cost of higher CPU/RAM usage.
  int nSeqMax = 1;

  // Number of threads to use for generation
  // The default in llama.cpp is often set to half the number of CPU cores (e.g., max(cpu_count // 2, 1)), so for 12 cores, that would be 6 threads.
  // But this default is not always optimal. Some users report better performance using fewer threads than cores
  // (e.g., 6 or 8 instead of 12) because hyper-threading and thread management overhead can reduce efficiency.
  int nThreads = 8;

  // Number of threads to use for batch processing
  // n_threads_batch: Threads used for processing batches.
  // Default is to follow n_threads if unset.
  // If n_threads_batch = -1 or unset, it usually defaults to the same number as n_threads.
  // For a 12-core CPU, try values from 6 to 12, matching or near your n_threads.
  int nThreadsBatch = 8;

  // RoPE scaling type
  LlamaRopeScalingType ropeScalingType = LlamaRopeScalingType.unspecified;

  // Pooling type for embeddings
  LlamaPoolingType poolingType = LlamaPoolingType.unspecified;

  // Attention type to use for embeddings
  LlamaAttentionType attentionType = LlamaAttentionType.unspecified;

  // RoPE base frequency, 0 = from model
  double ropeFreqBase = 0;

  // RoPE frequency scaling factor, 0 = from model
  double ropeFreqScale = 0;

  // YaRN extrapolation mix factor, negative = from model
  double yarnExtFactor = -1;

  // YaRN magnitude scaling factor
  double yarnAttnFactor = 1;

  // YaRN low correction dim
  double yarnBetaFast = 32;

  // YaRN high correction dim
  double yarnBetaSlow = 1;

  // YaRN original context size
  int yarnOrigCtx = 0;

  // Defragment the KV cache if holes/size > thold, < 0 disabled
  double defragThold = -1;

  // The llama_decode() call computes all logits, not just the last one
  // bool logitsAll = false;

  // If true, extract embeddings (together with logits)
  bool embeddings = false;

  // Whether to offload the KQV ops (including the KV cache) to GPU
  bool offloadKqv = true;

  // Whether to use flash attention [EXPERIMENTAL]
  bool flashAttn = false;

  // Whether to measure performance timings
  bool noPerfTimings = false;

  // Constructs and returns a `llama_context_params` object
  llama_context_params get() {
    final contextParams = Llama.lib.llama_context_default_params();

    contextParams.n_ctx = nCtx;
    contextParams.n_batch = nBatch;
    contextParams.n_ubatch = nUbatch;
    contextParams.n_seq_max = nSeqMax;
    contextParams.n_threads = nThreads;
    contextParams.n_threads_batch = nThreadsBatch;
    contextParams.rope_scaling_type = ropeScalingType.value;
    contextParams.pooling_type = poolingType.value;
    contextParams.attention_type = attentionType.value;
    contextParams.rope_freq_base = ropeFreqBase;
    contextParams.rope_freq_scale = ropeFreqScale;
    contextParams.yarn_ext_factor = yarnExtFactor;
    contextParams.yarn_attn_factor = yarnAttnFactor;
    contextParams.yarn_beta_fast = yarnBetaFast;
    contextParams.yarn_beta_slow = yarnBetaSlow;
    contextParams.yarn_orig_ctx = yarnOrigCtx;
    contextParams.defrag_thold = defragThold;
    // contextParams.logits_all = logitsAll;
    contextParams.embeddings = embeddings;
    contextParams.offload_kqv = offloadKqv;
    contextParams.flash_attn = flashAttn;
    contextParams.no_perf = noPerfTimings;

    return contextParams;
  }
}
