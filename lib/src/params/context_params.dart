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
  int nCtx = 512;

  // The maximum number of tokens to be processed at once at the API/application level.
  // In chat AI, this means how many tokens you send to the model to decode in a single batch.
  // Common defaults for n_batch are 32, 64, 128, or even 512 for larger systems
  // Use moderate values (n_batch=32–128, n_ubatch=8–64) for most chat AI setups, tuning for your hardware and prompt needs.
  // default is n_batch: 2048
  int nBatch = 2048;

  // Divides n_batch into smaller chunks that fit the compute buffer; helps manage memory, especially on GPUs
  // Typical values: Usually set lower than n_batch; 8, 16, 32, or 64 are common, depending on
  // your hardware. n_ubatch must always be less than or equal to n_batch.
  // Setting n_ubatch high can speed things up but uses more device memory; set lower if you hit
  // out-of-memory errors on your hardware.
  // For chat AI on consumer hardware: Try n_batch=64 and n_ubatch=16 or 32. Increase n_batch for longer prompts if your hardware allows.
  // Always ensure: n_ubatch ≤ n_batch.
  int nUbatch = 512;

  // Max number of sequences (i.e. distinct states for recurrent models)
  // a value of 1 means strictly single-threaded generation,
  // while higher values (e.g., 4, 8, 16) could allow for parallel processing—at the cost of higher CPU/RAM usage.
  int nSeqMax = 1;

  // Number of threads to use for generation
  // The default in llama.cpp is often set to half the number of CPU cores (e.g., max(cpu_count // 2, 1)), so for 12 cores, that would be 6 threads.
  // But this default is not always optimal. Some users report better performance using fewer threads than cores
  // (e.g., 6 or 8 instead of 12) because hyper-threading and thread management overhead can reduce efficiency.
  int nThreads = 4;

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

  // If true, extract embeddings (together with logits)
  bool embeddings = false;

  // Whether to offload the KQV ops (including the KV cache) to GPU
  bool offloadKqv = true;

  // Whether to use flash attention [EXPERIMENTAL]
  bool flashAttn = false;

  // Constructs and returns a `llama_context_params` object
  llama_context_params get({bool defaultParams = false}) {
    final contextParams = Llama.lib.llama_context_default_params();

    if (!defaultParams) {
      contextParams.n_ctx = nCtx;
      contextParams.n_batch = nBatch;
      contextParams.n_ubatch = nUbatch;
      contextParams.n_seq_max = nSeqMax;
      contextParams.n_threads = nThreads;
      contextParams.n_threads_batch = nThreads; // matches n_threads
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
      contextParams.embeddings = embeddings;
      contextParams.offload_kqv = offloadKqv;
      contextParams.flash_attn = flashAttn;
      contextParams.no_perf = true; // slows things down
    }

    return contextParams;
  }

  void printParams({bool defaultParams = false}) {
    final params = get(defaultParams: defaultParams);

    print('### llama_context_params');
    print('n_ctx: ${params.n_ctx}');
    print('n_batch: ${params.n_batch}');
    print('n_ubatch: ${params.n_ubatch}');
    print('n_seq_max: ${params.n_seq_max}');
    print('n_threads: ${params.n_threads}');
    print('n_threads_batch: ${params.n_threads_batch}');
    print('rope_scaling_type: ${params.rope_scaling_type}');
    print('pooling_type: ${params.pooling_type}');
    print('attention_type: ${params.attention_type}');
    print('rope_freq_base: ${params.rope_freq_base}');
    print('rope_freq_scale: ${params.rope_freq_scale}');
    print('yarn_ext_factor: ${params.yarn_ext_factor}');
    print('yarn_attn_factor: ${params.yarn_attn_factor}');
    print('yarn_beta_fast: ${params.yarn_beta_fast}');
    print('yarn_beta_slow: ${params.yarn_beta_slow}');
    print('yarn_orig_ctx: ${params.yarn_orig_ctx}');
    print('defrag_thold: ${params.defrag_thold}');
    print('embeddings: ${params.embeddings}');
    print('offload_kqv: ${params.offload_kqv}');
    print('flash_attn: ${params.flash_attn}');
    print('type_k: ${params.type_k}');
    print('type_v: ${params.type_v}');
    print('kv_unified: ${params.kv_unified}');
    print('no_perf: ${params.no_perf}');
    print('op_offload: ${params.op_offload}');
    print('swa_full: ${params.swa_full}');
  }
}

/*
  ## defaults from llama_context_params

  n_ctx: 512
  n_batch: 2048
  n_ubatch: 512
  n_seq_max: 1
  n_threads: 4
  n_threads_batch: 4
  rope_scaling_type: -1
  pooling_type: -1
  attention_type: -1
  rope_freq_base: 0.0
  rope_freq_scale: 0.0
  yarn_ext_factor: -1.0
  yarn_attn_factor: 1.0
  yarn_beta_fast: 32.0
  yarn_beta_slow: 1.0
  yarn_orig_ctx: 0
  defrag_thold: -1.0
  embeddings: false
  offload_kqv: true
  flash_attn: false
  type_k: 1
  type_v: 1
  kv_unified: false
  no_perf: true
  op_offload: true
  swa_full: true
*/
