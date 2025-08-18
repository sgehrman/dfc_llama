import 'dart:ffi';

import 'package:dfc_llama/src/llama_cpp.dart';

class SamplerParams {
  SamplerParams();

  // see common_params_sampling in common/common.h

  // The sampling chain can be very simple (greedy) or more
  // complex (top-k, top-p, etc).
  bool greedy = false;
  int seed = LLAMA_DEFAULT_SEED; // For distribution sampler
  // bool softmax = true;

  // Top-K sampling
  // @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  int topK = 40;

  // Top-P (nucleus) sampling
  // @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  double topP = 0.95;
  int topPKeep = 1;

  // Min-P sampling
  // @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
  double minP = 0.05;
  int minPKeep = 1;

  // Typical sampling
  // @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666
  double typical = 1;
  int typicalKeep = 1;

  // Temperature
  // @details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
  double temp = 0.80;

  // XTC sampling
  // @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
  double xtcTemperature = 1;
  double xtcStartValue = 0.1;
  int xtcKeep = 1;
  int xtcLength = 1;

  // Mirostat 1.0
  // @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  // @param tau The target cross-entropy (or surprise) value you want to achieve for the generated text
  // @param eta The learning rate used to update `mu` based on the error between target and observed surprisal
  // @param m The number of tokens considered in the estimation of `s_hat`
  double mirostatTau = 5;
  double mirostatEta = 0.10;
  int mirostatM = 100;

  // Mirostat 2.0
  // @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966
  double mirostat2Tau = 5;
  double mirostat2Eta = 0.10;

  // Penalties
  // @details Token penalties configuration
  int penaltyLastTokens =
      64; // last n tokens to penalize (0 = disable penalty, -1 = context size)
  double penaltyRepeat = 1; // 1.0 = disabled
  double penaltyFreq = 0; // 0.0 = disabled
  double penaltyPresent = 0; // 0.0 = disabled

  static Pointer<llama_sampler> setSamplerParams(
    llama_cpp lib,
    SamplerParams samplerParams,
    Pointer<llama_vocab> vocab,
  ) {
    Pointer<llama_sampler> smpl;

    final sparams = lib.llama_sampler_chain_default_params();
    sparams.no_perf = true;

    smpl = lib.llama_sampler_chain_init(sparams);

    if (samplerParams.greedy) {
      lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_greedy());
    }

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_dist(samplerParams.seed),
    );

    // if (samplerParams.softmax) {
    //   lib.llama_sampler_chain_add(smpl, lib.llama_sampler_init_softmax());
    // }

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_top_k(samplerParams.topK),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_top_p(samplerParams.topP, samplerParams.topPKeep),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_min_p(samplerParams.minP, samplerParams.minPKeep),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_typical(
        samplerParams.typical,
        samplerParams.typicalKeep,
      ),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_temp(samplerParams.temp),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_xtc(
        samplerParams.xtcTemperature,
        samplerParams.xtcStartValue,
        samplerParams.xtcKeep,
        samplerParams.xtcLength,
      ),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_mirostat(
        lib.llama_n_vocab(vocab),
        samplerParams.seed,
        samplerParams.mirostatTau,
        samplerParams.mirostatEta,
        samplerParams.mirostatM,
      ),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_mirostat_v2(
        samplerParams.seed,
        samplerParams.mirostat2Tau,
        samplerParams.mirostat2Eta,
      ),
    );

    lib.llama_sampler_chain_add(
      smpl,
      lib.llama_sampler_init_penalties(
        samplerParams.penaltyLastTokens,
        samplerParams.penaltyRepeat,
        samplerParams.penaltyFreq,
        samplerParams.penaltyPresent,
      ),
    );

    return smpl;
  }
}
