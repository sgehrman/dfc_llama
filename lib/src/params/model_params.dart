import 'dart:ffi';

import 'package:dfc_llama/src/llama.dart';
import 'package:dfc_llama/src/llama_cpp.dart';
import 'package:ffi/ffi.dart';

// Enum representing how to split the model across multiple GPUs
enum LlamaSplitMode {
  none, // single GPU
  layer, // split layers and KV across GPUs
  row, // split layers and KV across GPUs, use tensor parallelism if supported
}

class ModelParams {
  ModelParams();

  // Number of layers to store in VRAM
  int nGpuLayers = 99;

  // How to split the model across multiple GPUs
  LlamaSplitMode splitMode = LlamaSplitMode.none;

  // The GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
  int mainGpu = 0;

  // Proportion of the model (layers or rows) to offload to each GPU
  List<double> tensorSplit = [];

  // Comma separated list of RPC servers to use for offloading
  String rpcServers = '';

  // Override key-value pairs of the model metadata
  Map<String, dynamic> kvOverrides = {};

  // Only load the vocabulary, no weights
  bool vocabOnly = false;

  // Use mmap if possible
  bool useMemorymap = true;

  // Force system to keep model in RAM
  bool useMemoryLock = false;

  // Validate model tensor data
  bool checkTensors = false;

  // Pointers that need to be freed
  Pointer<Float>? _tensorSplitPtr;
  Pointer<Char>? _rpcServersPtr;

  // Constructs and returns a `llama_model_params` object with current settings
  llama_model_params get() {
    final modelParams = Llama.lib.llama_model_default_params();

    // Basic parameters
    modelParams.n_gpu_layers = nGpuLayers;
    // modelParams.split_mode = splitMode.index; // @TODO split_mode setter
    modelParams.main_gpu = mainGpu;
    modelParams.vocab_only = vocabOnly;
    modelParams.use_mmap = useMemorymap;
    modelParams.use_mlock = useMemoryLock;
    modelParams.check_tensors = checkTensors;

    // Handle tensor_split
    if (tensorSplit.isNotEmpty) {
      _tensorSplitPtr = malloc<Float>(tensorSplit.length);
      for (var i = 0; i < tensorSplit.length; i++) {
        _tensorSplitPtr![i] = tensorSplit[i];
      }
      modelParams.tensor_split = _tensorSplitPtr!;
    }

    // Complex pointers set to null
    modelParams.progress_callback = nullptr;
    modelParams.progress_callback_user_data = nullptr;
    modelParams.kv_overrides = nullptr;

    return modelParams;
  }

  // Free allocated memory
  void dispose() {
    if (_tensorSplitPtr != null) {
      malloc.free(_tensorSplitPtr!);
      _tensorSplitPtr = null;
    }
    if (_rpcServersPtr != null) {
      malloc.free(_rpcServersPtr!);
      _rpcServersPtr = null;
    }
  }
}
