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
  // default is 0 since some computer might not have a gpu
  int nGpuLayers = 0;

  // How to split the model across multiple GPUs
  LlamaSplitMode splitMode = LlamaSplitMode.none;

  // The GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
  int mainGpu = 0;

  // Proportion of the model (layers or rows) to offload to each GPU
  List<double> tensorSplit = [];

  // Use mmap if possible
  bool useMemorymap = true;

  // Force system to keep model in RAM
  bool useMemoryLock = false;

  // Validate model tensor data
  bool checkTensors = false;

  // Pointers that need to be freed
  Pointer<Float>? _tensorSplitPtr;

  // Free allocated memory
  void dispose() {
    if (_tensorSplitPtr != null) {
      malloc.free(_tensorSplitPtr!);
      _tensorSplitPtr = null;
    }
  }

  // Constructs and returns a `llama_model_params` object with current settings
  llama_model_params get({bool defaultParams = false}) {
    final modelParams = Llama.lib.llama_model_default_params();

    if (!defaultParams) {
      modelParams.n_gpu_layers = nGpuLayers;
      modelParams.main_gpu = mainGpu;
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
    }

    return modelParams;
  }

  void printParams({bool defaultParams = false}) {
    final params = get(defaultParams: defaultParams);

    print('### llama_model_params');
    print('n_gpu_layers: ${params.n_gpu_layers}');
    print('split_mode: ${params.split_mode}');
    print('main_gpu: ${params.main_gpu}');
    // print('tensor_split: ${duh.tensor_split}');
    // print('progress_callback: ${duh.progress_callback}');
    // print('progress_callback_user_data: ${duh.progress_callback_user_data}');
    // print('kv_overrides: ${duh.kv_overrides}');
    print('vocab_only: ${params.vocab_only}');
    print('use_mmap: ${params.use_mmap}');
    print('use_mlock: ${params.use_mlock}');
    print('check_tensors: ${params.check_tensors}');
    print('use_extra_bufts: ${params.use_extra_bufts}');
  }
}
