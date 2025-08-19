import 'package:dfc_llama/src/llama.dart';
import 'package:dfc_llama/src/params/context_params.dart';
import 'package:dfc_llama/src/params/model_params.dart';
import 'package:dfc_llama/src/params/sampler_params.dart';

// Base class for commands sent to the LlamaChild isolate
sealed class LlamaCommand {}

// Command to stop ongoing generation
class LlamaStop extends LlamaCommand {}

// Command to destroy the model context
class LlamaDestroy extends LlamaCommand {}

// Command to initialize the Llama library
class LlamaLoad extends LlamaCommand {
  LlamaLoad({
    required this.path,
    required this.libraryPath,
    required this.modelParams,
    required this.contextParams,
    required this.samplingParams,
  });
  final String path;
  final String libraryPath;
  final ModelParams modelParams;
  final ContextParams contextParams;
  final SamplerParams samplingParams;
}

// Command to send a prompt for generation
class LlamaPrompt extends LlamaCommand {
  LlamaPrompt(this.prompt);
  final String prompt;
}

// Response from the LlamaChild isolate
class LlamaResponse {
  LlamaResponse({
    required this.text,
    required this.isDone,
    this.status,
    this.errorDetails,
  });

  // Create a confirmation response
  factory LlamaResponse.confirmation(LlamaStatus status) {
    return LlamaResponse(text: '', isDone: false, status: status);
  }

  // Create an error response
  factory LlamaResponse.error(String errorMessage) {
    return LlamaResponse(
      text: '',
      isDone: true,
      status: LlamaStatus.error,
      errorDetails: errorMessage,
    );
  }
  final String text;
  final bool isDone;
  final LlamaStatus? status;
  final String? errorDetails;
}
