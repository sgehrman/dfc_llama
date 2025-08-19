import 'dart:async';

import 'package:dfc_llama/src/additions/llama_extension.dart';
import 'package:dfc_llama/src/chat.dart';
import 'package:dfc_llama/src/isolate_types.dart';
import 'package:dfc_llama/src/llama.dart';
import 'package:dfc_llama/src/params/context_params.dart';
import 'package:dfc_llama/src/params/model_params.dart';
import 'package:dfc_llama/src/params/sampler_params.dart';
import 'package:typed_isolate/typed_isolate.dart';

class LlamaChild extends IsolateChild<LlamaResponse, LlamaCommand> {
  LlamaChild(this.systemPrompt, {required this.verbose}) : super(id: 1);

  final String systemPrompt;
  final bool verbose;
  bool _shouldStop = false;
  Llama? llama;
  String _template = '';
  bool _firstPrompt = true;

  @override
  void onData(LlamaCommand data) {
    switch (data) {
      case LlamaStop():
        _handleStop();

      case LlamaDestroy():
        _handleDestroy();

      case LlamaLoad(
        :final path,
        :final libraryPath,
        :final modelParams,
        :final contextParams,
        :final samplingParams,
      ):
        _handleLoad(
          path: path,
          libraryPath: libraryPath,
          modelParams: modelParams,
          contextParams: contextParams,
          samplingParams: samplingParams,
        );

      case LlamaPrompt(:final prompt):
        _handlePrompt(prompt);
    }
  }

  void _handleStop() {
    _shouldStop = true;
  }

  void _handleDestroy() {
    _shouldStop = true;
    if (llama != null) {
      try {
        // I not seeing GPU memory freed, trying this but not sure if it helps
        llama?.dispose();
        llama = null;
      } catch (e) {
        print(e);
      }
    }
  }

  void _handleLoad({
    required String path,
    required String libraryPath,
    required ModelParams modelParams,
    required ContextParams contextParams,
    required SamplerParams samplingParams,
  }) {
    try {
      llama = Llama(
        modelPath: path,
        libraryPath: libraryPath,
        modelParamsDart: modelParams,
        contextParamsDart: contextParams,
        samplerParams: samplingParams,
        verbose: verbose,
      );

      _template = llama?.chatTemplate() ?? '';

      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      sendToParent(LlamaResponse.error('Error loading model: $e'));
    }
  }

  void _handlePrompt(String prompt) {
    _shouldStop = false;
    _sendPrompt(prompt);
  }

  Future<void> _sendPrompt(String prompt) async {
    if (llama == null) {
      sendToParent(
        LlamaResponse.error('Cannot generate: model not initialized'),
      );
      return;
    }

    try {
      sendToParent(
        LlamaResponse(text: '', isDone: false, status: LlamaStatus.generating),
      );

      String? newPrompt = prompt;

      if (_template.isNotEmpty) {
        newPrompt = llama?.applyTemplate(_template, [
          if (_firstPrompt && systemPrompt.isNotEmpty)
            Message(role: Role.system, content: systemPrompt),
          Message(role: Role.user, content: prompt),
        ]);

        _firstPrompt = false;
      }

      llama?.setPrompt(newPrompt ?? prompt);

      var asyncCount = 0;
      var generationDone = false;
      while (!generationDone && !_shouldStop) {
        final (text, isDone) = llama?.getNext() ?? ('', true);

        sendToParent(
          LlamaResponse(
            text: text,
            isDone: isDone,
            status: isDone ? LlamaStatus.ready : LlamaStatus.generating,
          ),
        );

        generationDone = isDone;

        // gets us out of the loop so we can process the stop command if it comes in
        asyncCount++;
        if (asyncCount.isEven) {
          await Future.delayed(const Duration(milliseconds: 1));
        }
      }

      if (_shouldStop) {
        sendToParent(
          LlamaResponse(text: '', isDone: true, status: LlamaStatus.ready),
        );
      }
    } catch (e) {
      sendToParent(LlamaResponse.error('Generation error: $e'));
    }
  }
}
