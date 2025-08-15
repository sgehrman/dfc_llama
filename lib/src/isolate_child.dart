import 'dart:async';

import 'package:dfc_llama/src/additions/llama_extension.dart';
import 'package:dfc_llama/src/chat.dart';
import 'package:dfc_llama/src/context_params.dart';
import 'package:dfc_llama/src/isolate_types.dart';
import 'package:dfc_llama/src/llama.dart';
import 'package:dfc_llama/src/llama_input.dart';
import 'package:dfc_llama/src/model_params.dart';
import 'package:dfc_llama/src/sampler_params.dart';
import 'package:typed_isolate/typed_isolate.dart';

class LlamaChild extends IsolateChild<LlamaResponse, LlamaCommand> {
  LlamaChild(this.systemPrompt) : super(id: 1);

  final String systemPrompt;

  bool shouldStop = false;
  Llama? llama;
  String _template = '';
  bool _firstPrompt = true;

  @override
  void onData(LlamaCommand data) {
    switch (data) {
      case LlamaStop():
        _handleStop();

      case LlamaClear():
        _handleClear();

      case LlamaLoad(
        :final path,
        :final modelParams,
        :final contextParams,
        :final samplingParams,
        :final mmprojPath,
      ):
        _handleLoad(
          path,
          modelParams,
          contextParams,
          samplingParams,
          mmprojPath,
        );

      case LlamaPrompt(:final prompt, :final promptId, :final images):
        _handlePrompt(prompt, promptId, images);

      case LlamaInit(:final libraryPath):
        _handleInit(libraryPath);
    }
  }

  void _handleStop() {
    shouldStop = true;
    sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
  }

  void _handleClear() {
    shouldStop = true;
    if (llama != null) {
      try {
        llama!.clear();
        sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
      } catch (e) {
        sendToParent(LlamaResponse.error('Error clearing context: $e'));
      }
    } else {
      sendToParent(LlamaResponse.error('Cannot clear: model not initialized'));
    }
  }

  void _handleLoad(
    String path,
    ModelParams modelParams,
    ContextParams contextParams,
    SamplerParams samplingParams,
    String? mmprojPath,
  ) {
    try {
      if (mmprojPath != null) {
        llama = Llama(
          path,
          modelParams,
          contextParams,
          samplingParams,
          false,
          mmprojPath,
        );
      } else {
        llama = Llama(path, modelParams, contextParams, samplingParams);
      }

      _template = llama?.chatTemplate() ?? '';

      sendToParent(LlamaResponse.confirmation(LlamaStatus.ready));
    } catch (e) {
      sendToParent(LlamaResponse.error('Error loading model: $e'));
    }
  }

  void _handlePrompt(String prompt, String promptId, List<LlamaImage>? images) {
    shouldStop = false;
    _sendPrompt(prompt, promptId, images);
  }

  void _handleInit(String? libraryPath) {
    Llama.libraryPath = libraryPath;
    sendToParent(LlamaResponse.confirmation(LlamaStatus.uninitialized));
  }

  Future<void> _sendPrompt(
    String prompt,
    String promptId,
    List<LlamaImage>? images,
  ) async {
    if (llama == null) {
      sendToParent(
        LlamaResponse.error('Cannot generate: model not initialized', promptId),
      );
      return;
    }

    try {
      sendToParent(
        LlamaResponse(
          text: '',
          isDone: false,
          status: LlamaStatus.generating,
          promptId: promptId,
        ),
      );

      if (images != null && images.isNotEmpty) {
        final stream = llama!.generateWithMeda(prompt, inputs: images);

        await for (final token in stream) {
          if (shouldStop) {
            break;
          }

          sendToParent(
            LlamaResponse(
              text: token,
              isDone: false,
              status: LlamaStatus.generating,
              promptId: promptId,
            ),
          );
        }

        sendToParent(
          LlamaResponse(
            text: '',
            isDone: true,
            status: LlamaStatus.ready,
            promptId: promptId,
          ),
        );
      } else {
        String? newPrompt = prompt;

        if (_template.isNotEmpty) {
          newPrompt = llama?.applyTemplate(_template, [
            if (_firstPrompt)
              Message(
                role: Role.system,
                content: systemPrompt.isNotEmpty
                    ? systemPrompt
                    : 'You are a helpful, funny assistant. Keep your answers informative but brief.',
              ),
            Message(role: Role.user, content: prompt),
          ]);

          _firstPrompt = false;
        }

        llama!.setPrompt(newPrompt ?? prompt);

        var asyncCount = 0;
        var generationDone = false;
        while (!generationDone && !shouldStop) {
          final (text, isDone) = llama!.getNext();

          sendToParent(
            LlamaResponse(
              text: text,
              isDone: isDone,
              status: isDone ? LlamaStatus.ready : LlamaStatus.generating,
              promptId: promptId,
            ),
          );

          generationDone = isDone;

          // gets us out of the loop so we can process the stop command if it comes in
          asyncCount++;
          if (asyncCount.isEven) {
            await Future.delayed(const Duration(milliseconds: 1));
          }
        }
      }

      if (shouldStop) {
        sendToParent(
          LlamaResponse(
            text: '',
            isDone: true,
            status: LlamaStatus.ready,
            promptId: promptId,
          ),
        );
      }
    } catch (e) {
      sendToParent(LlamaResponse.error('Generation error: $e', promptId));
    }
  }
}
