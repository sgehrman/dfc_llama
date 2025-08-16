import 'dart:async';

import 'package:dfc_llama/src/completion_event.dart';
import 'package:dfc_llama/src/isolate_child.dart';
import 'package:dfc_llama/src/isolate_types.dart';
import 'package:dfc_llama/src/llama.dart';
import 'package:typed_isolate/typed_isolate.dart';

class LlamaParent {
  LlamaParent({
    required this.loadCommand,
    this.systemPrompt = '',
    this.verbose = false,
  });

  final LlamaLoad loadCommand;
  final String systemPrompt;
  final bool verbose;

  StreamController<String> _controller = StreamController<String>.broadcast();
  final _childIsolate = IsolateParent<LlamaCommand, LlamaResponse>();
  StreamSubscription<LlamaResponse>? _subscription;
  bool _isGenerating = false;
  LlamaStatus _status = LlamaStatus.uninitialized;
  LlamaStatus get status => _status;
  Completer<void>? _readyCompleter;
  Completer<void>? _operationCompleter;
  final Map<String, Completer<void>> _promptCompleters = {};
  final _completionController = StreamController<CompletionEvent>.broadcast();
  String _currentPromptId = '';
  final List<_QueuedPrompt> _promptQueue = [];
  bool _isProcessingQueue = false;

  // ------------------------------------
  // getters

  Stream<String> get stream => _controller.stream;
  bool get isGenerating => _isGenerating;
  Stream<CompletionEvent> get completions => _completionController.stream;

  Future<void> init() async {
    _readyCompleter = Completer<void>();

    _isGenerating = false;
    _status = LlamaStatus.uninitialized;
    _childIsolate.init();

    await _subscription?.cancel();
    _subscription = _childIsolate.stream.listen(_childIsolateListener);

    await _childIsolate.spawn(LlamaChild(systemPrompt, verbose: verbose));

    await _sendCommand(
      LlamaInit(
        Llama.libraryPath,
        loadCommand.modelParams,
        loadCommand.contextParams,
        loadCommand.samplingParams,
      ),
      'library initialization',
    );

    _status = LlamaStatus.loading;
    await _sendCommand(loadCommand, 'model loading');

    await _readyCompleter!.future;
  }

  Future<void> reset() async {
    if (_isGenerating) {
      await _stopGeneration();
    }

    if (_controller.isClosed) {
      _controller = StreamController<String>.broadcast();

      await _subscription?.cancel();
      _subscription = _childIsolate.stream.listen(_childIsolateListener);
    }

    await _sendCommand(LlamaClear(), 'context clearing');
  }

  Future<String> sendPrompt(String prompt) {
    if (loadCommand.contextParams.embeddings) {
      throw StateError(
        'This LlamaParent instance is configured for embeddings only and cannot generate text.',
      );
    }

    final queuedPrompt = _QueuedPrompt(prompt);
    _promptQueue.add(queuedPrompt);

    if (!_isProcessingQueue) {
      _processNextPrompt();
    }

    return queuedPrompt.idCompleter.future;
  }

  Future<void> waitForCompletion(String promptId) async {
    if (!_promptCompleters.containsKey(promptId)) {
      return;
    }

    final completer = _promptCompleters[promptId]!;
    await completer.future;
  }

  Future<void> stop() async {
    await _stopGeneration();
  }

  Future<void> dispose() async {
    _isGenerating = false;
    _status = LlamaStatus.disposed;

    await _subscription?.cancel();

    if (!_controller.isClosed) {
      await _controller.close();
    }

    if (!_completionController.isClosed) {
      await _completionController.close();
    }

    for (final completer in _promptCompleters.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('Parent disposed'));
      }
    }
    _promptCompleters.clear();

    for (final queuedPrompt in _promptQueue) {
      if (!queuedPrompt.idCompleter.isCompleted) {
        queuedPrompt.idCompleter.completeError(StateError('Parent disposed'));
      }
    }
    _promptQueue.clear();

    _childIsolate.sendToChild(id: 1, data: LlamaDestroy());

    await Future.delayed(const Duration(milliseconds: 100));

    await _childIsolate.dispose();
  }

  // ------------------------------------------------------------
  // private

  void _processNextPrompt() {
    if (_promptQueue.isEmpty) {
      _isProcessingQueue = false;

      return;
    }

    _isProcessingQueue = true;
    final nextPrompt = _promptQueue.removeAt(0);

    _currentPromptId = DateTime.now().millisecondsSinceEpoch.toString();

    _promptCompleters[_currentPromptId] = Completer<void>();

    nextPrompt.idCompleter.complete(_currentPromptId);

    _isGenerating = true;
    _status = LlamaStatus.generating;

    _childIsolate.sendToChild(
      id: 1,
      data: LlamaPrompt(nextPrompt.prompt, _currentPromptId),
    );

    _promptCompleters[_currentPromptId]!.future
        .then((_) {
          _processNextPrompt();
        })
        .catchError((e) {
          _processNextPrompt();
        });
  }

  Future<void> _stopGeneration() async {
    if (_isGenerating) {
      await _sendCommand(LlamaStop(), 'generation stopping');
      _isGenerating = false;
    }
  }

  Future<void> _sendCommand(LlamaCommand command, String description) {
    _operationCompleter = Completer<void>();

    _childIsolate.sendToChild(data: command, id: 1);

    return _operationCompleter!.future.timeout(
      Duration(seconds: description == 'model loading' ? 60 : 30),
      onTimeout: () {
        throw TimeoutException('Operation "$description" timed out');
      },
    );
  }

  void _childIsolateListener(LlamaResponse data) {
    if (data.status != null) {
      _status = data.status!;

      if (data.status == LlamaStatus.ready &&
          _readyCompleter != null &&
          !_readyCompleter!.isCompleted) {
        _readyCompleter!.complete();
      }
    }

    if (data.text.isNotEmpty) {
      _controller.add(data.text);
    }

    if (data.isConfirmation) {
      if (_operationCompleter != null && !_operationCompleter!.isCompleted) {
        _operationCompleter!.complete();
      }
    }

    if (data.isDone) {
      _isGenerating = false;

      final promptId = data.promptId ?? _currentPromptId;
      if (_promptCompleters.containsKey(promptId)) {
        if (!_promptCompleters[promptId]!.isCompleted) {
          _promptCompleters[promptId]!.complete();
        }
        _promptCompleters.remove(promptId);
      }

      CompletionEvent event;
      if (data.status == LlamaStatus.error) {
        event = CompletionEvent(
          promptId,
          success: false,
          errorDetails: data.errorDetails,
        );
      } else {
        event = CompletionEvent(promptId, success: true);
      }

      _completionController.add(event);
    }
  }
}

// ======================================================================

class _QueuedPrompt {
  _QueuedPrompt(this.prompt);
  final String prompt;
  final Completer<String> idCompleter = Completer<String>();
}
