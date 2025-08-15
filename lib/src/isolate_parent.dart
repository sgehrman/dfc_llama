import 'dart:async';

import 'package:dfc_llama/dfc_llama.dart';
import 'package:typed_isolate/typed_isolate.dart';

class CompletionEvent {
  final String promptId;
  final bool success;
  final String? errorDetails;

  CompletionEvent(this.promptId, this.success, [this.errorDetails]);
}

// ======================================================================

class _QueuedPrompt {
  final String prompt;
  final Completer<String> idCompleter = Completer<String>();
  final Object? scope;
  final List<LlamaImage>? images;

  _QueuedPrompt(this.prompt, this.scope, {this.images});
}

// ======================================================================

class _QueuedPromptWithImages extends _QueuedPrompt {
  _QueuedPromptWithImages(super.prompt, List<LlamaImage> images, super.scope)
    : super(images: images);
}

// ======================================================================

class LlamaParent {
  LlamaParent({required this.loadCommand, this.systemPrompt = ''});

  final String systemPrompt;
  StreamController<String> _controller = StreamController<String>.broadcast();
  final _parent = IsolateParent<LlamaCommand, LlamaResponse>();

  StreamSubscription<LlamaResponse>? _subscription;
  bool _isGenerating = false;

  LlamaStatus _status = LlamaStatus.uninitialized;
  LlamaStatus get status => _status;

  final LlamaLoad loadCommand;

  Completer<void>? _readyCompleter;
  Completer<void>? _operationCompleter;

  final Map<String, Completer<void>> _promptCompleters = {};

  Stream<String> get stream => _controller.stream;

  bool get isGenerating => _isGenerating;

  final _completionController = StreamController<CompletionEvent>.broadcast();
  Stream<CompletionEvent> get completions => _completionController.stream;

  String _currentPromptId = "";

  final List<dynamic> _scopes = [];

  final List<_QueuedPrompt> _promptQueue = [];
  bool _isProcessingQueue = false;

  void _onData(LlamaResponse data) {
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

      for (final scope in _scopes) {
        scope.handleResponse(data);
      }
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
        event = CompletionEvent(promptId, false, data.errorDetails);
      } else {
        event = CompletionEvent(promptId, true);
      }

      _completionController.add(event);

      for (final scope in _scopes) {
        scope.handleCompletion(event);
      }
    }
  }

  Future<void> init() async {
    _readyCompleter = Completer<void>();

    _isGenerating = false;
    _status = LlamaStatus.uninitialized;
    _parent.init();

    await _subscription?.cancel();
    _subscription = _parent.stream.listen(_onData);

    await _parent.spawn(LlamaChild(systemPrompt));

    await _sendCommand(
      LlamaInit(
        Llama.libraryPath,
        loadCommand.modelParams,
        loadCommand.contextParams,
        loadCommand.samplingParams,
      ),
      "library initialization",
    );

    _status = LlamaStatus.loading;
    await _sendCommand(loadCommand, "model loading");

    await _readyCompleter!.future;
  }

  void reset() async {
    await _reset();
  }

  Future<void> _sendCommand(LlamaCommand command, String description) async {
    _operationCompleter = Completer<void>();

    _parent.sendToChild(data: command, id: 1);

    return await _operationCompleter!.future.timeout(
      Duration(seconds: description == "model loading" ? 60 : 30),
      onTimeout: () {
        throw TimeoutException('Operation "$description" timed out');
      },
    );
  }

  Future<void> _reset() async {
    if (_isGenerating) {
      await _stopGeneration();
    }

    if (_controller.isClosed) {
      _controller = StreamController<String>.broadcast();

      await _subscription?.cancel();
      _subscription = _parent.stream.listen(_onData);
    }

    await _sendCommand(LlamaClear(), "context clearing");
  }

  Future<void> _stopGeneration() async {
    if (_isGenerating) {
      await _sendCommand(LlamaStop(), "generation stopping");
      _isGenerating = false;
    }
  }

  Future<String> sendPrompt(String prompt, {Object? scope}) async {
    if (loadCommand.contextParams.embeddings) {
      throw StateError(
        "This LlamaParent instance is configured for embeddings only and cannot generate text.",
      );
    }

    final queuedPrompt = _QueuedPrompt(prompt, scope);
    _promptQueue.add(queuedPrompt);

    if (!_isProcessingQueue) {
      _processNextPrompt();
    }

    return queuedPrompt.idCompleter.future;
  }

  Future<void> _processNextPrompt() async {
    if (_promptQueue.isEmpty) {
      _isProcessingQueue = false;
      return;
    }

    _isProcessingQueue = true;
    final nextPrompt = _promptQueue.removeAt(0);

    _currentPromptId = DateTime.now().millisecondsSinceEpoch.toString();

    _promptCompleters[_currentPromptId] = Completer<void>();

    nextPrompt.idCompleter.complete(_currentPromptId);

    if (nextPrompt.scope != null) {
      (nextPrompt.scope as dynamic).addPromptId(_currentPromptId);
    }

    _isGenerating = true;
    _status = LlamaStatus.generating;

    _parent.sendToChild(
      id: 1,
      data: LlamaPrompt(
        nextPrompt.prompt,
        _currentPromptId,
        images: nextPrompt.images,
      ),
    );

    _promptCompleters[_currentPromptId]!.future
        .then((_) {
          _processNextPrompt();
        })
        .catchError((e) {
          _processNextPrompt();
        });
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

  dynamic getScope() {
    final scope = LlamaScope(this);
    _scopes.add(scope);
    return scope;
  }

  Future<void> dispose() async {
    _isGenerating = false;
    _status = LlamaStatus.disposed;

    await _subscription?.cancel();

    final scopesToDispose = List.from(_scopes);
    for (final scope in scopesToDispose) {
      await scope.dispose();
    }
    _scopes.clear();

    if (!_controller.isClosed) {
      await _controller.close();
    }

    if (!_completionController.isClosed) {
      await _completionController.close();
    }

    for (final completer in _promptCompleters.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError("Parent disposed"));
      }
    }
    _promptCompleters.clear();

    for (final queuedPrompt in _promptQueue) {
      if (!queuedPrompt.idCompleter.isCompleted) {
        queuedPrompt.idCompleter.completeError(StateError("Parent disposed"));
      }
    }
    _promptQueue.clear();

    _parent.sendToChild(id: 1, data: LlamaClear());

    await Future.delayed(const Duration(milliseconds: 100));

    _parent.dispose();
  }

  Future<String> sendPromptWithImages(
    String prompt,
    List<LlamaImage> images, {
    Object? scope,
  }) async {
    if (loadCommand.contextParams.embeddings) {
      throw StateError(
        "This LlamaParent instance is configured for embeddings only and cannot generate text.",
      );
    }

    final queuedPrompt = _QueuedPromptWithImages(prompt, images, scope);
    _promptQueue.add(queuedPrompt);

    if (!_isProcessingQueue) {
      _processNextPrompt();
    }

    return queuedPrompt.idCompleter.future;
  }
}
