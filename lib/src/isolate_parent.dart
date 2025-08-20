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

  final StreamController<String> _outputController =
      StreamController<String>.broadcast();
  final _childIsolate = IsolateParent<LlamaCommand, LlamaResponse>();
  StreamSubscription<LlamaResponse>? _subscription;
  final _loadedCompleter = Completer<bool>();
  final _completionController = StreamController<CompletionEvent>.broadcast();

  // ------------------------------------
  // getters

  Stream<String> get stream => _outputController.stream;
  Stream<CompletionEvent> get completions => _completionController.stream;

  Future<void> init() async {
    _childIsolate.init();

    _subscription = _childIsolate.stream.listen(_childIsolateListener);

    await _childIsolate.spawn(LlamaChild(systemPrompt, verbose: verbose));

    _sendCommand(loadCommand);
  }

  void sendPrompt(LlamaPrompt prompt) {
    _sendPrompt(prompt);
  }

  Future<void> _sendPrompt(LlamaPrompt prompt) async {
    final loaded = await _loadedCompleter.future;

    if (loaded) {
      _childIsolate.sendToChild(id: 1, data: prompt);
    } else {
      // should we tell parent, or just assume they already got an error completion?
    }
  }

  void stop() {
    _sendCommand(LlamaStop());
  }

  Future<void> dispose() async {
    await _subscription?.cancel();

    if (!_outputController.isClosed) {
      await _outputController.close();
    }

    if (!_completionController.isClosed) {
      await _completionController.close();
    }

    _childIsolate.sendToChild(id: 1, data: LlamaDestroy());

    await Future.delayed(const Duration(milliseconds: 100));

    await _childIsolate.dispose();
  }

  // ------------------------------------------------------------
  // private

  void _sendCommand(LlamaCommand command) {
    _childIsolate.sendToChild(data: command, id: 1);
  }

  void _childIsolateListener(LlamaResponse data) {
    if (data.status != null) {
      if (!_loadedCompleter.isCompleted) {
        _loadedCompleter.complete(data.status == LlamaStatus.ready);
      }
    }

    if (data.text.isNotEmpty) {
      _outputController.add(data.text);
    }

    if (data.isDone) {
      // if hasn't completed by now, it's an error or something
      if (!_loadedCompleter.isCompleted) {
        _loadedCompleter.complete(false);
      }

      if (data.status == LlamaStatus.error) {
        _completionController.add(
          CompletionEvent(success: false, errorDetails: data.errorDetails),
        );
      } else {
        _completionController.add(CompletionEvent(success: true));
      }
    }
  }
}
