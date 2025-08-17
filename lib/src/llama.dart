import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' show sqrt;
import 'dart:typed_data';

import 'package:dfc_llama/src/llama_cpp.dart';
import 'package:dfc_llama/src/params/context_params.dart';
import 'package:dfc_llama/src/params/model_params.dart';
import 'package:dfc_llama/src/params/sampler_params.dart';
import 'package:ffi/ffi.dart';

typedef LlamaLogCallback =
    Void Function(
      UnsignedInt level,
      Pointer<Char> text,
      Pointer<Void> userData,
    );
typedef LlamaLogCallbackDart =
    void Function(int level, Pointer<Char> text, Pointer<Void> userData);

class LlamaException implements Exception {
  LlamaException(this.message, [this.originalError]);
  final String message;
  final dynamic originalError;

  @override
  String toString() =>
      'LlamaException: $message${originalError != null ? ' ($originalError)' : ''}';
}

enum LlamaStatus { uninitialized, loading, ready, generating, error, disposed }

// Throws [LlamaException] if model loading or initialization fails.
class Llama {
  Llama({
    required this.modelPath,
    required this.modelParamsDart,
    required this.contextParamsDart,
    required this.samplerParams,
    this.verbose = false,
  }) {
    try {
      _initializeLlama(modelPath, modelParamsDart, samplerParams);

      // Always initialize the batch, even if contextParamsDart is null
      final contextParams = contextParamsDart.get();
      batch = lib.llama_batch_init(contextParams.n_batch, 0, 1);

      _isInitialized = true; // Mark initialization as complete
      _status = LlamaStatus.ready;
    } catch (e) {
      _status = LlamaStatus.error;
      dispose(); // Dispose resources on initialization failure
      throw LlamaException('Failed to initialize Llama', e);
    }
  }

  final String modelPath;
  final ModelParams modelParamsDart;
  final ContextParams contextParamsDart;
  final SamplerParams samplerParams;
  final bool verbose;

  static llama_cpp? _lib;
  late Pointer<llama_model> model;
  late Pointer<llama_context> context;
  late Pointer<llama_vocab> vocab;
  late llama_batch batch;

  Pointer<llama_sampler> _smpl = nullptr;
  Pointer<llama_token> _tokens = nullptr;
  Pointer<llama_token> _tokenPtr = nullptr;
  int _nPrompt = 0;
  int _nPos = 0;

  bool _isInitialized = false; // Track if initialization is complete

  bool _isDisposed = false;
  LlamaStatus _status = LlamaStatus.uninitialized;

  static String? libraryPath = Platform.isAndroid ? 'libllama.so' : null;

  // Gets the current status of the Llama instance
  LlamaStatus get status => _status;

  // Checks if the instance has been disposed
  bool get isDisposed => _isDisposed;

  static llama_cpp get lib {
    if (_lib == null) {
      if (libraryPath != null) {
        _lib = llama_cpp(DynamicLibrary.open(libraryPath!));
      } else {
        _lib = llama_cpp(DynamicLibrary.process());
      }
    }
    return _lib!;
  }

  llama_cpp getLib() {
    return _lib!;
  }

  static void llamaLogCallbackNull(
    int level,
    Pointer<Char> text,
    Pointer<Void> userData,
  ) {}

  // Initializes the Llama instance with the given parameters
  void _initializeLlama(
    String modelPath,
    ModelParams modelParamsDart,
    SamplerParams samplerParams,
  ) {
    if (verbose == false) {
      final nullCallbackPointer = Pointer.fromFunction<LlamaLogCallback>(
        Llama.llamaLogCallbackNull,
      );
      lib.llama_log_set(nullCallbackPointer, nullptr);
    }

    lib.llama_backend_init();

    final modelParams = modelParamsDart.get();

    final modelPathPtr = modelPath.toNativeUtf8().cast<Char>();
    Pointer<llama_model> loadedModel = nullptr; // Use a local variable
    try {
      loadedModel = lib.llama_load_model_from_file(modelPathPtr, modelParams);
      if (loadedModel == nullptr) {
        throw LlamaException('Could not load model at $modelPath');
      }
      model = loadedModel; // Assign to the class member after the check
      vocab = lib.llama_model_get_vocab(
        model,
      ); // Get the vocab *after* model is loaded.
    } finally {
      malloc.free(modelPathPtr);
    }

    final contextParams = contextParamsDart.get();
    Pointer<llama_context> loadedContext = nullptr;
    try {
      loadedContext = lib.llama_new_context_with_model(model, contextParams);

      if (loadedContext == nullptr) {
        lib.llama_model_free(model);
        throw LlamaException('Could not create context!');
      }
      context = loadedContext;
    } catch (e) {
      if (loadedContext != nullptr) {
        lib.llama_free(loadedContext);
      }
      lib.llama_model_free(model);
      rethrow;
    }

    _smpl = SamplerParams.setSamplerParams(lib, samplerParams, vocab);

    _tokenPtr = malloc<llama_token>();
  }

  // Sets the prompt for text generation.
  //
  // [prompt] - The input prompt text
  // [onProgress] - Optional callback for tracking tokenization progress
  //
  // Throws [ArgumentError] if prompt is empty
  // Throws [LlamaException] if tokenization fails
  void setPrompt(
    String prompt, {
    void Function(int current, int total)? onProgress,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError('Prompt cannot be empty');
    }
    if (_isDisposed) {
      throw StateError('Cannot set prompt on disposed instance');
    }

    Pointer<Utf8>? promptUtf8Ptr;

    try {
      _status = LlamaStatus.generating;

      if (_nPos == 0) {
        if (context.address != 0) {
          final mem = lib.llama_get_memory(context);
          lib.llama_memory_clear(mem, true);
        }
        batch.n_tokens = 0;
      }

      promptUtf8Ptr = prompt.toNativeUtf8();
      final promptBytes = promptUtf8Ptr.length;
      final promptCharPtr = promptUtf8Ptr.cast<Char>();

      _nPrompt = -lib.llama_tokenize(
        vocab,
        promptCharPtr,
        promptBytes,
        nullptr,
        0,
        true,
        true,
      );

      if (_nPrompt <= 0) {
        throw LlamaException(
          'Failed to estimate token count (returned $_nPrompt)',
        );
      }

      if (_tokens != nullptr) {
        malloc.free(_tokens);
      }
      _tokens = malloc<llama_token>(_nPrompt);
      final actualTokens = lib.llama_tokenize(
        vocab,
        promptCharPtr,
        promptBytes,
        _tokens,
        _nPrompt,
        true,
        true,
      );

      if (actualTokens < 0) {
        malloc.free(_tokens);
        _tokens = nullptr;
        throw LlamaException(
          'Failed to tokenize prompt (returned $actualTokens)',
        );
      }
      _nPrompt = actualTokens;

      final batchCapacity = contextParamsDart.nBatch;
      if (_nPrompt > batchCapacity) {
        malloc.free(_tokens);
        _tokens = nullptr;
        throw LlamaException(
          'Prompt token count ($_nPrompt) exceeds batch capacity ($batchCapacity)',
        );
      }

      for (var i = 0; i < _nPrompt; i++) {
        batch.token[i] = _tokens[i];
        batch.pos[i] = _nPos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = calloc<llama_seq_id>()..value = 0;
        batch.logits[i] = i == _nPrompt - 1 ? 1 : 0;
      }
      batch.n_tokens = _nPrompt;
    } catch (e) {
      _status = LlamaStatus.error;
      if (_tokens != nullptr) {
        malloc.free(_tokens);
        _tokens = nullptr;
      }
      rethrow;
    } finally {
      if (promptUtf8Ptr != null) {
        malloc.free(promptUtf8Ptr);
      }
    }
  }

  // Generates the next token in the sequence.
  //
  // Returns a tuple containing the generated text and a boolean indicating if generation is complete.
  // Throws [LlamaException] if generation fails.
  (String, bool) getNext() {
    if (_isDisposed) {
      throw StateError('Cannot generate text on disposed instance');
    }

    try {
      if (lib.llama_decode(context, batch) != 0) {
        throw LlamaException('Failed to eval');
      }
      _nPos += batch.n_tokens;
      final newTokenId = lib.llama_sampler_sample(_smpl, context, -1);

      if (lib.llama_token_is_eog(vocab, newTokenId)) {
        return ('', true);
      }

      const bufSize = 256;
      final buf = malloc<Char>(bufSize);
      try {
        final n = lib.llama_token_to_piece(
          vocab,
          newTokenId,
          buf,
          bufSize,
          0,
          true,
        );
        if (n < 0) {
          throw LlamaException('Failed to convert token to piece');
        }
        // String piece = utf8.decode(buf.cast<Uint8>().asTypedList(n));
        var piece = '';
        final bytes = buf.cast<Uint8>().asTypedList(n);
        try {
          piece = utf8.decode(bytes);
        } catch (e) {
          piece = utf8.decode(bytes, allowMalformed: true);
        }

        batch.token[0] = newTokenId;
        batch.pos[0] = _nPos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0].value = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        final isEos = newTokenId == lib.llama_token_eos(vocab);
        return (piece, isEos);
      } finally {
        malloc.free(buf);
      }
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error generating text', e);
    }
  }

  // Provides a stream of generated text tokens
  Stream<String> generateText() async* {
    if (_isDisposed) {
      throw StateError('Cannot generate text on disposed instance');
    }

    try {
      while (true) {
        final (text, isDone) = getNext();
        if (isDone) {
          break;
        }
        yield text;
      }
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error in text generation stream', e);
    }
  }

  // Disposes of all resources held by this instance
  void dispose() {
    if (_isDisposed) {
      return;
    }
    if (_tokens != nullptr) {
      malloc.free(_tokens);
    }
    if (_tokenPtr != nullptr) {
      malloc.free(_tokenPtr);
    }
    if (_smpl != nullptr) {
      lib.llama_sampler_free(_smpl);
    }

    // Only access late fields if initialization was completed
    if (_isInitialized) {
      if (context.address != 0) {
        lib.llama_free(context);
      }
      if (model.address != 0) {
        lib.llama_model_free(model);
      }

      // Free the batch only if it was initialized
      try {
        lib.llama_batch_free(batch);
      } catch (e) {
        // Batch not initialized, ignore
      }
    }

    // if (_mctx != nullptr) lib.mtmd_free(_mctx); // <- crash - double free

    lib.llama_backend_free();

    _isDisposed = true;
    _status = LlamaStatus.disposed;
  }

  // Clears the current state of the Llama instance
  // This allows reusing the same instance for a new generation
  // without creating a new instance
  void clear() {
    if (_isDisposed) {
      throw StateError('Cannot clear disposed instance');
    }

    try {
      if (_tokens != nullptr) {
        malloc.free(_tokens);
        _tokens = nullptr;
      }
      _nPrompt = 0;
      _nPos = 0;

      if (_isInitialized && context.address != 0) {
        final mem = lib.llama_get_memory(context);
        lib.llama_memory_clear(mem, true);
      }

      if (batch.seq_id != nullptr) {
        final batchCapacity = contextParamsDart.nBatch;
        if (batchCapacity > 0) {
          for (var i = 0; i < batchCapacity; ++i) {
            if (batch.seq_id[i] != nullptr) {
              calloc.free(batch.seq_id[i]);
              batch.seq_id[i] = nullptr; // Set to null as batch is reused
            }
          }
        }
      }
      batch.n_tokens = 0;

      _status = LlamaStatus.ready;
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Failed to clear Llama state', e);
    }
  }

  // Converts a text string to a list of token IDs
  //
  // [text] - The input text to tokenize
  // [addBos] - Whether to add the beginning-of-sequence token
  //
  // Returns a List of integer token IDs
  //
  // Throws [ArgumentError] if text is empty
  // Throws [LlamaException] if tokenization fails
  List<int> tokenize({required String text, required bool addBos}) {
    if (_isDisposed) {
      throw StateError('Cannot tokenize with disposed instance');
    }

    if (text.isEmpty) {
      throw ArgumentError('Text cannot be empty');
    }

    try {
      final textPtr = text.toNativeUtf8().cast<Char>();

      try {
        final nTokens = -lib.llama_tokenize(
          vocab,
          textPtr,
          text.length,
          nullptr,
          0,
          addBos,
          true,
        );

        if (nTokens <= 0) {
          throw LlamaException('Failed to determine token count');
        }

        final tokens = malloc<llama_token>(nTokens);

        try {
          final actualTokens = lib.llama_tokenize(
            vocab,
            textPtr,
            text.length,
            tokens,
            nTokens,
            addBos,
            true,
          );

          if (actualTokens < 0) {
            throw LlamaException('Tokenization failed');
          }
          return List<int>.generate(actualTokens, (i) => tokens[i]);
        } finally {
          malloc.free(tokens);
        }
      } finally {
        malloc.free(textPtr);
      }
    } catch (e) {
      throw LlamaException('Error during tokenization', e);
    }
  }

  // Generates embeddings for the given prompt.
  //
  // [prompt] - The input text for which to generate embeddings.
  // [addBos] - Whether to add the beginning-of-sequence token.
  // [normalize] - Whether to normalize the embeddings (default: true).
  //
  // Returns a List of floats representing the embedding.
  List<double> getEmbeddings(
    String prompt, {
    bool addBos = true,
    bool normalize = true,
  }) {
    if (_isDisposed) {
      throw StateError('Cannot generate embeddings on disposed instance');
    }

    if (prompt.isEmpty) {
      throw ArgumentError('Prompt cannot be empty');
    }

    llama_batch? promptBatch;

    try {
      // Tokenize the input text
      var tokens = tokenize(text: prompt, addBos: addBos);
      var nTokens = tokens.length;

      // Check if token count exceeds batch size
      final batchSize = contextParamsDart.nBatch;
      if (nTokens > batchSize) {
        tokens = tokens.sublist(0, batchSize - 1);
        nTokens = tokens.length;
      }

      // Create a batch for the tokens
      promptBatch = lib.llama_batch_init(nTokens, 0, 1);

      // Setup the batch with the tokens
      for (var i = 0; i < nTokens; i++) {
        promptBatch.token[i] = tokens[i];
        promptBatch.pos[i] = i; // Use position within sequence
        promptBatch.n_seq_id[i] = 1;
        promptBatch.seq_id[i] = calloc<llama_seq_id>()..value = 0;
        promptBatch.logits[i] = i == nTokens - 1
            ? 1
            : 0; // Set logits flag for last token
      }
      promptBatch.n_tokens = nTokens;

      // Clear the KV cache
      // lib.llama_kv_cache_clear(context);
      final mem = lib.llama_get_memory(context);
      lib.llama_memory_clear(mem, true);

      // Process the batch
      var isEncoderOnly = false;
      isEncoderOnly =
          lib.llama_model_has_encoder(model) &&
          !lib.llama_model_has_decoder(model);

      if (isEncoderOnly) {
        if (lib.llama_encode(context, promptBatch) != 0) {
          throw LlamaException('Failed to encode prompt for embeddings');
        }
      } else {
        if (lib.llama_decode(context, promptBatch) != 0) {
          throw LlamaException('Failed to decode prompt for embeddings');
        }
      }

      // Get the embeddings
      final nEmbd = lib.llama_n_embd(model);
      Pointer<Float> embeddingsPtr;

      try {
        // First try sequence embeddings
        embeddingsPtr = lib.llama_get_embeddings_seq(context, 0);
      } catch (e) {
        try {
          // Then try last token embeddings
          embeddingsPtr = lib.llama_get_embeddings_ith(context, nTokens - 1);
        } catch (e) {
          // Finally fall back to default embeddings
          embeddingsPtr = lib.llama_get_embeddings(context);
        }
      }

      if (embeddingsPtr == nullptr) {
        throw LlamaException('Failed to get embeddings');
      }

      // Convert to Dart list
      final embeddings = List<double>.filled(nEmbd, 0);
      for (var i = 0; i < nEmbd; i++) {
        embeddings[i] = embeddingsPtr[i];
      }

      // Normalize if requested
      if (normalize) {
        var sum = 0.0;
        for (var i = 0; i < nEmbd; i++) {
          sum += embeddings[i] * embeddings[i];
        }
        final norm = sqrt(sum);
        if (norm > 0) {
          for (var i = 0; i < nEmbd; i++) {
            embeddings[i] = embeddings[i] / norm;
          }
        }
      }

      return embeddings;
    } catch (e) {
      _status = LlamaStatus.error;
      throw LlamaException('Error generating embeddings', e);
    } finally {
      // Clean up in finally block to ensure it happens even if there's an exception
      if (promptBatch != null) {
        // Free sequence IDs
        for (var i = 0; i < promptBatch.n_tokens; i++) {
          if (promptBatch.seq_id[i] != nullptr) {
            calloc.free(promptBatch.seq_id[i]);
            promptBatch.seq_id[i] = nullptr; // Set to nullptr after freeing
          }
        }
        lib.llama_batch_free(promptBatch);
      }
    }
  }

  // season management

  // Loads a saved session state
  // Returns true if successful, false if session file doesn't exist
  bool loadSession(String path) {
    final bytes = File(path).readAsBytesSync();
    final hdr = ByteData.sublistView(bytes, 0, 12);
    final magic = hdr.getUint32(0, Endian.little);
    final version = hdr.getUint32(4, Endian.little);
    if (magic != 0x4C4C5346 || version != 1) {
      throw LlamaException('Bad session header');
    }
    _nPos = hdr.getUint32(8, Endian.little);

    final stateBytes = bytes.sublist(12);
    final ptr = malloc<Uint8>(stateBytes.length)
      ..asTypedList(stateBytes.length).setAll(0, stateBytes);

    lib.llama_set_state_data(context, ptr);
    malloc.free(ptr);
    return true;
  }

  // Saves the current session state
  void saveSession(String path) {
    final size = lib.llama_get_state_size(context);
    final buf = malloc<Uint8>(size);
    lib.llama_copy_state_data(context, buf);

    final bytes = buf.asTypedList(size);

    final header = ByteData(12)
      ..setUint32(
        0,
        0x4C4C5346,
        Endian.little,
      ) // "F S L L" magic, pick anything
      ..setUint32(4, 1, Endian.little) // version
      ..setUint32(8, _nPos, Endian.little); // position

    final out = BytesBuilder()
      ..add(header.buffer.asUint8List())
      ..add(bytes);

    File(path).writeAsBytesSync(out.toBytes(), flush: true);
    malloc.free(buf);
  }

  // Ensures there is enough space in the context for a new batch of tokens.
  // If the context is full, it removes older tokens to make space, preserving
  // the first `_nKeep` tokens if specified.
  // void _ensureContextHasSpace(int tokensInBatch, int keep) {
  //   final nCtx = contextParamsDart.nCtx;

  //   if (_nPos + tokensInBatch > nCtx) {
  //     final tokensToRemove = (_nPos + tokensInBatch) - nCtx;

  //     if (tokensToRemove <= 0) {
  //       return;
  //     }

  //     final removalStartPos = keep;
  //     final removalEndPos = keep + tokensToRemove;
  //     lib.llama_kv_self_seq_rm(context, 0, removalStartPos, removalEndPos);
  //     lib.llama_kv_self_seq_add(
  //         context, 0, removalEndPos, _nPos, -tokensToRemove);
  //     _nPos -= tokensToRemove;
  //   }
  // }
}
