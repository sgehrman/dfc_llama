import 'dart:ffi';

import 'package:dfc_llama/src/chat.dart';
import 'package:dfc_llama/src/llama.dart';
import 'package:dfc_llama/src/llama_cpp.dart';
import 'package:ffi/ffi.dart';

extension LlamaExtension on Llama {
  String chatTemplate() {
    final ptr = lib.llama_model_chat_template(model, nullptr);

    //  String piece = '';
    //     final bytes = buf.cast<Uint8>().asTypedList(n);
    //     try {
    //       piece = utf8.decode(bytes);
    //     } catch (e) {
    //       piece = utf8.decode(bytes, allowMalformed: true);
    //     }

    if (ptr != nullptr) {
      final utf8Ptr = ptr.cast<Utf8>();

      return utf8Ptr.toDartString();
    }

    return 'chatml';
  }

  String applyTemplate(String template, List<Message> messages) {
    final templatePtr = template.toNativeUtf8().cast<Char>();
    final nMsg = messages.length;
    // Allocate array of llama_chat_message
    final chatPtr = calloc<llama_chat_message>(nMsg);
    final allocatedRoles = <Pointer<Char>>[];
    final allocatedContents = <Pointer<Char>>[];
    try {
      for (var i = 0; i < nMsg; i++) {
        final msg = messages[i];
        final rolePtr = msg.role.value.toNativeUtf8().cast<Char>();
        final contentPtr = msg.content.toNativeUtf8().cast<Char>();
        chatPtr[i].role = rolePtr;
        chatPtr[i].content = contentPtr;
        allocatedRoles.add(rolePtr);
        allocatedContents.add(contentPtr);
      }

      // Prepare output buffer
      // Estimate buffer size: 2x total content length (conservative)
      final totalLen = messages.fold<int>(
        0,
        (sum, m) => sum + m.content.length + m.role.value.length,
      );

      final bufLen = totalLen * 2 + 1024;
      final bufPtr = calloc<Char>(bufLen);
      try {
        final written = lib.llama_chat_apply_template(
          templatePtr,
          chatPtr,
          nMsg,
          true, // add_assistant
          bufPtr,
          bufLen,
        );
        if (written <= 0) {
          throw LlamaException(
            'llama_chat_apply_template failed or buffer too small',
          );
        }
        final result = bufPtr.cast<Utf8>().toDartString(length: written);

        return result;
      } finally {
        calloc.free(bufPtr);
      }
    } finally {
      allocatedRoles.forEach(calloc.free);
      allocatedContents.forEach(calloc.free);

      calloc.free(chatPtr);
      calloc.free(templatePtr);
    }
  }
}
