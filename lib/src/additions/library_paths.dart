import 'dart:io';

import 'package:flutter/services.dart';

class LibraryPaths {
  LibraryPaths._();

  static String get path {
    if (Platform.isWindows) {
      return 'llama.dll';
    } else if (Platform.isLinux || Platform.isAndroid) {
      return 'libllama.so';
    } else if (Platform.isMacOS || Platform.isIOS) {
      return 'llama.framework/llama';
    } else {
      throw PlatformException(code: 'LibraryPaths: Unsupported Platform');
    }
  }
}
