import 'dart:io';

import 'package:dfc_llama/dfc_llama.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

void main() {
  // final contextParams = ContextParams();

  // print('\n## built\n');
  // contextParams.printParams();

  // print('\n## defaults\n');
  // contextParams.printParams(defaultParams: true);

  // final modelParams = ModelParams();

  // print('\n## built\n');
  // modelParams.printParams();

  // print('\n## defaults\n');
  // modelParams.printParams(defaultParams: true);

  runApp(const LlamaApp());
}

class LlamaApp extends StatefulWidget {
  const LlamaApp({super.key});

  @override
  State<LlamaApp> createState() => LlamaAppState();
}

class LlamaAppState extends State<LlamaApp> {
  final TextEditingController controller = TextEditingController();
  Llama? model;
  List<String> messages = [];
  String? modelPath;
  bool busy = false;

  void loadModel() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: 'Load Model File',
      type: FileType.any,
      allowMultiple: false,
    );

    if (result == null ||
        result.files.isEmpty ||
        result.files.single.path == null) {
      throw Exception('No file selected');
    }

    File resultFile = File(result.files.single.path!);

    final exists = await resultFile.exists();
    if (!exists) {
      throw Exception('File does not exist');
    }

    final contextParams = ContextParams();
    // contextParams.nCtx = 111;
    // contextParams.nBatch = 512;

    final samplerParams = SamplerParams();
    // samplerParams.temp = 1.0;
    // samplerParams.topK = 64;
    // samplerParams.topP = 0.95;
    // samplerParams.penaltyRepeat = 1.1;

    final modelParams = ModelParams();
    // default of 99 crashes, not enough memory
    // 30 crashed too
    modelParams.nGpuLayers = 0;

    model = Llama(
      modelPath: resultFile.path,
      libraryPath: LibraryPaths.path,
      modelParamsDart: modelParams,
      contextParamsDart: contextParams,
      samplerParams: samplerParams,
      verbose: false,
    );

    print(model!.status);
    print(model!.batch);
    print(model!.isDisposed);
    print(model!.model.address);

    setState(() {
      modelPath = resultFile.path;
    });
  }

  void onSubmitted(String value) async {
    if (model == null) {
      return;
    }

    setState(() {
      busy = true;
      controller.clear();
    });

    try {
      // Initialize chat history with system prompt
      //       ChatHistory chatHistory = ChatHistory();
      //       chatHistory.addMessage(role: Role.system, content: """
      // You are a helpful, concise assistant. Keep your answers informative but brief.""");

      //       // Add user message to history
      //       chatHistory.addMessage(role: Role.user, content: value);

      //       // Add empty assistant message that will be filled by the model
      //       chatHistory.addMessage(role: Role.assistant, content: "");

      //       // Prepare prompt for the model
      //       String prompt = chatHistory.exportFormat(ChatFormat.chatml,
      //           leaveLastAssistantOpen: true);

      print(value);

      model!.setPrompt(value);
      StringBuffer str = StringBuffer();

      while (true) {
        var (token, done) = model!.getNext();
        str.write(token);

        print(token);

        if (done) break;
      }

      messages.add(str.toString());

      print(str);
    } catch (e) {
      print('Error: ${e.toString()}');
    }

    setState(() => busy = false);
  }

  void onStop() {
    // model?.stop();
    setState(() => busy = false);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: buildHome());
  }

  Widget buildHome() {
    return Scaffold(appBar: buildAppBar(), body: buildBody());
  }

  PreferredSizeWidget buildAppBar() {
    return AppBar(
      title: Text(modelPath ?? 'No model loaded'),
      leading: IconButton(
        icon: const Icon(Icons.folder_open),
        onPressed: loadModel,
      ),
    );
  }

  Widget buildBody() {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: messages.length,
            itemBuilder: (context, index) {
              final message = messages[index];
              return ListTile(title: Text(message));
            },
          ),
        ),
        buildInputField(),
      ],
    );
  }

  Widget buildInputField() {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: controller,
              onSubmitted: onSubmitted,
              decoration: const InputDecoration(
                labelText: 'Enter your message',
                border: OutlineInputBorder(),
              ),
            ),
          ),
          busy
              ? IconButton(
                  icon: const Icon(Icons.stop),
                  onPressed: () => onStop(),
                )
              : IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: () => onSubmitted(controller.text),
                ),
        ],
      ),
    );
  }
}
