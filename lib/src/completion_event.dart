class CompletionEvent {
  CompletionEvent(this.promptId, {required this.success, this.errorDetails});

  final String promptId;
  final bool success;
  final String? errorDetails;
}
