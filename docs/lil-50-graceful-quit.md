# LIL-50: Graceful Desktop Quit

The desktop assistant uses one shutdown path on Windows and WSL:

1. The HUD `QUIT` button or `Ctrl+Shift+Q` calls `request_shutdown()`.
2. In WSL hybrid mode, the Windows shell sends the local bridge `quit` command
   and waits for its acknowledgement.
3. The main backend loop exits and runs the existing `stop()` cleanup for the
   active turn, audio, providers, bridge, and Windows shell.

The Windows shortcut listener belongs to the hybrid shell and is stopped when
that shell exits. Native Windows keeps the existing backend listener, so the two
paths do not run together.

Manual validation:

```text
python run_assistant.py
```

While idle and during an answer, verify both `QUIT` and `Ctrl+Shift+Q`. The
assistant should log `Stopping Live2D Assistant` followed by `Goodbye`, and no
assistant or Windows pet-shell process should remain.
