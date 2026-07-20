from scripts.windows_pet_shell import _create_quit_hotkey


def test_windows_quit_hotkey_forwards_to_qt_shell():
    calls = []

    class Shell:
        def request_quit(self):
            calls.append("quit")

    class Keyboard:
        class GlobalHotKeys:
            def __init__(self, bindings):
                self.bindings = bindings

    listener = _create_quit_hotkey(Shell(), Keyboard)
    listener.bindings["<ctrl>+<shift>+q"]()

    assert calls == ["quit"]
