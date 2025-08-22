"""
Keyboard controller utility for key state tracking using pynput.
"""
from pynput import keyboard
from collections import defaultdict
import threading
from typing import Any, DefaultDict


class KeyboardController:
    """
    General keyboard state tracker using pynput.
    """

    def __init__(self) -> None:
        self.key_states: DefaultDict[Any, bool] = defaultdict(bool)
        self.lock = threading.Lock()

        # 启动监听线程
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key: Any) -> None:
        with self.lock:
            self.key_states[key] = True

    def _on_release(self, key: Any) -> None:
        with self.lock:
            self.key_states[key] = False

    def is_pressed(self, key: Any) -> bool:
        """
        Check if the specified key is currently pressed.
        Args:
            key: pynput.keyboard.Key or pynput.keyboard.KeyCode
        Returns:
            True if pressed, False otherwise
        """
        with self.lock:
            return self.key_states.get(key, False)

    def close(self) -> None:
        """
        Release resources and stop the listener.
        """
        self.listener.stop()
