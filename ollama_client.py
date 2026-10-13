from __future__ import annotations

import base64
import json
import mimetypes
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import requests


APP_NAME = "Ollama Desk"
CONFIG_PATH = Path.home() / ".ollama_desk.json"
DEFAULT_MODELS = [
    "glm-ocr:latest",
    "qwen3:8b",
    "qwen2.5-coder:7b",
    "qwen2.5vl:3b",
    "bge-m3:latest",
]
MODEL_HINTS = {
    "glm-ocr:latest": "OCR / tables / formulas",
    "qwen3:8b": "Chinese chat / reasoning",
    "qwen2.5-coder:7b": "Programming / debugging",
    "qwen2.5vl:3b": "Images / documents",
    "bge-m3:latest": "Text embeddings",
}
VISION_MODELS = ("glm-ocr", "qwen2.5vl", "qwen3.5", "gemma3", "llava")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
TEXT_SUFFIXES = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".xml",
    ".html", ".css", ".sql", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".sh", ".ps1", ".log",
}


@dataclass
class Attachment:
    path: Path
    kind: str


def load_config() -> dict:
    defaults = {
        "api_url": "http://203.0.113.10:11434",
        "timeout": 300,
        "keep_alive": "10m",
        "models": DEFAULT_MODELS,
        "selected_model": "qwen3:8b",
        "thinking": False,
        "system_prompt": "You are a helpful assistant. Reply in Chinese unless asked otherwise.",
    }
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        defaults.update(data)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return defaults


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class OllamaDesk(tk.Tk):
    BG = "#111715"
    PANEL = "#19211e"
    PANEL_2 = "#202b27"
    TEXT = "#e8eee9"
    MUTED = "#93a39a"
    ACCENT = "#61d6a4"
    ACCENT_DARK = "#244d3d"
    USER_BG = "#20352d"
    ERROR = "#ff8f8f"

    def __init__(self) -> None:
        super().__init__()
        self.config_data = load_config()
        self.messages: list[dict] = []
        self.attachments: list[Attachment] = []
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.active_response: requests.Response | None = None
        self.current_assistant_text = ""
        self.thinking_started = False

        self.title(APP_NAME)
        self.geometry("1180x760")
        self.minsize(880, 620)
        self.configure(bg=self.BG)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._configure_styles()
        self._build_ui()
        self._render_welcome()
        self.after(80, self._drain_events)
        self.after(250, self.refresh_models)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure(
            "TButton", background=self.PANEL_2, foreground=self.TEXT,
            borderwidth=0, padding=(12, 8), font=("Segoe UI", 10),
        )
        style.map("TButton", background=[("active", self.ACCENT_DARK)])
        style.configure(
            "Accent.TButton", background=self.ACCENT, foreground="#08110d",
            font=("Segoe UI Semibold", 10), padding=(16, 9),
        )
        style.map("Accent.TButton", background=[("active", "#86e8be")])
        style.configure(
            "TCombobox", fieldbackground=self.PANEL_2, background=self.PANEL_2,
            foreground=self.TEXT, arrowcolor=self.ACCENT, borderwidth=0,
            padding=6,
        )
        style.map("TCombobox", fieldbackground=[("readonly", self.PANEL_2)])

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        sidebar = ttk.Frame(root, style="Panel.TFrame", width=250)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="OLLAMA\nDESK", bg=self.PANEL, fg=self.ACCENT,
            font=("Segoe UI Semibold", 18), justify="left",
        ).pack(anchor="w", padx=22, pady=(24, 18))

        ttk.Button(sidebar, text="+  New conversation", command=self.new_chat).pack(
            fill="x", padx=16, pady=(0, 20)
        )
        self._sidebar_label(sidebar, "MODEL")
        self.model_var = tk.StringVar(value=self.config_data["selected_model"])
        self.model_combo = ttk.Combobox(
            sidebar, textvariable=self.model_var, state="readonly",
            values=self.config_data["models"],
        )
        self.model_combo.pack(fill="x", padx=16)
        self.model_combo.bind("<<ComboboxSelected>>", self._model_changed)
        self.model_hint = tk.Label(
            sidebar, bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9),
            justify="left", wraplength=210,
        )
        self.model_hint.pack(anchor="w", padx=18, pady=(8, 18))

        self._sidebar_label(sidebar, "QUICK ACTIONS")
        for label, prompt in [
            ("OCR text", "Text Recognition: 请识别图片中的全部文字，保持原始结构。"),
            ("Table to Markdown", "Table Recognition: 请将图片中的表格转换为 Markdown。"),
            ("Explain file", "请阅读附件并解释其主要内容。"),
            ("Review code", "请审查附件代码，指出错误、风险和改进建议。"),
        ]:
            ttk.Button(
                sidebar, text=label, command=lambda value=prompt: self.set_prompt(value)
            ).pack(fill="x", padx=16, pady=3)

        bottom = ttk.Frame(sidebar, style="Panel.TFrame")
        bottom.pack(side="bottom", fill="x", padx=16, pady=18)
        ttk.Button(bottom, text="API settings", command=self.open_settings).pack(fill="x")
        ttk.Button(bottom, text="Export conversation", command=self.export_chat).pack(
            fill="x", pady=(7, 0)
        )

        main = ttk.Frame(root)
        main.pack(side="left", fill="both", expand=True)

        header = ttk.Frame(main)
        header.pack(fill="x", padx=26, pady=(20, 10))
        self.header_title = tk.Label(
            header, text=self.model_var.get(), bg=self.BG, fg=self.TEXT,
            font=("Segoe UI Semibold", 16),
        )
        self.header_title.pack(side="left")
        self.status_var = tk.StringVar(value="Connecting...")
        tk.Label(
            header, textvariable=self.status_var, bg=self.BG, fg=self.MUTED,
            font=("Segoe UI", 9),
        ).pack(side="right")

        chat_frame = ttk.Frame(main)
        chat_frame.pack(fill="both", expand=True, padx=26)
        self.chat = tk.Text(
            chat_frame, bg=self.BG, fg=self.TEXT, insertbackground=self.TEXT,
            relief="flat", borderwidth=0, wrap="word", state="disabled",
            font=("Segoe UI", 11), padx=8, pady=8, spacing1=3, spacing3=7,
        )
        scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)
        self.chat.tag_configure("user_label", foreground=self.ACCENT, font=("Segoe UI Semibold", 10))
        self.chat.tag_configure("assistant_label", foreground="#9db7ff", font=("Segoe UI Semibold", 10))
        self.chat.tag_configure("body", foreground=self.TEXT, lmargin1=6, lmargin2=6)
        self.chat.tag_configure("muted", foreground=self.MUTED)
        self.chat.tag_configure("error", foreground=self.ERROR)

        self.attachments_frame = ttk.Frame(main)
        self.attachments_frame.pack(fill="x", padx=30, pady=(4, 0))

        composer = tk.Frame(main, bg=self.PANEL_2, highlightthickness=1, highlightbackground="#314039")
        composer.pack(fill="x", padx=26, pady=(8, 24))
        self.input = tk.Text(
            composer, height=4, bg=self.PANEL_2, fg=self.TEXT,
            insertbackground=self.TEXT, relief="flat", borderwidth=0,
            wrap="word", font=("Segoe UI", 11), padx=12, pady=10,
        )
        self.input.pack(fill="both", expand=True)
        self.input.bind("<Return>", self._handle_return)

        toolbar = tk.Frame(composer, bg=self.PANEL_2)
        toolbar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(toolbar, text="Attach files", command=self.add_attachments).pack(side="left")
        ttk.Button(toolbar, text="Clear files", command=self.clear_attachments).pack(side="left", padx=6)
        self.thinking_var = tk.BooleanVar(value=bool(self.config_data.get("thinking", False)))
        ttk.Checkbutton(
            toolbar, text="Thinking", variable=self.thinking_var,
            command=self._thinking_changed,
        ).pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(toolbar, text="Stop", command=self.stop_generation, state="disabled")
        self.stop_button.pack(side="right", padx=(6, 0))
        self.send_button = ttk.Button(toolbar, text="Send", style="Accent.TButton", command=self.send)
        self.send_button.pack(side="right")
        self._model_changed()

    def _sidebar_label(self, parent: tk.Widget, text: str) -> None:
        tk.Label(
            parent, text=text, bg=self.PANEL, fg=self.MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w", padx=18, pady=(0, 7))

    def _render_welcome(self) -> None:
        self._append_chat("OLLAMA DESK\n", "assistant_label")
        self._append_chat(
            "Connected to your local model server. Select a model, attach files, "
            "and start a conversation. Images work with GLM-OCR and Qwen2.5-VL.\n\n",
            "muted",
        )

    def _model_changed(self, _event=None) -> None:
        model = self.model_var.get()
        self.config_data["selected_model"] = model
        self.header_title.configure(text=model)
        self.model_hint.configure(text=MODEL_HINTS.get(model, "Ollama model"))
        save_config(self.config_data)

    def _thinking_changed(self) -> None:
        self.config_data["thinking"] = self.thinking_var.get()
        save_config(self.config_data)

    def _handle_return(self, event: tk.Event):
        if event.state & 0x0001:
            return None
        self.send()
        return "break"

    def set_prompt(self, prompt: str) -> None:
        self.input.delete("1.0", "end")
        self.input.insert("1.0", prompt)
        self.input.focus_set()

    def add_attachments(self) -> None:
        files = filedialog.askopenfilenames(
            title="Attach files",
            filetypes=[
                ("Supported files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.pdf *.docx *.txt *.md *.py *.js *.json *.csv"),
                ("All files", "*.*"),
            ],
        )
        existing = {item.path for item in self.attachments}
        for name in files:
            path = Path(name)
            if path in existing:
                continue
            suffix = path.suffix.lower()
            kind = "image" if suffix in IMAGE_SUFFIXES else "document"
            self.attachments.append(Attachment(path, kind))
        self._render_attachments()

    def clear_attachments(self) -> None:
        self.attachments.clear()
        self._render_attachments()

    def _render_attachments(self) -> None:
        for child in self.attachments_frame.winfo_children():
            child.destroy()
        for index, item in enumerate(self.attachments):
            label = tk.Label(
                self.attachments_frame, text=f"  {item.path.name}  x  ",
                bg=self.ACCENT_DARK, fg=self.TEXT, cursor="hand2",
                font=("Segoe UI", 9), padx=4, pady=4,
            )
            label.pack(side="left", padx=(0, 6))
            label.bind("<Button-1>", lambda _event, i=index: self.remove_attachment(i))

    def remove_attachment(self, index: int) -> None:
        if 0 <= index < len(self.attachments):
            self.attachments.pop(index)
            self._render_attachments()

    def send(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        text = self.input.get("1.0", "end").strip()
        if not text and not self.attachments:
            return
        model = self.model_var.get()
        if model.startswith("bge-m3"):
            self._send_embedding(text, self.attachments)
            return

        images = [item for item in self.attachments if item.kind == "image"]
        if images and not model.startswith(VISION_MODELS):
            messagebox.showwarning(
                "Model does not support images",
                "Choose glm-ocr or qwen2.5vl:3b to send image files.",
            )
            return
        try:
            content, encoded_images = self._prepare_content(text, self.attachments)
        except Exception as exc:
            messagebox.showerror("Attachment error", str(exc))
            return

        display = text or "Please process the attached files."
        if self.attachments:
            display += "\n" + "\n".join(f"[Attached: {a.path.name}]" for a in self.attachments)
        self._append_message("You", display, "user_label")

        message = {"role": "user", "content": content}
        if encoded_images:
            message["images"] = encoded_images
        self.messages.append(message)
        self.input.delete("1.0", "end")
        self.clear_attachments()
        self.current_assistant_text = ""
        self.thinking_started = False
        self._append_chat("ASSISTANT\n", "assistant_label")
        self._set_busy(True)
        self.stop_event.clear()
        thinking = self.thinking_var.get()
        self.worker = threading.Thread(
            target=self._chat_worker, args=(model, thinking), daemon=True
        )
        self.worker.start()

    def _prepare_content(self, prompt: str, attachments: list[Attachment]) -> tuple[str, list[str]]:
        content = prompt or "Please process the attached files."
        images: list[str] = []
        documents: list[str] = []
        for item in attachments:
            if item.kind == "image":
                images.append(base64.b64encode(item.path.read_bytes()).decode("ascii"))
            else:
                text = self._read_document(item.path)
                documents.append(f"\n\n--- FILE: {item.path.name} ---\n{text}")
        if documents:
            content += "".join(documents)
        return content, images

    def _read_document(self, path: Path) -> str:
        suffix = path.suffix.lower()
        max_chars = 120_000
        if suffix in TEXT_SUFFIXES or not suffix:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF support requires: pip install pypdf") from exc
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        elif suffix == ".docx":
            try:
                from docx import Document
            except ImportError as exc:
                raise RuntimeError("Word support requires: pip install python-docx") from exc
            text = "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
        else:
            mime = mimetypes.guess_type(path.name)[0] or "unknown"
            raise RuntimeError(f"Unsupported document type: {suffix or mime}")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[File truncated at 120,000 characters]"
        return text

    def _chat_worker(self, model: str, thinking: bool) -> None:
        payload = {
            "model": model,
            "messages": ([{"role": "system", "content": self.config_data["system_prompt"]}]
                         if self.config_data.get("system_prompt") else []) + self.messages,
            "stream": True,
            "think": thinking,
            "keep_alive": self.config_data.get("keep_alive", "10m"),
        }
        try:
            url = self.config_data["api_url"].rstrip("/") + "/api/chat"
            with requests.post(
                url, json=payload, stream=True,
                timeout=(10, int(self.config_data.get("timeout", 300))),
            ) as response:
                self.active_response = response
                if not response.ok:
                    detail = response.text[:800]
                    raise RuntimeError(f"HTTP {response.status_code}: {detail}")
                for line in response.iter_lines():
                    if self.stop_event.is_set():
                        break
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    thinking = data.get("message", {}).get("thinking", "")
                    if chunk:
                        self.events.put(("chunk", chunk))
                    elif thinking:
                        self.events.put(("thinking", thinking))
            self.events.put(("done", None))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.active_response = None

    def _send_embedding(self, text: str, attachments: list[Attachment]) -> None:
        if any(item.kind == "image" for item in attachments):
            messagebox.showwarning(
                "Embedding model",
                "BGE-M3 accepts text only. Remove image attachments or choose a vision model.",
            )
            return
        try:
            content, _ = self._prepare_content(text, attachments)
        except Exception as exc:
            messagebox.showerror("Attachment error", str(exc))
            return
        if not content.strip():
            messagebox.showinfo("Embedding input", "Enter text or attach a text document.")
            return
        display = text or "Generate embeddings for the attached files."
        if attachments:
            display += "\n" + "\n".join(f"[Attached: {a.path.name}]" for a in attachments)
        self._append_message("You", display, "user_label")
        self.input.delete("1.0", "end")
        self.clear_attachments()
        self._set_busy(True)
        model = self.model_var.get()

        def worker() -> None:
            try:
                response = requests.post(
                    self.config_data["api_url"].rstrip("/") + "/api/embed",
                    json={"model": model, "input": content, "keep_alive": "5m"},
                    timeout=int(self.config_data.get("timeout", 300)),
                )
                response.raise_for_status()
                vector = response.json()["embeddings"][0]
                preview = ", ".join(f"{value:.5f}" for value in vector[:12])
                self.events.put(("embedding", f"Vector dimension: {len(vector)}\nPreview: [{preview}, ...]"))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def stop_generation(self) -> None:
        self.stop_event.set()
        if self.active_response is not None:
            self.active_response.close()
        self.status_var.set("Stopping...")

    def _drain_events(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "chunk":
                    if self.thinking_started and not self.current_assistant_text:
                        self._append_chat("\n", "body")
                    self.status_var.set("Generating...")
                    self.current_assistant_text += value
                    self._append_chat(value, "body")
                elif event == "thinking":
                    if not self.thinking_started:
                        self.thinking_started = True
                        self._append_chat("Thinking...\n", "muted")
                    self.status_var.set("Thinking...")
                elif event == "done":
                    if self.current_assistant_text:
                        self.messages.append({"role": "assistant", "content": self.current_assistant_text})
                    self._append_chat("\n\n", "body")
                    self._set_busy(False)
                elif event == "embedding":
                    self._append_message("Embedding", value, "assistant_label")
                    self._set_busy(False)
                elif event == "error":
                    self._append_chat(f"\nError: {value}\n\n", "error")
                    self._set_busy(False)
                elif event == "models":
                    models = value or DEFAULT_MODELS
                    self.config_data["models"] = models
                    self.model_combo.configure(values=models)
                    if self.model_var.get() not in models:
                        self.model_var.set(models[0])
                        self._model_changed()
                    self.status_var.set(f"Online | {len(models)} models")
                    save_config(self.config_data)
                elif event == "connection_error":
                    self.status_var.set("Offline | Check API settings")
        except queue.Empty:
            pass
        self.after(80, self._drain_events)

    def _set_busy(self, busy: bool) -> None:
        self.send_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.model_combo.configure(state="disabled" if busy else "readonly")
        self.status_var.set("Generating..." if busy else "Ready")

    def _append_message(self, role: str, text: str, label_tag: str) -> None:
        self._append_chat(f"{role.upper()}\n", label_tag)
        self._append_chat(text + "\n\n", "body")

    def _append_chat(self, text: str, tag: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def new_chat(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Stop the current response before starting a new chat.")
            return
        self.messages.clear()
        self.clear_attachments()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._render_welcome()

    def refresh_models(self) -> None:
        def worker() -> None:
            try:
                response = requests.get(
                    self.config_data["api_url"].rstrip("/") + "/api/tags", timeout=10
                )
                response.raise_for_status()
                names = [model["name"] for model in response.json().get("models", [])]
                self.events.put(("models", names))
            except Exception as exc:
                self.events.put(("connection_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def open_settings(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("API settings")
        dialog.geometry("520x410")
        dialog.resizable(False, False)
        dialog.configure(bg=self.PANEL)
        dialog.transient(self)
        dialog.grab_set()

        api_var = tk.StringVar(value=self.config_data["api_url"])
        timeout_var = tk.StringVar(value=str(self.config_data.get("timeout", 300)))
        keep_var = tk.StringVar(value=self.config_data.get("keep_alive", "10m"))

        def field(label: str, variable: tk.StringVar) -> None:
            tk.Label(dialog, text=label, bg=self.PANEL, fg=self.MUTED).pack(anchor="w", padx=24, pady=(14, 5))
            tk.Entry(
                dialog, textvariable=variable, bg=self.PANEL_2, fg=self.TEXT,
                insertbackground=self.TEXT, relief="flat", font=("Segoe UI", 10),
            ).pack(fill="x", padx=24, ipady=8)

        field("Ollama API URL", api_var)
        field("Request timeout (seconds)", timeout_var)
        field("Model keep-alive", keep_var)
        tk.Label(dialog, text="System prompt", bg=self.PANEL, fg=self.MUTED).pack(anchor="w", padx=24, pady=(14, 5))
        prompt = tk.Text(dialog, height=4, bg=self.PANEL_2, fg=self.TEXT, insertbackground=self.TEXT, relief="flat")
        prompt.pack(fill="x", padx=24)
        prompt.insert("1.0", self.config_data.get("system_prompt", ""))
        result_var = tk.StringVar()
        tk.Label(dialog, textvariable=result_var, bg=self.PANEL, fg=self.ACCENT).pack(anchor="w", padx=24, pady=8)

        def test_api() -> None:
            try:
                response = requests.get(api_var.get().rstrip("/") + "/api/version", timeout=8)
                response.raise_for_status()
                result_var.set(f"Connected | Ollama {response.json().get('version', '?')}")
            except Exception as exc:
                result_var.set(f"Connection failed: {exc}")

        def apply() -> None:
            try:
                timeout = int(timeout_var.get())
            except ValueError:
                messagebox.showerror("Invalid timeout", "Timeout must be an integer.", parent=dialog)
                return
            self.config_data.update({
                "api_url": api_var.get().rstrip("/"), "timeout": timeout,
                "keep_alive": keep_var.get(), "system_prompt": prompt.get("1.0", "end").strip(),
            })
            save_config(self.config_data)
            dialog.destroy()
            self.refresh_models()

        buttons = ttk.Frame(dialog, style="Panel.TFrame")
        buttons.pack(fill="x", padx=24, pady=8)
        ttk.Button(buttons, text="Test", command=test_api).pack(side="left")
        ttk.Button(buttons, text="Save", style="Accent.TButton", command=apply).pack(side="right")

    def export_chat(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".md", filetypes=[("Markdown", "*.md"), ("Text", "*.txt")]
        )
        if not path:
            return
        lines = [f"# Conversation with {self.model_var.get()}\n"]
        for message in self.messages:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"## {role}\n\n{message.get('content', '')}\n")
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def on_close(self) -> None:
        self.stop_generation()
        save_config(self.config_data)
        self.destroy()


if __name__ == "__main__":
    app = OllamaDesk()
    app.mainloop()
