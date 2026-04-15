import asyncio

from src.llm import GemmaTextVisionLLM, Message


class FakeGemma:
    model_id = "fake/gemma"

    def __init__(self):
        self.chat_calls = []
        self.stream_calls = []

    def preload(self):
        pass

    def cleanup(self):
        pass

    async def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return "bonjour"

    async def chat_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        for chunk in ["bon", "jour"]:
            yield chunk


def test_gemma_text_vision_llm_routes_last_user_message_to_text():
    llm = GemmaTextVisionLLM(FakeGemma())
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Salut"),
        Message(role="assistant", content="Bonjour"),
        Message(role="user", content="Comment vas-tu ?"),
    ]

    result = asyncio.run(llm.chat(messages))

    assert result.content == "bonjour"
    call = llm.gemma.chat_calls[0]
    assert call["text"] == "Comment vas-tu ?"
    assert len(call["history"]) == 3
    assert call["history"][0]["role"] == "system"
    assert call["images"] is None


def test_gemma_text_vision_llm_stream_includes_screen_context_when_enabled():
    llm = GemmaTextVisionLLM(FakeGemma())
    llm._include_screen_in_conversation = True
    llm._get_screen_context = lambda: ["frame"]
    messages = [Message(role="user", content="Que vois-tu ?")]

    chunks = asyncio.run(_collect(llm.chat_stream(messages)))

    assert "".join(chunks) == "bonjour"
    call = llm.gemma.stream_calls[0]
    assert call["images"] == ["frame"]


async def _collect(gen):
    data = []
    async for item in gen:
        data.append(item)
    return data
