import requests

from pipecat.frames.frames import Frame, LLMMessagesFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame

from pipecat.services.ai_services import LLMService

from openai.types.chat import ChatCompletionMessageParam


class FastAPILLMContext:
    def __init__(
        self,
        messages: list[ChatCompletionMessageParam] | None = None,
    ):
        self._messages: list[ChatCompletionMessageParam] = messages if messages else []

    @staticmethod
    def from_messages(messages: list[dict]) -> "FastAPILLMContext":
        context = FastAPILLMContext()
        
        for message in messages:
            if "name" not in message:
                message["name"] = message["role"]
            context.add_message(message)
        return context

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        return self._messages

    def add_message(self, message: ChatCompletionMessageParam):
        self._messages.append(message)

    def add_messages(self, messages: list[ChatCompletionMessageParam]):
        self._messages.extend(messages)

    def set_messages(self, messages: list[ChatCompletionMessageParam]):
        self._messages[:] = messages

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        return self._messages


class BaseFastAPIService(LLMService):

    def __init__(
        self,
        req_url: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._req_url = req_url

    async def _process_context(self, context: OpenAILLMContext):
        await self.start_ttfb_metrics()

        messages = context.messages

        response = requests.post(
            url=self._req_url,
            json={
                "state": {
                    "messages": [
                        {
                            "type": "human",
                            "content": content,
                        }
                    ],
                    "is_last_step": False,
                    "remaining_steps": 10,
                },
                "config": {
                    "configurable": {
                        "thread_id": message.author.name,
                        "user_id": message.author.id,
                    }
                },
            },
            stream=True
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
