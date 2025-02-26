from typing import List

from pipecat.frames.frames import Frame, LLMMessagesFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from pipecat.services.openai import BaseOpenAILLMService

from openai.types.chat import ChatCompletionMessageParam

from client.client import AgentClient


class FastAPIService(BaseOpenAILLMService):

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        **kwargs,
    ):
        super().__init__(model=model, base_url=base_url, **kwargs)
        self.set_model_name(model)
        self._client = self.create_client(base_url=base_url)

    def create_client(self, base_url=None, **kwargs):
        return AgentClient(base_url=base_url, agent="chatbot")

    async def get_chat_completions(self, context, messages: List[ChatCompletionMessageParam]):
        chunk = self._client.astream(
            message=context.messages[-1]["content"],
            model=self.model_name,
            # TODO: thread-id를 구분할 수 있는 ID 정의 필요
            thread_id="random number",
            agent_config={},
            stream_tokens=True,
        )
        return chunk

    async def _process_context(self, context: str):

        await self.start_ttfb_metrics()

        chunk_stream = await self._stream_chat_completions(context)

        async for chunk in chunk_stream:
            await self.stop_ttfb_metrics()
            if isinstance(chunk, str):
                # TODO: 답변이 끝나면 agent-service-toolkit의 ChatMessage 객체가
                # chunk로 흘러들어옴.
                await self.push_frame(LLMTextFrame(chunk))

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
