from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_core.language_models.llms import LLMResult
from langchain_core.callbacks.manager import Callbacks
from typing import List, Any, Union, Optional

class CompatibleChatOpenAI(ChatOpenAI):
    """A wrapper around ChatOpenAI that handles ragas compatibility issues"""
    def _convert_to_message_lists(self, messages):
        if isinstance(messages, StringPromptValue):
            return [[HumanMessage(content=messages.text)]]
        elif isinstance(messages, str):
            return [[HumanMessage(content=messages)]]
        elif isinstance(messages, list):
            if not messages:
                return [[]]
            if isinstance(messages[0], list):
                result = []
                for msg_list in messages:
                    converted_list = []
                    for msg in msg_list:
                        if isinstance(msg, str):
                            converted_list.append(HumanMessage(content=msg))
                        elif isinstance(msg, StringPromptValue):
                            converted_list.append(HumanMessage(content=msg.text))
                        elif isinstance(msg, BaseMessage):
                            converted_list.append(msg)
                        else:
                            converted_list.append(HumanMessage(content=str(msg)))
                    result.append(converted_list)
                return result
            else:
                converted_list = []
                for msg in messages:
                    if isinstance(msg, str):
                        converted_list.append(HumanMessage(content=msg))
                    elif isinstance(msg, StringPromptValue):
                        converted_list.append(HumanMessage(content=msg.text))
                    elif isinstance(msg, BaseMessage):
                        converted_list.append(msg)
                    else:
                        converted_list.append(HumanMessage(content=str(msg)))
                return [converted_list]
        else:
            return [[HumanMessage(content=str(messages))]]

    async def generate(
        self,
        messages,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        converted_messages = self._convert_to_message_lists(messages)
        return await super().agenerate(
            converted_messages,
            stop=stop,
            callbacks=callbacks,
            **kwargs
        )

    async def agenerate(
        self,
        messages,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        converted_messages = self._convert_to_message_lists(messages)
        return await super().agenerate(
            converted_messages,
            stop=stop,
            callbacks=callbacks,
            **kwargs
        ) 