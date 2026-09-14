from dataclasses import dataclass, field
from typing import Any

from .text_normalizer import normalize_text
from .nlp_preprocessor import extract_nlp_features
from .obfuscation import extract_obfuscation_features
from .security_features import extract_security_features


@dataclass
class SecurityContext:
    """
    Represents all information entering an LLM application.

    IMPORTANT:
    system_instruction is trusted application-controlled context.

    All other sources are treated as untrusted until the detection
    engine explicitly validates them.
    """

    system_instruction: str = ""
    user_prompt: str = ""

    conversation_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    retrieved_documents: list[dict[str, Any]] = field(
        default_factory=list
    )

    tool_outputs: list[dict[str, Any]] = field(
        default_factory=list
    )

    def validate(self):
        if not isinstance(self.system_instruction, str):
            raise TypeError("system_instruction must be a string")

        if not isinstance(self.user_prompt, str):
            raise TypeError("user_prompt must be a string")

        for name in [
            "conversation_history",
            "retrieved_documents",
            "tool_outputs"
        ]:
            if not isinstance(getattr(self, name), list):
                raise TypeError(f"{name} must be a list")


class ContextPreprocessor:

    def __init__(self, text_processor):
        self.text_processor = text_processor

    def process_message(self, content: str) -> dict:
        return self.text_processor(content)

    def process(self, context: SecurityContext) -> dict:
        context.validate()

        user = self.process_message(
            context.user_prompt
        )

        history = []

        for message in context.conversation_history:
            history.append({
                "role": message.get("role", "unknown"),
                "processed": self.process_message(
                    message.get("content", "")
                )
            })

        documents = []

        for document in context.retrieved_documents:
            documents.append({
                "source": document.get("source", "unknown"),
                "processed": self.process_message(
                    document.get("content", "")
                )
            })

        tools = []

        for output in context.tool_outputs:
            tools.append({
                "tool": output.get("tool", "unknown"),
                "processed": self.process_message(
                    output.get("content", "")
                )
            })

        return {
            "trusted_context": {
                "system_instruction": context.system_instruction
            },

            "untrusted_context": {
                "user_prompt": user,
                "conversation_history": history,
                "retrieved_documents": documents,
                "tool_outputs": tools
            }
        }
