"""
BaseAgent — orchestrates the five agent layers through injected interfaces.
No layer knows about any other layer; all coupling is inverted.
"""
from typing import Optional, Any, Dict, List
from agent.interfaces import (
    ILLM,
    IMemory,
    IGuardrail,
    IAugmentation,
    IEvaluation,
    IObservation,
)
import pdb

class BaseAgent:
    def __init__(
        self,
        *,
        llm: ILLM,
        memory: IMemory,
        guardrails: IGuardrail,
        augmentation: IAugmentation,
        evaluation: IEvaluation,
        observation: IObservation,
    ):
        self.llm = llm
        self.memory = memory
        self.guardrails = guardrails
        self.augmentation = augmentation
        self.evaluation = evaluation
        self.observation = observation

    async def run(
        self,
        session_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        **llm_kwargs: Any,
    ) -> str:

        with self.observation.trace("agent.run"):
            self.observation.log("info", f"Starting agent run for session {session_id}")

            # 1. Create LLM session
            await self.llm.create_session(session_id, system_prompt)

            # 2. Guardrail check on user input
            blocked = await self.guardrails.check_input(
                user_prompt, {"session_id": session_id}
            )
            if blocked:
                self.observation.log("warn", "Input rejected by guardrails")
                return "Input rejected by guardrails."

            # 3. Token limit check before inference
            allowed = await self.memory.check_against_token_limit(
                user_prompt, verbose=True
            )
            if not allowed:
                self.observation.log("warn", "Token limit exceeded")
                return "Context window limit exceeded."

            # 4. Fetch augmentation tools for the LLM
            await self.augmentation.tool_connect()
            # pdb.set_trace()
            tools = self.augmentation.show_tools()

            # 5. First inference call with tools available
            result = await self.llm.inference(
                client_id=session_id,
                prompt_string=user_prompt,
                tools=tools,
                max_tokens=max_tokens,
                **llm_kwargs,
            )

            # 6. If the LLM requested a tool call, validate and execute it
            if result.get("tool_call"):
                self.observation.log(
                    "info", f"Tool call requested: {result['tool_call']}"
                )
                tool_result = await self.augmentation.validate_llm_response(result)

                # Re-check token limit before feeding tool result back
                allowed = await self.memory.check_against_token_limit(
                    tool_result, verbose=True
                )
                if not allowed:
                    return "Context window limit exceeded after tool result."

                result = await self.llm.inference(
                    client_id=session_id,
                    prompt_string=tool_result,
                    max_tokens=max_tokens,
                    **llm_kwargs,
                )

            # 7. Guardrail check on final output
            blocked = await self.guardrails.check_output(
                result.get("output", ""), {"session_id": session_id}
            )
            if blocked:
                self.observation.log("warn", "Output rejected by guardrails")
                return "Output rejected by guardrails."

            # 8. Evaluation
            eval_result = await self.evaluation.evaluate(
                question="",
                options="",
                operand_string=result.get("output", "")
            )
            # passed = 1 if eval_result.get("passed") else 0
            # self.observation.log("info", f"eval_passed={passed}")

            # 9. Cleanup
            await self.llm.delete_session(session_id)

            return result.get("output", "")
