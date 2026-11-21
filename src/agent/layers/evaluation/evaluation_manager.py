## Description
#   This class facilitates agent evaluation to ensure quality of 
#   agent outputs in terms of format, completeness, and accuracy.
#   This class shares no members between instances. 
#   S.F.B
##

# Imports
import os, sys
import pdb
import json
import subprocess
import argparse
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from orchestration.llm_orchestrator import LLM
from memory.memory_manager import MemoryManager
from augmentation.augmentation_manager import AugmentationManager
from agent.load_folder_yamls import load_folder_yamls, prune_to_matching_key
from agent.layers.evaluation.kev_client import DecisionEngine, Choice, Noul, Score

load_dotenv()
kev_container = os.getenv("KEV_CONTAINER_NAME")
kev_port = os.getenv("KEV_PORT")

class QuestionSpec(BaseModel):
    """One evaluation question expressed as plain data — no typesafe_sdk import needed by caller."""
    type: Literal["noul", "choice", "score"] = "noul"
    text: str
    options: list[str] | None = None   # used for choice options / score criteria

class EvaluationInput(BaseModel):
    state: str
    questions: dict[str, QuestionSpec]  # question_name -> spec


class EvaluationManager():
    def __init__(self, *, eval_config):
        self.config = eval_config
        self.engine = DecisionEngine(
            base_url=self.get_kev_ip(), 
            api_key="local", 
            model="kev-0.8b"
        )

    def get_kev_ip(self):
        try:
            ip = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                    kev_container,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            ip = ip.stdout.strip()
            return f'http://{ip}:{kev_port}'

        except Exception as e:
            print(f"Failed to get KEV IP address:\n{e}")
            return None

    def format_eval(self, fmt: str):
        if 'json' == fmt:
            try: 
                test_json = json.loads(fmt)
            except:
                print('Not Valid Json!')
                return {'Passed': False, 'Test': 'JSON Formatting'}
        else:
            pass

    def _to_kev_questions(self, questions: dict[str, QuestionSpec]):
        """Convert plain QuestionSpecs into typesafe_sdk Noul/Choice/Score objects.

        The caller never needs to import typesafe_sdk; this mapping is internal.
        """
        kev_questions: dict[str, Noul | Choice | Score] = {}
        for name, spec in questions.items():
            if spec.type == "noul":
                kev_questions[name] = Noul(instructions=spec.text)
            elif spec.type == "choice":
                options = {opt: opt for opt in (spec.options or [])}
                kev_questions[name] = Choice(instructions=spec.text, criteria=options)
            elif spec.type == "score":
                kev_questions[name] = Score(
                    instructions=spec.text,
                    criteria=spec.options or [],
                )
        return kev_questions

    def evaluate(self, state: str, questions: dict[str, QuestionSpec]) -> dict[str, Any]:
        # Validate the input shape without exposing typesafe_sdk to the caller
        EvaluationInput(state=state, questions=questions)

        kev_questions = self._to_kev_questions(questions)
        out1 = self.engine.ask(state=state, questions=kev_questions)
        return out1


def main(config):
    em = EvaluationManager(eval_config=config)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-client", "--client_id", required=False, help="Session Identifier.")
    # parser.add_argument("-s", "--system_prompt", help="System prompt for LLM session.")
    # parser.add_argument("-p", "--prompt", help="Prompt for the LLM")
    # args = parser.parse_args()

    config = load_folder_yamls({
        'path': 'layers/evaluation',
        'key': 'test'
    })
    em = EvaluationManager(eval_config=config)

    # Example: run plain-data questions against a state (no typesafe_sdk import needed)
    state = "User forgot password and is locked out."
    questions = {
        "is_security": QuestionSpec(type="noul", text="Is this a security breach?"),
        "route": QuestionSpec(type="choice", text="Queue?", options=["support", "secops"]),
        "severity": QuestionSpec(
            type="score",
            text="How severe is the reported issue?",
            options=[
                "Cosmetic; no impact to functionality",
                "Broken or degraded feature, but workaround exists",
                "Blocking issue; no workaround exists",
            ],
        ),
    }
    result = em.evaluate(state=state, questions=questions)
    pdb.set_trace()
    print(f"State: {state}\n{result}")
    