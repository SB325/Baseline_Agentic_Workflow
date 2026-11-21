import atexit
from typing import Any
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score


class DecisionEngine:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str = "kev-4b"):
        self._client = TypeSafeClient(base_url=base_url, api_key=api_key, model=model)
        atexit.register(self.close)

    def ask(
        self,
        state: str,
        questions: dict[str, Noul | Choice | Score],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Run any set of dynamic questions against a new state and return a flat dict."""
        kwargs = {"state": state, "questions": questions}
        if model:
            kwargs["model"] = model

        resp = self._client.system_one(**kwargs)

        # Unpack typed results into a single flat dictionary keyed by question name
        results: dict[str, Any] = {}
        for k, v in (resp.nouls or {}).items():
            results[k] = v.noul
        for k, v in (resp.choices or {}).items():
            results[k] = v.choice
        for k, v in (resp.scores or {}).items():
            results[k] = v.score
        return results

    def close(self) -> None:
        self._client.close()
