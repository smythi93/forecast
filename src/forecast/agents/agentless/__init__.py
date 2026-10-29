"""
Agentless, run as published, inside FORECAST.

Agentless is a pipeline of command-line stages. This adapter replays the
authors' SWE-bench recipe (``README_swebench.md`` at the pinned commit) stage by
stage, calling each stage's own ``main()`` with the recipe's arguments:

1. LLM file localization; related-element localization over compressed files
   (top 3 files); four edit-location samples at temperature 0.8, merged into
   four location sets;
2. repair on each location set: 10 samples each (1 greedy, 9 at 0.8), 40 in
   total, as search/replace edits with chain of thought;
3. regression tests: tests passing on the original repository, filtered by the
   LLM, run against every patch;
4. reproduction tests: 40 generated samples, each run on the original
   repository, one selected by majority vote and run against every patch;
5. reranking by regression and reproduction results with deduplication and
   majority voting.

FORECAST supplies three things, and changes nothing else:

* the issue: every stage reads the instance's problem statement from the
  dataset; the adapter serves a dataset holding only this instance, whose
  problem statement is :meth:`~forecast.agents.base.RepairAgent.issue_text`
  -- the issue plus, in ``FORECAST`` mode, the diagnosis;
* the model: every stage builds its LLM through ``make_model``, which returns
  :class:`ForecastDecoder`, sending each request through FORECAST's
  :meth:`~forecast.agents.models.utils.ModelClient.chat`;
* the environment: tests run in the instance's SWE-bench image through the
  harness, as upstream does.

Two deviations, documented in the paper: the embedding-based file retrieval
(OpenAI ``text-embedding-3-small``) is skipped. No Claude model offers
embeddings, so file localization is the LLM's alone, a configuration
Agentless supports. And every step may answer in up to
:data:`MIN_MAX_TOKENS` tokens, more than Agentless's 300 to 1024, in both modes. Agentless targets an older SWE-bench harness API;
:mod:`forecast.agents.agentless.compat` maps it onto the installed one without
touching Agentless.

Upstream: https://github.com/OpenAutoCoder/Agentless at the commit pinned in
``agents/agentless.json``, used from that checkout.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from forecast.agents.base import AgentMode, RepairContext
from forecast.agents.loop import LoopRepairAgent

#: Where the pinned Agentless checkout lives; override with
#: ``FORECAST_AGENTLESS_PATH``.
DEFAULT_CHECKOUT = Path(__file__).resolve().parents[4] / "agents" / "repos" / "agentless"

#: Model label passed to Agentless's command lines. Agentless only accepts a
#: fixed list of names; the label selects token counting and nothing else, as
#: the model actually used is FORECAST's (see :class:`ForecastDecoder`).
MODEL_LABEL = "claude-3-5-sonnet-20241022"
BACKEND_LABEL = "anthropic"

#: The least answer length any step gets. Agentless gives localization 300
#: tokens and repair and test generation 1024; a model that reasons before
#: answering is cut off there, so localization finds nothing and repair and
#: test samples end inside their edit or test.
MIN_MAX_TOKENS = 4096


def checkout() -> Path:
    path = Path(os.environ.get("FORECAST_AGENTLESS_PATH", DEFAULT_CHECKOUT))
    if not (path / "agentless").is_dir():
        raise RuntimeError(
            f"Agentless checkout not found at {path}; run agents/install_upstream.py "
            "or set FORECAST_AGENTLESS_PATH."
        )
    return path


def forecast_decoder_class():
    """
    Agentless's decoder interface, implemented over FORECAST's client.

    Mirrors upstream's ``AnthropicChatDecoder``: one request per sample, the
    message sent as the single user turn, no system message, per-sample usage.
    """
    from agentless.util.model import DecoderBase

    class ForecastDecoder(DecoderBase):
        client = None  # set per run by the adapter
        MAX_RETRIES = 40

        def codegen(self, message, num_samples: int = 1, prompt_cache: bool = False):
            if self.temperature == 0:
                assert num_samples == 1
            messages = (
                message
                if isinstance(message, list)
                else [{"role": "user", "content": message}]
            )
            if prompt_cache and "claude" in self.client.model and isinstance(message, str):
                # As upstream's Anthropic decoder: cache the prompt that is sent
                # several times (request_anthropic_engine).
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": message, "cache_control": {"type": "ephemeral"}}
                ]}]
            trajs = []
            for _ in range(num_samples):
                reply = self._request(messages)
                if reply is None:  # upstream yields an empty sample once retries run out
                    trajs.append(
                        {"response": "", "usage": {"completion_tokens": 0, "prompt_tokens": 0}}
                    )
                    continue
                trajs.append(
                    {
                        "response": reply.content,
                        "usage": {
                            "completion_tokens": reply.completion_tokens,
                            "prompt_tokens": reply.prompt_tokens,
                        },
                    }
                )
            return trajs

        def _request(self, messages):
            # Upstream's retry policy (request_anthropic_engine): up to 40
            # attempts, waiting 10 s times the attempt number in between.
            for attempt in range(self.MAX_RETRIES):
                try:
                    return self.client.chat(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                    )[0]
                except Exception:
                    self.logger.error("Unknown error. Waiting...", exc_info=True)
                    time.sleep(10 * attempt)
            return None

        def is_direct_completion(self) -> bool:
            return False

    return ForecastDecoder


def _dataset(row: dict):
    """A stand-in for ``datasets.load_dataset`` serving one instance."""
    from datasets import Dataset, DatasetDict

    table = Dataset.from_list([row])

    def load_dataset(name=None, split=None, **_kwargs):
        return table if split is not None else DatasetDict({"test": table})

    return load_dataset


@contextlib.contextmanager
def agentless_session(row: dict, model_client, work_dir: Path) -> Iterator[dict]:
    """
    Make Agentless's stages see one instance and talk through FORECAST.

    Patches, for the duration of the session, each stage module's
    ``load_dataset`` and ``make_model``, and runs in *work_dir*, where Agentless
    keeps its relative ``playground`` and ``logs`` directories.

    :param row: The instance's SWE-bench row, issue already replaced.
    :param model_client: FORECAST client every stage's LLM calls go through.
    :param work_dir: Directory receiving all stage outputs.
    :returns: The imported stage modules, by name.
    """
    root = str(checkout())
    if root not in sys.path:
        sys.path.insert(0, root)
    from forecast.agents.agentless import compat

    compat.install()

    import importlib

    names = {
        "localize": "agentless.fl.localize",
        "fl": "agentless.fl.FL",
        "repair": "agentless.repair.repair",
        "rerank": "agentless.repair.rerank",
        "run_regression": "agentless.test.run_regression_tests",
        "select_regression": "agentless.test.select_regression_tests",
        "generate_reproduction": "agentless.test.generate_reproduction_tests",
        "run_reproduction": "agentless.test.run_reproduction_tests",
        "run_tests": "agentless.test.run_tests",
        "preprocess": "agentless.util.preprocess_data",
        # Some stages import make_model inside functions, from here.
        "model": "agentless.util.model",
    }
    modules = {key: importlib.import_module(name) for key, name in names.items()}
    compat.adapt_modules(modules["run_tests"])

    decoder = forecast_decoder_class()
    decoder.client = model_client

    def make_model(model, backend, logger, batch_size=1, max_tokens=1024, temperature=0.0):
        return decoder(
            name=model_client.model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max(max_tokens, MIN_MAX_TOKENS),
            temperature=temperature,
        )

    load_dataset = _dataset(row)
    saved: list[tuple[Any, str, Any]] = []

    def patch(module, attr, value):
        if hasattr(module, attr):
            saved.append((module, attr, getattr(module, attr)))
            setattr(module, attr, value)

    for module in modules.values():
        patch(module, "make_model", make_model)
        patch(module, "load_dataset", load_dataset)
    previous_cwd = os.getcwd()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    try:
        yield modules
    finally:
        os.chdir(previous_cwd)
        for module, attr, value in reversed(saved):
            setattr(module, attr, value)


def _has_output(path: Path, instance_id: str) -> bool:
    """Whether a stage wrote a non-empty result for *instance_id* to *path*."""
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("instance_id") == instance_id and any(
            entry.get(key) for key in ("found_files", "found_related_locs", "found_edit_locs")
        ):
            return True
    return False


def _results_without_reproduction(folder: Path, num: int) -> None:
    """
    Record repair sample *num* in *folder* as tested without a reproduction test.

    When no generated test reproduces the issue, selection writes no
    ``reproduction_tests.jsonl``. Upstream, run on the whole benchmark, always
    has that file, just without a line for this instance; its patches then get
    results without a reproduction verdict, and rerank falls back to the
    regression tests. The same results are written here.
    """
    processed = folder / f"output_{num}_processed.jsonl"
    (folder / f"output_{num}_reproduction_test_results.jsonl").write_text(processed.read_text())


def _main(module, argv: list[str]) -> None:
    """Run a stage's command-line ``main()`` with *argv*, as the shell would."""
    saved = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
    finally:
        sys.argv = saved


class AgentlessRepairAgent(LoopRepairAgent):
    """Agentless with FORECAST's issue text, model, and image."""

    #: Recipe parameters (``README_swebench.md``).
    TOP_N = 3
    EDIT_LOCATION_SAMPLES = 4
    REPAIR_SAMPLES = 10
    REPRODUCTION_SAMPLES = 40
    CONTEXT_WINDOW = 10
    #: Docker workers for test runs; upstream's recipe uses 10 across instances.
    TEST_WORKERS = 4

    @classmethod
    def names(cls) -> list[str]:
        return ["Agentless", "agentless"]

    def __init__(
        self,
        name: str = "Agentless",
        mode: AgentMode = AgentMode.BASELINE,
        model: str = "claude-haiku-4-5-20251001",
        output_dir: str = ".",
    ):
        super().__init__(name=name, mode=mode, model=model, output_dir=output_dir)

    def run_loop(
        self,
        context: RepairContext,
        model_client,
        work_dir: Path,
        log: logging.Logger,
    ) -> tuple[str, dict[str, Any]]:
        row = dict(self.task_info(context), problem_statement=self.get_prompt(context))
        iid = row["instance_id"]
        # The harness names containers sweb.eval.<instance>.<run_id>; a run id
        # unique to this run keeps concurrent arms and reruns apart.
        tag = f"{self.mode.value}{uuid.uuid4().hex[:8]}"
        model = ["--model", MODEL_LABEL, "--backend", BACKEND_LABEL]
        w = work_dir / "agentless"
        top_n = str(self.TOP_N)
        locs = self.EDIT_LOCATION_SAMPLES
        patches = self.REPAIR_SAMPLES

        with agentless_session(row, model_client, w) as m:
            log.info("Agentless: localization")
            _main(m["localize"], ["--file_level", "--output_folder", "file_level",
                                  "--num_threads", "1", "--skip_existing", "--target_id", iid, *model])
            # Upstream skips an instance a stage localizes nothing for; with one
            # instance, that ends the run without a patch.
            if not _has_output(w / "file_level" / "loc_outputs.jsonl", iid):
                return "", {"exit_status": "no_locations_file_level"}
            _main(m["localize"], ["--related_level", "--output_folder", "related_elements",
                                  "--top_n", top_n, "--compress_assign", "--compress",
                                  "--start_file", "file_level/loc_outputs.jsonl",
                                  "--num_threads", "1", "--skip_existing", "--target_id", iid, *model])
            if not _has_output(w / "related_elements" / "loc_outputs.jsonl", iid):
                return "", {"exit_status": "no_locations_related_elements"}
            _main(m["localize"], ["--fine_grain_line_level", "--output_folder", "edit_location_samples",
                                  "--top_n", top_n, "--compress", "--temperature", "0.8",
                                  "--num_samples", str(locs),
                                  "--start_file", "related_elements/loc_outputs.jsonl",
                                  "--num_threads", "1", "--skip_existing", "--target_id", iid, *model])
            if not _has_output(w / "edit_location_samples" / "loc_outputs.jsonl", iid):
                return "", {"exit_status": "no_locations_edit_location_samples"}
            _main(m["localize"], ["--merge", "--output_folder", "edit_location_individual",
                                  "--top_n", top_n, "--num_samples", str(locs),
                                  "--start_file", "edit_location_samples/loc_outputs.jsonl"])

            folders = []
            for i in range(locs):
                folder = f"repair_sample_{i + 1}"
                folders.append(folder)
                log.info("Agentless: repair on location set %d", i + 1)
                _main(m["repair"], ["--loc_file", f"edit_location_individual/loc_merged_{i}-{i}_outputs.jsonl",
                                    "--output_folder", folder, "--loc_interval", f"--top_n={top_n}",
                                    f"--context_window={self.CONTEXT_WINDOW}",
                                    "--max_samples", str(patches), "--cot", "--diff_format",
                                    "--gen_and_process", "--num_threads", "1", "--target_id", iid, *model])

            log.info("Agentless: regression tests")
            _main(m["run_regression"], ["--run_id", f"{tag}_generate_regression_tests",
                                        "--output_file", "passing_tests.jsonl",
                                        "--instance_ids", iid])
            _main(m["select_regression"], ["--passing_tests", "passing_tests.jsonl",
                                           "--output_folder", "select_regression",
                                           "--target_id", iid, *model])
            for folder in folders:
                for num in range(patches):
                    _main(m["run_regression"], ["--regression_tests", "select_regression/output.jsonl",
                                                "--predictions_path", f"{folder}/output_{num}_processed.jsonl",
                                                "--run_id", f"{tag}_{folder}_regression_{num}",
                                                "--num_workers", str(self.TEST_WORKERS),
                                                "--instance_ids", iid])

            log.info("Agentless: reproduction tests")
            samples = str(self.REPRODUCTION_SAMPLES)
            _main(m["generate_reproduction"], ["--max_samples", samples,
                                               "--output_folder", "reproduction_test_samples",
                                               "--num_threads", "1", "--target_id", iid, *model])
            for num in range(self.REPRODUCTION_SAMPLES):
                _main(m["run_reproduction"], ["--run_id", f"{tag}_reproduction_test_generation_filter_sample_{num}",
                                              "--test_jsonl",
                                              f"reproduction_test_samples/output_{num}_processed_reproduction_test.jsonl",
                                              "--num_workers", str(self.TEST_WORKERS), "--testing",
                                              "--instance_ids", iid])
            _main(m["generate_reproduction"], ["--max_samples", samples,
                                               "--output_folder", "reproduction_test_samples",
                                               "--output_file", "reproduction_tests.jsonl", "--select",
                                               "--target_id", iid, *model])
            selected = w / "reproduction_test_samples" / "reproduction_tests.jsonl"
            for folder in folders:
                for num in range(patches):
                    if not selected.exists():
                        _results_without_reproduction(w / folder, num)
                        continue
                    _main(m["run_reproduction"], ["--test_jsonl", "reproduction_test_samples/reproduction_tests.jsonl",
                                                  "--predictions_path", f"{folder}/output_{num}_processed.jsonl",
                                                  "--run_id", f"{tag}_{folder}_reproduction_{num}",
                                                  "--num_workers", str(self.TEST_WORKERS),
                                                  "--instance_ids", iid])

            log.info("Agentless: rerank")
            _main(m["rerank"], ["--patch_folder", ",".join(f"{f}/" for f in folders),
                                "--num_samples", str(locs * patches), "--deduplicate",
                                "--regression", "--reproduction", "--output_file", "all_preds.jsonl"])

        patch = ""
        preds = w / "all_preds.jsonl"
        if preds.exists():
            for line in preds.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("instance_id") == iid:
                    patch = entry.get("model_patch") or ""
        return patch, {"exit_status": "ranked" if patch else "no_patch"}


def register(context):
    """
    Plugin entry point for FORECAST's ``forecast.plugins`` group.
    """

    @context.register_agent(names=AgentlessRepairAgent.names(), replace=True)
    class _RegisteredAgentlessRepairAgent(AgentlessRepairAgent):
        pass

    return _RegisteredAgentlessRepairAgent


__all__ = ["AgentlessRepairAgent", "agentless_session", "forecast_decoder_class", "register"]
