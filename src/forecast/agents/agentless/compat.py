"""
Agentless's SWE-bench harness calls, mapped onto the installed harness.

Agentless (pinned commit) was written against an older ``swebench`` whose API
has since moved. Its logic is kept as is; only these call sites are adapted:

* ``swebench.harness.test_spec`` exported ``TestSpec`` and the script-list
  builders; they now live in submodules, and ``TestSpec`` gained three
  required fields (``language``, ``docker_specs``, ``namespace``), filled with
  the values ``make_test_spec`` uses for Python instances and locally built
  images;
* ``swebench.harness.utils.get_test_directives`` moved to
  ``swebench.harness.test_spec.python``;
* ``get_dataset_from_preds`` gained a required ``rewrite_reports`` (False is
  the old behaviour);
* ``build_env_images`` no longer defaults its image tags to ``latest``;
* ``run_instance`` returned ``(instance_id, {instance_id: {"resolved": ...}})``;
  it now returns ``{"completed": ..., "resolved": ...}``. Only Agentless's own
  reference is adapted (:func:`adapt_modules`), so FORECAST's harness calls see
  the current API;
* ``get_logs_eval(log_path)`` now takes the instance's test spec first and only
  parses output between ``Start/End Test Output`` markers, which Agentless's
  own eval scripts do not print; the old semantics are restored, with the test
  spec recovered from the log path (``.../<instance_id>/test_output.txt``);
* its Docker client (:class:`_Client`) gets a longer timeout
  (:data:`DOCKER_TIMEOUT`) and lists images once per process, instance images
  only: before each round of tests Agentless lists every image, including
  intermediate layers, only to print how many instance images exist (it never
  removes them). With thousands of layers and many parallel workers, the
  daemon, which answers such listings one at a time, took up to an hour per
  round, or longer than the library's 60 seconds.

Install with :func:`install` before importing Agentless's test modules, and
call :func:`adapt_modules` on them after.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

_INSTALLED = False

#: Seconds Agentless's Docker client waits for the daemon.
DOCKER_TIMEOUT = 900


def install() -> None:
    """Patch the harness modules Agentless imports from; idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    import swebench.harness.grading as grading
    import swebench.harness.run_evaluation as run_evaluation
    import swebench.harness.test_spec as test_spec_pkg
    import swebench.harness.utils as utils
    from swebench.harness.test_spec import create_scripts, python, test_spec
    from swebench.harness.test_spec.test_spec import make_test_spec

    base = test_spec.TestSpec

    @dataclasses.dataclass
    class TestSpec(base):  # noqa: D401 - same class, old constructor
        language: str = "py"
        docker_specs: dict = dataclasses.field(default_factory=dict)
        namespace: str = None

    test_spec_pkg.TestSpec = TestSpec
    test_spec_pkg.make_env_script_list = create_scripts.make_env_script_list
    test_spec_pkg.make_repo_script_list = create_scripts.make_repo_script_list
    utils.get_test_directives = python.get_test_directives

    get_dataset_from_preds = run_evaluation.get_dataset_from_preds

    def _get_dataset_from_preds(
        dataset_name, split, instance_ids, predictions, run_id,
        rewrite_reports=False, exclude_completed=True,
    ):
        return get_dataset_from_preds(
            dataset_name, split, instance_ids, predictions, run_id,
            rewrite_reports, exclude_completed,
        )

    run_evaluation.get_dataset_from_preds = _get_dataset_from_preds

    get_logs_eval = grading.get_logs_eval
    specs: dict[str, object] = {}

    def _spec(instance_id: str):
        if instance_id not in specs:
            from swebench.harness.utils import load_swebench_dataset

            for row in load_swebench_dataset("princeton-nlp/SWE-bench_Lite", "test"):
                if row["instance_id"] == instance_id:
                    specs[instance_id] = make_test_spec(row, namespace=None)
                    break
        return specs[instance_id]

    def _get_logs_eval(*args):
        if len(args) == 2:  # already the new call shape
            return get_logs_eval(*args)
        # The old semantics Agentless relies on (swebench 2.x): its own eval
        # scripts print no Start/End Test Output markers, which the new
        # version requires. A log counts when git reported applying the patch
        # and no harness failure marker appears; the whole log after the
        # applied-patch marker is parsed.
        from swebench.harness.constants import (
            APPLY_PATCH_FAIL,
            APPLY_PATCH_PASS,
            RESET_FAILED,
            TESTS_ERROR,
            TESTS_TIMEOUT,
        )
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

        (log_path,) = args
        spec = _spec(Path(log_path).parent.name)
        content = Path(log_path).read_text(errors="replace")
        failed = [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT,
                  "Failed to reset task environment"]
        if any(x in content for x in failed) or "applied patch" not in content.lower():
            return {}, False
        content = content.split(f"{APPLY_PATCH_PASS} (pred)")[-1]
        return MAP_REPO_TO_PARSER[spec.repo](content, spec), True

    grading.get_logs_eval = _get_logs_eval

    import swebench.harness.docker_build as docker_build

    build_env_images = docker_build.build_env_images

    def _build_env_images(client, dataset, force_rebuild=False, max_workers=4):
        # The image tags used to default to "latest"; now they must be given.
        return build_env_images(
            client, dataset, force_rebuild, max_workers,
            namespace=None, instance_image_tag="latest", env_image_tag="latest",
        )

    docker_build.build_env_images = _build_env_images
    _INSTALLED = True


#: Image references Agentless matches its instance image keys against.
INSTANCE_IMAGES = ["sweb.eval.*", "*/sweb.eval.*"]
_LISTING = None


class _Images:
    """A client's images, whose ``list`` returns the instance images, listed once."""

    def __init__(self, images):
        self._images = images

    def __getattr__(self, name):
        return getattr(self._images, name)

    def list(self, *args, **kwargs):
        global _LISTING
        if _LISTING is None:
            _LISTING = self._images.list(filters={"reference": INSTANCE_IMAGES})
        return _LISTING


class _Client:
    """A Docker client whose ``images`` is :class:`_Images`; all else passes through."""

    def __init__(self, client):
        self._client = client
        self.images = _Images(client.images)

    def __getattr__(self, name):
        return getattr(self._client, name)


def adapt_modules(run_tests_module) -> None:
    """
    Give Agentless's ``run_tests`` module ``run_instance`` with its old result
    shape, and a Docker client with :data:`DOCKER_TIMEOUT`. Patches that
    module's own references only.
    """
    import types

    import docker
    import swebench.harness.run_evaluation as run_evaluation

    def from_env(**kwargs):
        kwargs.setdefault("timeout", DOCKER_TIMEOUT)
        return _Client(docker.from_env(**kwargs))

    run_tests_module.docker = types.SimpleNamespace(from_env=from_env, errors=docker.errors)

    current = run_evaluation.run_instance

    def run_instance(test_spec, pred, *args, **kwargs):
        outcome = current(test_spec, pred, *args, **kwargs)
        instance_id = test_spec.instance_id
        return instance_id, {instance_id: {"resolved": bool(outcome.get("resolved"))}}

    run_tests_module.run_instance = run_instance
