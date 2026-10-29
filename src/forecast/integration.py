"""
Integration layer between FORECAST diagnoses and repair agents.

Provides serialization helpers (:class:`DiagnosisSerializer`) that convert
:class:`~forecast.diagnosis.CauseChain` objects to JSON, Markdown, or plain
text, and loaders that recover the rendered chain from a stored diagnosis.
Agents inject that text through
:meth:`~forecast.agents.base.RepairAgent.issue_text`.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from forecast.diagnosis import (
    CauseChain,
    Diagnosis,
    Atom,
    Conjunction,
    Disjunction,
    Divergence,
)


class DiagnosisSerializer:
    """
    Serialize :class:`~forecast.diagnosis.Diagnosis` objects to various formats.
    """

    @staticmethod
    def to_json(diagnosis: Diagnosis) -> Dict[str, Any]:
        """
        Convert a diagnosis to a JSON-serializable dictionary.

        Recursively converts :class:`~forecast.diagnosis.CauseChain`,
        :class:`~forecast.diagnosis.Conjunction`,
        :class:`~forecast.diagnosis.Disjunction`, and
        :class:`~forecast.diagnosis.Atom` nodes.

        :param diagnosis: Diagnosis object to serialize.
        :returns: Dictionary representation suitable for ``json.dump``.
        """
        if isinstance(diagnosis, CauseChain):
            return {
                "type": "cause_chain",
                "discriminability": diagnosis.discriminability,
                "mean_discriminability": diagnosis.mean_discriminability,
                "level": diagnosis.level,
                # The rendered chain is what gets injected into agent prompts,
                # so it is part of the serialized form rather than something
                # every consumer has to reconstruct from the causes.
                "text": diagnosis.to_text(),
                "causes": [
                    DiagnosisSerializer.to_json(cause) for cause in diagnosis.causes
                ],
            }
        elif isinstance(diagnosis, Conjunction):
            return {
                "type": "conjunction",
                "discriminability": diagnosis.discriminability,
                "level": diagnosis.level,
                "atoms": [
                    DiagnosisSerializer.to_json(atom) for atom in diagnosis.atoms
                ],
            }
        elif isinstance(diagnosis, Disjunction):
            return {
                "type": "disjunction",
                "discriminability": diagnosis.discriminability,
                "level": diagnosis.level,
                "atoms": [
                    DiagnosisSerializer.to_json(atom) for atom in diagnosis.atoms
                ],
            }
        elif isinstance(diagnosis, Divergence):
            return {
                "type": "divergence",
                "caller": diagnosis.caller,
                "callee": diagnosis.callee,
                "feature": str(diagnosis.feature),
                "discriminability": diagnosis.discriminability,
                "level": diagnosis.level,
                "text": diagnosis.to_text(),
            }
        elif isinstance(diagnosis, Atom):
            return {
                "type": "atom",
                "feature": str(diagnosis.feature),
                "negated": diagnosis.negated,
                "discriminability": diagnosis.discriminability,
                "level": diagnosis.level,
                "text": diagnosis.to_text(),
            }
        else:
            return {"type": "unknown", "value": str(diagnosis)}

    @staticmethod
    def to_markdown(diagnosis: CauseChain) -> str:
        """
        Render a :class:`~forecast.diagnosis.CauseChain` as Markdown.

        Produces a numbered list of steps, each showing the discriminability
        score and the natural-language cause text, connected by ``⬇️ CAUSES``
        arrows.

        :param diagnosis: CauseChain to render.
        :returns: Markdown-formatted string.
        """
        lines = [
            "# Failure Diagnosis",
            "",
            "This diagnosis shows the cause-effect chain leading to the test failure.",
            "",
        ]

        for i, cause in enumerate(diagnosis.causes, start=1):
            lines.append(f"## Step {i}")
            lines.append(f"**Discriminability**: {cause.discriminability:.4f}")
            lines.append("")
            lines.append(cause.to_text())
            lines.append("")
            if i < len(diagnosis.causes):
                lines.append("⬇️ **CAUSES**")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_plain_text(diagnosis: CauseChain, numbered: bool = True) -> str:
        """
        Render a :class:`~forecast.diagnosis.CauseChain` as plain text.

        :param diagnosis: CauseChain to render.
        :param numbered: When ``True`` (default) each step is preceded by its
            1-based index.
        :returns: Plain-text string.
        """
        if numbered:
            return diagnosis.to_text()
        else:
            # Unnumbered version
            parts = []
            for cause in diagnosis.causes:
                parts.append(cause.to_text())
            return "\n CAUSES \n".join(parts)


def save_diagnosis(
    diagnosis: CauseChain,
    output_path: Path,
    file_format: str = "json",
) -> None:
    """
    Serialize a :class:`~forecast.diagnosis.CauseChain` and write it to a file.

    :param diagnosis: CauseChain to save.
    :param output_path: Destination file path.  Parent directories are created
        automatically.
    :param file_format: Output format — ``"json"`` (default), ``"markdown"``,
        or ``"text"``.
    :raises ValueError: If *file_format* is not one of the accepted values.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "json":
        data = DiagnosisSerializer.to_json(diagnosis)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
    elif file_format == "markdown":
        content = DiagnosisSerializer.to_markdown(diagnosis)
        with open(output_path, "w") as f:
            f.write(content)
    elif file_format == "text":
        content = DiagnosisSerializer.to_plain_text(diagnosis)
        with open(output_path, "w") as f:
            f.write(content)
    else:
        raise ValueError(f"Unknown file_format: {file_format}")


def load_diagnosis_json(input_path: Path) -> Dict[str, Any]:
    """
    Load a diagnosis from a JSON file previously written by :func:`save_diagnosis`.

    :param input_path: Path to the JSON file.
    :returns: Dictionary representation of the diagnosis.
    """
    with open(input_path) as f:
        return json.load(f)


def diagnosis_text_from_json(data: Dict[str, Any]) -> Optional[str]:
    """
    Recover the rendered cause chain from a serialized diagnosis.

    This is the text that gets injected into an agent's prompt.  Diagnoses
    written before the chain text was stored explicitly are reconstructed from
    their individual causes, so previously cached results stay usable.

    :param data: Dictionary as produced by :meth:`DiagnosisSerializer.to_json`.
    :returns: The rendered chain; ``""`` for a diagnosis without causes;
        ``None`` when *data* carries no diagnosis.
    """
    if not isinstance(data, dict):
        return None

    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text

    causes = data.get("causes")
    if not isinstance(causes, list):
        return None
    # A diagnosis that ran and kept no cause is a finding -- the passing and
    # failing runs are indistinguishable -- and must stay distinguishable from
    # a missing diagnosis, so it is "" rather than None.
    if not causes:
        return ""

    lines = []
    for index, cause in enumerate(causes, start=1):
        rendered = _cause_text(cause)
        if not rendered:
            continue
        discriminability = cause.get("discriminability")
        if isinstance(discriminability, (int, float)):
            lines.append(
                f"{index}. {rendered} (discriminability: {discriminability:.2f})"
            )
        else:
            lines.append(f"{index}. {rendered}")
    return "\n LEADS TO \n".join(lines)


def _cause_text(cause: Dict[str, Any]) -> Optional[str]:
    """Render one serialized cause, joining compound diagnoses with AND / OR."""
    if not isinstance(cause, dict):
        return None
    text = cause.get("text")
    if isinstance(text, str) and text.strip():
        return text
    atoms = cause.get("atoms")
    if isinstance(atoms, list) and atoms:
        separator = " OR " if cause.get("type") == "disjunction" else " AND "
        parts = [str(atom.get("text")) for atom in atoms if atom.get("text")]
        if parts:
            return separator.join(parts)
    return None
