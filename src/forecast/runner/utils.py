import io
import os
import tarfile
from pathlib import Path

from forecast.logger import LOGGER

#: Oldest ``sflkitlib`` that honours the trace budgets FORECAST configures.
#: Older releases ignore them and trace without limit, which is what made
#: individual runs produce traces of hundreds of gigabytes.
MIN_SFLKITLIB_VERSION = "0.2.3"


def docker_install(
    container, packages: list[str], conda_env: str | None = None
) -> None:
    """
    Install *packages* into *container* (called once after start).

    :param container: Docker container object (from Docker SDK).
    :param packages: List of package names to install (e.g. ["pytest", "coverage"]).
    :param conda_env: Optional name of the conda environment to install into (e.g. "forecast-env").
                      If ``None``, packages are installed into the default environment.
    :return: None
    """
    cmd = []
    if conda_env:
        cmd += ["conda", "run", "-n", conda_env]
    cmd += ["python", "-m", "pip", "install", *packages, "-q"]
    exit_code, output = container.exec_run(
        cmd,
        workdir="/",
    )
    if exit_code != 0:
        detail = (
            output.decode("utf-8", errors="replace")
            if isinstance(output, (bytes, bytearray))
            else str(output)
        )
        raise RuntimeError(
            "Failed to install packages in container "
            f"(conda_env={conda_env}): {packages}\n{detail}"
        )


def docker_install_tests(container, conda_env: str | None = None) -> None:
    """
    Install pytest and coverage into *container* (called once after start).

    :param container: Docker container object (from Docker SDK).
    :param conda_env: Optional name of the conda environment to install into (e.g. "forecast-env").
                      If ``None``, packages are installed into the default environment.
    :return: None
    """
    docker_install(container, ["pytest", "coverage"], conda_env)


def local_sflkitlib_source() -> Path | None:
    """
    Locate a local ``sflkitlib`` checkout to install into containers.

    The runtime tracer runs *inside* the container, so the trace budgets only
    take effect if the container has a version that understands them.  When
    ``sflkitlib`` is installed from a working copy on the host — as it is when
    developing the tracer — that same working copy should be used in the
    container instead of whatever PyPI currently offers.

    ``FORECAST_SFLKITLIB`` overrides the search with an explicit path to a
    checkout, wheel, or sdist.

    :returns: Path to install from, or ``None`` to fall back to PyPI.
    """
    override = os.getenv("FORECAST_SFLKITLIB")
    if override:
        path = Path(override).expanduser()
        if path.exists():
            return path
        LOGGER.warning(
            "FORECAST_SFLKITLIB points at %s, which does not exist; "
            "falling back to the published package.",
            path,
        )
        return None

    try:
        import sflkitlib
    except ImportError:
        return None

    module_path = Path(getattr(sflkitlib, "__file__", "") or "").resolve()
    if not module_path.exists() or "site-packages" in module_path.parts:
        # A regular installation; the published package is what we want.
        return None
    # Editable/source install: walk up from src/sflkitlib/__init__.py to the
    # project root that holds the build configuration.
    for parent in module_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    return None


def docker_install_sflkit(container, conda_env: str | None = None) -> None:
    """
    Install the runtime tracer into *container* (called once after start).

    :param container: Docker container object (from Docker SDK).
    :param conda_env: Optional name of the conda environment to install into (e.g. "forecast-env").
                      If ``None``, packages are installed into the default environment.
    :return: None
    """
    source = local_sflkitlib_source()
    if source is None:
        try:
            docker_install(
                container, [f"sflkitlib>={MIN_SFLKITLIB_VERSION}"], conda_env
            )
        except RuntimeError:
            # No release satisfies the pin yet: the budgets live in a version
            # that has not been published.  Tracing still works without them,
            # so install what exists rather than refusing to run — but say so,
            # because traces will be unbounded.
            LOGGER.warning(
                "No sflkitlib >= %s is available to install; falling back to the "
                "latest release. Trace budgets will NOT be enforced, so traces "
                "may grow very large. Point FORECAST_SFLKITLIB at a checkout to "
                "restore them.",
                MIN_SFLKITLIB_VERSION,
            )
            docker_install(container, ["sflkitlib"], conda_env)
        return

    LOGGER.info("Installing sflkitlib into container from %s", source)
    target = "/tmp/forecast_sflkitlib"
    container.exec_run(["rm", "-rf", target])
    _put_directory(container, source, target)
    docker_install(container, [target], conda_env)


def docker_install_all(container, conda_env: str | None = None) -> None:
    """
    Install all required packages into *container* (called once after start).

    :param container: Docker container object (from Docker SDK).
    :param conda_env: Optional name of the conda environment to install into (e.g. "forecast-env").
                      If ``None``, packages are installed into the default environment.
    :return: None
    """
    docker_install(container, ["pytest", "coverage"], conda_env)
    docker_install_sflkit(container, conda_env)


def _put_directory(container, source: Path, target: str) -> None:
    """
    Copy a host file or directory into *container* at *target*.

    :param container: Docker container object (from Docker SDK).
    :param source: Host path to copy.
    :param target: Destination path inside the container.
    """
    source = Path(source)
    container.exec_run(["mkdir", "-p", target])
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        if source.is_dir():
            for child in source.iterdir():
                # Build artefacts and virtual environments would balloon the
                # transfer without contributing to the install.
                if child.name in {".git", ".venv", "venv", "build", "dist"}:
                    continue
                tar.add(child, arcname=child.name)
        else:
            tar.add(source, arcname=source.name)
    buffer.seek(0)
    if not container.put_archive(target, buffer):
        raise RuntimeError(f"Failed to copy {source} into container at {target}.")


def write_file_to_container(container, path, content) -> None:
    """
    Write *content* to a file at *path* inside *container*.

    :param container: Docker container object (from Docker SDK).
    :param path: Path inside the container where the file should be written (e.g. "/app/test.py").
    :param content: String content to write to the file.
    :return: None
    """
    data = content.encode()
    tarstream = io.BytesIO()

    with tarfile.open(fileobj=tarstream, mode="w") as tar:
        info = tarfile.TarInfo(name=os.path.basename(path))
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    tarstream.seek(0)
    container.put_archive(os.path.dirname(path), tarstream)


def read_file_from_container(container, path, decode=True) -> str | bytes:
    """
    Read the content of a file at *path* inside *container*. Returns a string if *decode* is ``True``,
    otherwise returns bytes.

    :param container: Docker container object (from Docker SDK).
    :param path: Path inside the container of the file to read (e.g. "/app/test.py").
    :param decode: Whether to decode the file content as UTF-8 and return a string. If ``False``, returns raw bytes.
    :return: Content of the file as a string (if *decode* is ``True``) or bytes (if *decode* is ``False``).
    """
    # Get tar archive of the file
    stream, stat = container.get_archive(path)

    # Combine chunks into BytesIO
    file_like = io.BytesIO()
    for chunk in stream:
        file_like.write(chunk)
    file_like.seek(0)

    # Open the tar archive
    with tarfile.open(fileobj=file_like) as tar:
        # There should be only one file in the archive
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        content = f.read()

    if decode:
        return content.decode()
    return content


def read_file_from_client(client, image, path, decode=True):
    """
    Create a temporary container from *image*, read the file at *path* inside it, and return the content.
    The container is removed after reading. Returns a string if *decode* is ``True``, otherwise returns bytes.

    :param client: Docker client object (from Docker SDK).
    :param image: Name of the Docker image to use for the temporary container (e.g. "python:3.9-slim").
    :param path: Path inside the container of the file to read (e.g. "/app/test.py").
    :param decode: Whether to decode the file content as UTF-8 and return a string. If ``False``, returns raw bytes.
    :return: Content of the file as a string (if *decode* is ``True``) or bytes (if *decode* is ``False``).
    """
    container = client.containers.create(image, command="sleep infinity", tty=True)
    try:
        container.start()
        return read_file_from_container(container, path, decode)
    finally:
        container.remove(force=True)
