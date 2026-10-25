from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def resolve_output_dir(path):
    output = Path(path).expanduser()
    if output.is_absolute():
        return output

    if output.parts and output.parts[0].casefold() == "runs":
        output = Path(*output.parts[1:])

    runs_dir = RUNS_DIR.resolve()
    resolved = (runs_dir / output).resolve()
    try:
        resolved.relative_to(runs_dir)
    except ValueError as exc:
        raise ValueError(f"Relative output path must stay inside {runs_dir}: {path}") from exc
    return resolved
