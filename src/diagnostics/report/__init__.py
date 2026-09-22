"""Report package with a lightweight public entry point."""


def run(root):
    """Generate all report artifacts for a completed experiment run."""
    from .pipeline import run as generate

    return generate(root)


__all__ = ["run"]
