from __future__ import annotations


def tag_or_tag_dot_distance(version) -> str:
    """Match legacy `git describe --tags` formatting used by old setup.py.

    - Clean tag: `X.Y.Z`
    - With distance: `X.Y.Z.N`
    """
    tag = version.tag.public
    if getattr(version, "distance", 0) in (0, None):
        return tag
    return f"{tag}.{version.distance}"
