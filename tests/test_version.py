import re

from timeserio.version import get_git_version


def test_version():
    ver = get_git_version(abbrev=7)
    # Versions should comply with semver, which is where this regex is from
    # see semver.org and github.com/python-semver/python-semver
    VER_RE = re.compile(
        r"""
            ^
            (?P<major>0|[1-9]\d*)
            \.
            (?P<minor>0|[1-9]\d*)
            \.
            (?P<patch>0|[1-9]\d*)
            (?:-(?P<prerelease>
                (?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)
                (?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*
            ))?
            (?:\+(?P<build>
                [0-9a-zA-Z-]+
                (?:\.[0-9a-zA-Z-]+)*
            ))?
            $
            """,
        re.VERBOSE,
    )

    assert re.match(VER_RE, ver)
