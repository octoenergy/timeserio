import re

from timeserio.version import get_git_version


def test_version():
    ver = get_git_version(abbrev=7)
    # Versions should comply with PEP440, which is where this regex is from
    # see https://peps.python.org/pep-0440/ and regex here:
    # https://github.com/pypa/packaging/blob/b5f0efdf3986725c2a45437e03d0ee5939c21192/packaging/version.py#L159 # noqa
    VER_RE = re.compile(
        r"""
        v?
        (?:
            (?:(?P<epoch>[0-9]+)!)?                           # epoch
            (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
            (?P<pre>                                          # pre-release
                [-_\.]?
                (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
                [-_\.]?
                (?P<pre_n>[0-9]+)?
            )?
            (?P<post>                                         # post release
                (?:-(?P<post_n1>[0-9]+))
                |
                (?:
                    [-_\.]?
                    (?P<post_l>post|rev|r)
                    [-_\.]?
                    (?P<post_n2>[0-9]+)?
                )
            )?
            (?P<dev>                                          # dev release
                [-_\.]?
                (?P<dev_l>dev)
                [-_\.]?
                (?P<dev_n>[0-9]+)?
            )?
        )
        (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
        """,
        re.VERBOSE,
    )

    assert re.match(VER_RE, ver)
