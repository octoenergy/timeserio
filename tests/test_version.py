import re

from timeserio.version import get_git_version


def test_version():
    ver = get_git_version(abbrev=7)
    VER_RE = r"^(\d+.\d+.\d+)(-\d+)*(-g\S{7})?(-dirty)?$"
    assert re.match(VER_RE, ver)
