# based on https://gist.github.com/jpmens/6248478
__all__ = ["get_git_version"]

import os
import subprocess

SHA_ABBREV = 7  # number of characters to abbreviate SHA to


def get_home_dir(home_dir=None):
    return home_dir or os.path.normpath(
        os.path.join(os.path.abspath(__file__), os.pardir, os.pardir)
    )


def get_git_dir(home_dir=None):
    home_dir = get_home_dir(home_dir)
    return os.path.join(home_dir, ".git")


def get_release_version_file(home_dir=None):
    home_dir = get_home_dir(home_dir)
    return os.path.join(home_dir, "timeserio", "RELEASE-VERSION")


def call_git_describe(abbrev, home_dir=None):
    GIT_DIR = get_git_dir(home_dir)
    try:
        stdout = subprocess.check_output(
            [
                "git",
                f"--git-dir={GIT_DIR}",
                "describe",
                "--tags",
                f"--abbrev={abbrev:d}",
                "--dirty",
            ],
            stderr=subprocess.PIPE
        )
        return stdout.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def read_release_version(home_dir=None):
    RELEASE_VERSION = get_release_version_file(home_dir)
    try:
        with open(RELEASE_VERSION, "r") as f:
            version = f.readlines()[0]
            return version.strip()
    except FileNotFoundError:
        return None


def write_release_version(version, home_dir=None):
    RELEASE_VERSION = get_release_version_file(home_dir)
    with open(RELEASE_VERSION, "w") as f:
        f.write(f"{version:s}\n")


def get_git_version(abbrev=7, home_dir=None):
    # Read in the version that's currently in RELEASE-VERSION.
    release_version = read_release_version(home_dir)
    # First try to get the current version using “git describe”.
    version = call_git_describe(abbrev, home_dir)

    # If that doesn't work, fall back on the value that's in
    # RELEASE-VERSION.
    if version is None:
        version = release_version

    # If we still don't have anything, that's an error.
    if version is None:
        raise ValueError("Cannot find the version number!")

    # If the current version is different from what's in the
    # RELEASE-VERSION file, update the file to be current.
    if version != release_version:
        write_release_version(version, home_dir)

    # Finally, return the current version.
    return version


# Allow executing with extra globals from exec()
try:
    home_dir = HOME_DIR  # type: ignore
except NameError:
    home_dir = None

__version__ = get_git_version(SHA_ABBREV, home_dir)


if __name__ == "__main__":
    print(__version__)
