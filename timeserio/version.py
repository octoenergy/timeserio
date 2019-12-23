# based on https://gist.github.com/jpmens/6248478
__all__ = ["get_git_version"]

import subprocess

SHA_ABBREV = 7  # number of characters to abbreviate SHA to


def call_git_describe(abbrev):
    try:
        stdout = subprocess.check_output(
            ["git", "describe", "--tags", f"--abbrev={abbrev:d}", "--dirty"]
        )
        return stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return None


def read_release_version():
    try:
        with open("RELEASE-VERSION", "r") as f:
            version = f.readlines()[0]
            return version.strip()
    except FileNotFoundError:
        return None


def write_release_version(version):
    with open("RELEASE-VERSION", "w") as f:
        f.write(f"{version:s}\n")


def get_git_version(abbrev=7):
    # Read in the version that's currently in RELEASE-VERSION.
    release_version = read_release_version()

    # First try to get the current version using “git describe”.
    version = call_git_describe(abbrev)

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
        write_release_version(version)

    # Finally, return the current version.
    return version


__version__ = get_git_version(SHA_ABBREV)


if __name__ == "__main__":
    print(__version__)
