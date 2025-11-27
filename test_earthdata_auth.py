#!/usr/bin/env python3
"""
Earthdata IMERG listing checker.

The tests below run offline, but a real invocation still needs valid
Earthdata credentials in your ``.netrc``. To get set up:

1. Create a free Earthdata Login account: https://urs.earthdata.nasa.gov/
2. Sign in at https://disc.gsfc.nasa.gov/ to authorize the GES DISC app (one-time).
3. Add a ``.netrc`` entry (Linux/macOS: ``~/.netrc``; Windows: ``%USERPROFILE%\.netrc``):

     machine urs.earthdata.nasa.gov login <USERNAME> password <PASSWORD>

4. Restrict permissions (Linux/macOS: ``chmod 600 ~/.netrc``).
5. Run this script directly to verify your login against an IMERG directory.
"""

import os
import re
import sys
import netrc
import pytest
import requests

URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/IMERG/3B-HHR-E.MS.MRG.3IMERG/2025/02/"


def find_netrc():
    cands = [os.path.join(os.environ.get("USERPROFILE", ""), ".netrc"), os.path.expanduser("~/.netrc")]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return None


def main(
    session_factory=requests.Session,
    netrc_loader=netrc.netrc,
    find_netrc_func=find_netrc,
    exit_func=sys.exit,
    url=URL,
):
    path = find_netrc_func()
    print(f".netrc: {path or '(not found)'}")
    if not path:
        exit_func(1)
        return

    try:
        login, account, password = netrc_loader(path).authenticators("urs.earthdata.nasa.gov")
        print(f"Found creds for urs.earthdata.nasa.gov: user={login}")
    except Exception as e:
        print(f"Netrc parse/authenticator error: {e}")
        exit_func(2)
        return

    session = session_factory()
    session.trust_env = True  # honors HTTPS_PROXY if you have one
    # requests will follow redirects; GES DISC challenges and picks up .netrc
    r = session.get(url, allow_redirects=True)
    print("Final URL:", r.url)
    print("Status   :", r.status_code)
    print("Length   :", len(r.content))
    print("First 200 bytes:\n", r.text[:200])

    if r.status_code == 200 and re.search(r"\.HDF5", r.text, re.I):
        print("\nOK: listing looks good (found HDF5 links).")
    elif r.status_code in (401, 403):
        print("\nAuth problem: check .netrc username/password and GES DISC app authorization.")
    elif r.status_code == 302:
        print("\nRedirected (likely to login). Auth not being applied.")
    elif r.status_code == 404:
        print("\n404: directory not found. Try a different product/month to verify login works:")
        print("  https://gpm1.gesdisc.eosdis.nasa.gov/data/IMERG/3B-HHR.MS.MRG.3IMERG/2024/12/")
    else:
        print("\nUnexpected status; body above may show a login form if auth failed.")


if __name__ == "__main__":
    main()


# Offline-friendly tests -----------------------------------------------------


class DummyNetrc:
    def __init__(self, authenticators_return):
        self._authenticators_return = authenticators_return

    def authenticators(self, host):  # pragma: no cover - trivial
        return self._authenticators_return


class DummyResponse:
    def __init__(self, status_code, text, url="https://example.test/"):
        self.status_code = status_code
        self.text = text
        self.content = text.encode()
        self.url = url


class DummySession:
    def __init__(self, response):
        self._response = response
        self.trust_env = None

    def get(self, url, allow_redirects=True):  # pragma: no cover - trivial
        return self._response


def _exit_with(code):
    raise SystemExit(code)


def test_missing_netrc_exits_with_code_1(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            session_factory=lambda: DummySession(DummyResponse(200, "")),
            netrc_loader=lambda path: DummyNetrc(("user", None, "pass")),
            find_netrc_func=lambda: None,
            exit_func=_exit_with,
        )

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert ".netrc: (not found)" in out


def test_netrc_parse_error_exits_with_code_2(capsys):
    def bad_loader(path):
        raise netrc.NetrcParseError(path, 1, "bad")

    with pytest.raises(SystemExit) as excinfo:
        main(
            session_factory=lambda: DummySession(DummyResponse(200, "")),
            netrc_loader=bad_loader,
            find_netrc_func=lambda: "/tmp/.netrc",
            exit_func=_exit_with,
        )

    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "Netrc parse/authenticator error" in out


def test_successful_listing_reports_hdf5_links(capsys):
    response = DummyResponse(200, "<a href='file.HDF5'>file.HDF5</a>")
    main(
        session_factory=lambda: DummySession(response),
        netrc_loader=lambda path: DummyNetrc(("user", None, "pass")),
        find_netrc_func=lambda: "/tmp/.netrc",
        exit_func=_exit_with,
    )

    out = capsys.readouterr().out
    assert "OK: listing looks good" in out
    assert "Status   : 200" in out


def test_auth_problem_status_reports_warning(capsys):
    response = DummyResponse(401, "unauthorized")
    main(
        session_factory=lambda: DummySession(response),
        netrc_loader=lambda path: DummyNetrc(("user", None, "pass")),
        find_netrc_func=lambda: "/tmp/.netrc",
        exit_func=_exit_with,
    )

    out = capsys.readouterr().out
    assert "Status   : 401" in out
    assert "Auth problem" in out
