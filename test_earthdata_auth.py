#!/usr/bin/env python3
import os, sys, re, netrc, requests

URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/IMERG/3B-HHR-E.MS.MRG.3IMERG/2025/02/"

def find_netrc():
    cands = [os.path.join(os.environ.get("USERPROFILE",""), ".netrc"),
             os.path.expanduser("~/.netrc")]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    return None

def main():
    path = find_netrc()
    print(f".netrc: {path or '(not found)'}")
    if not path:
        sys.exit(1)

    try:
        login, account, password = netrc.netrc(path).authenticators("urs.earthdata.nasa.gov")
        print(f"Found creds for urs.earthdata.nasa.gov: user={login}")
    except Exception as e:
        print(f"Netrc parse/authenticator error: {e}")
        sys.exit(2)

    s = requests.Session()
    s.trust_env = True  # honors HTTPS_PROXY if you have one
    # requests will follow redirects; GES DISC challenges and picks up .netrc
    r = s.get(URL, allow_redirects=True)
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