import requests, base64, os

key    = os.environ.get("T212_API_KEY", "")
secret = os.environ.get("T212_API_SECRET", "")
creds  = base64.b64encode(f"{key}:{secret}".encode()).decode()
headers = {"Authorization": f"Basic {creds}"}

r = requests.get("https://live.trading212.com/api/v0/equity/metadata/instruments", headers=headers)
instruments = r.json()

isa_tickers = set()
for inst in instruments:
    raw = inst.get("ticker", "")
    if "_US_EQ" in raw:
        clean = raw.replace("_US_EQ", "")
        if clean and clean.isalpha():
            isa_tickers.add(clean)

with open("tickers.txt") as f:
    current = [t.strip() for t in f if t.strip()]

filtered = [t for t in current if t in isa_tickers]

with open("tickers.txt", "w") as f:
    f.write("\n".join(filtered))

print(f"Before: {len(current)} | After: {len(filtered)} | Removed: {len(current)-len(filtered)}")