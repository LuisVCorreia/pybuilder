import requests
from typing import Dict, List, Tuple
from web3 import Web3


# The same list of relays that rbuilder uses:
RELAY_URLS = [
    "https://0xac6e77dfe25ecd6110b8e780608cce0dab71fdd5ebea22a16c0205200f2f8e2e3ad3b71d3499c54ad14d6c21b41a37ae@boost-relay.flashbots.net",
    "https://0x8b5d2e73e2a3a55c6c87b8b6eb92e0149a125c852751db1422fa951e42a09b82c142c3ea98d0d9930b056a3bc9896b8f@bloxroute.max-profit.blxrbdn.com",
    "https://0xad0a8bb54565c2211cee576363f3a347089d2f07cf72679d16911d740262694cadb62d7fd7483f27afd714ca0f1b9118@bloxroute.ethical.blxrbdn.com",
    "https://0xb0b07cd0abef743db4260b0ed50619cf6ad4d82064cb4fbec9d3ec530f7c5e6793d9f286c4e082c0244ffb9f2658fe88@bloxroute.regulated.blxrbdn.com",
    "https://0xb3ee7afcf27f1f1259ac1787876318c6584ee353097a50ed84f51a1f21a323b3736f271a895c7ce918c038e4265918be@relay.edennetwork.io",
    "https://0x98650451ba02064f7b000f5768cf0cf4d4e492317d82871bdc87ef841a0743f69f0f1eea11168503240ac35d101c9135@mainnet-relay.securerpc.com",
    "https://0xa1559ace749633b997cb3fdacffb890aeebdb0f5a3b6aaa7eeeaf1a38af0a8fe88b9e4b1f61f236d2e64d95733327a62@relay.ultrasound.money",
    "https://0xa7ab7a996c8584251c8f925da3170bdfd6ebc75d50f5ddc4050a6fdc77f2a3b5fce2cc750d0865e05d7228af97d69561@agnostic-relay.net",
    "https://0xa15b52576bcbf1072f4a011c0f99f9fb6c66f3e1ff321f11f461d15e31b1cb359caa092c71bbded0bae5b5ea401aab7e@aestus.live",
    "https://0x8c7d33605ecef85403f8b7289c8058f440cbb6bf72b055dfe2f3e2c6695b6a1ea5a9cd0eb3a7982927a463feb4c3dae2@relay.wenmerge.com",
]



def get_one_delivered_payload(relay_url: str, block_number: int, timeout: float = 5.0) -> Dict:
    """
    Fetch a single ProposerPayloadDelivered from a relay for block_number,
    handling both wrapped and unwrapped JSON.
    """
    url = relay_url.rstrip("/") + "/relay/v1/data/bidtraces/proposer_payload_delivered"
    params = {"block_number": block_number}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        # This will catch ConnectionError (DNS), timeouts, HTTP errors, etc
        print(f"⚠️  Relay {relay_url} failed: {e}")
        return None
    
    body = resp.json()

    if isinstance(body, dict) and "Ok" in body:
        payloads = body["Ok"] or []
    elif isinstance(body, list):
        payloads = body
    else:
        payloads = []

    if not payloads:
        return None
    return payloads[0]


def get_delivered_payloads(block_number: int) -> List[Tuple[str, Dict]]:
    """
    Query all relays for delivered payloads; return list of (relay_url, payload dict).
    """
    out: List[Tuple[str, Dict]] = []
    for relay in RELAY_URLS:
        payload = get_one_delivered_payload(relay, block_number)
        if payload is not None:
            out.append((relay, payload))
    return out


def fetch_winning_bid_trace(block_hash: str, block_number: int) -> Dict:
    delivered = get_delivered_payloads(block_number)
    if not delivered:
        raise RuntimeError(f"No payload delivered for block {block_number}")

    candidates = []
    for relay, pl in delivered:
        ph = pl.get("block_hash")
        if not ph:
            continue

        url = relay.rstrip("/") + "/relay/v1/data/bidtraces/builder_blocks_received"
        try:
            resp = requests.get(url, params={"block_hash": ph}, timeout=5.0)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  BuilderBlockReceived failed on {relay}: {e}")
            continue

        body = resp.json()
        bid_data = body.get("Ok") if isinstance(body, dict) and "Ok" in body else body
        if isinstance(bid_data, list):
            bid_data = bid_data[0] if bid_data else None
        if not bid_data:
            continue

        ts = bid_data.get("timestamp_ms") or bid_data.get("timestamp")
        if ts is None:
            continue
        bid_data["timestamp_ms"] = int(ts)

        # Convert bid value from wei to eth
        value_wei = int(bid_data.get("value", 0))
        bid_data["value"] = Web3.from_wei(value_wei, "ether")

        candidates.append(bid_data)

    if not candidates:
        raise RuntimeError(f"No winning bid trace for block {block_number}")

    # Earliest timestamp wins
    return min(candidates, key=lambda b: b["timestamp_ms"])
