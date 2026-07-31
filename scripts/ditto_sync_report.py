"""Measure the real lip-sync offset from a DITTO_SYNC_CSV capture.

Run the pod with

    DITTO_SYNC_CSV=/workspace/ditto_sync.csv

speak a few sentences, then

    python scripts/ditto_sync_report.py /workspace/ditto_sync.csv

Each row is one shown frame: how loud the audio bound to it was, and how open
the mouth in it was. Cross-correlating the two columns says how many 40ms
frames the mouth leads (+) or lags (-) the sound. 0 means aligned.
"""

import csv
import sys


def load(path):
    audio, mouth = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            audio.append(float(row["audio_rms"]))
            mouth.append(float(row["mouth_open"]))
    return audio, mouth


def zscore(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = var ** 0.5 or 1.0
    return [(x - mean) / sd for x in xs]


def correlate(a, b, lag):
    """Correlation of mouth shifted by `lag` frames against audio."""
    pairs = [(a[i], b[i + lag]) for i in range(len(a))
             if 0 <= i + lag < len(b)]
    if len(pairs) < 10:
        return 0.0
    return sum(x * y for x, y in pairs) / len(pairs)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ditto_sync.csv"
    audio, mouth = load(path)
    if len(audio) < 50:
        sys.exit(f"{path}: only {len(audio)} frames, speak longer before reporting")

    # Silence rows carry no timing information and would swamp the correlation.
    speaking = [i for i, v in enumerate(audio) if v > 0.005]
    if len(speaking) < 25:
        sys.exit(f"{path}: only {len(speaking)} frames had audible speech")

    a, m = zscore(audio), zscore(mouth)
    scores = {lag: correlate(a, m, lag) for lag in range(-12, 13)}
    best = max(scores, key=scores.get)

    print(f"{len(audio)} frames, {len(speaking)} with speech\n")
    for lag in range(-6, 7):
        bar = "#" * max(0, int(scores[lag] * 40))
        print(f"  lag {lag:+3d} ({lag * 40:+5d}ms)  {scores[lag]:+.3f}  {bar}")
    print(f"\nbest offset: {best} frames = {best * 40}ms", end="  ")
    if best == 0:
        print("→ aligned")
    elif best > 0:
        print("→ mouth LAGS audio; nothing to add, the pairing slipped")
    else:
        print(f"→ mouth LEADS audio; try DITTO_AV_OFFSET_MS={-best * 40}")


if __name__ == "__main__":
    main()
