"""
Generate a realistic *synthetic* music dataset (Spotify-style audio features).

Each track has audio features drawn from genre-specific distributions, plus a
`popularity` score (0-100) produced by a hidden formula + noise, and a binary
`hit` label (popularity >= 70). The data is simulated — no real songs — but the
feature relationships are designed to be believable so the analysis is meaningful.

Run:  python generate_dataset.py   ->   songs.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# genre -> (mean feature profile, base popularity bias, n tracks)
# features: danceability, energy, valence, acousticness, speechiness, tempo(bpm), loudness(dB), duration(min)
GENRES = {
    "Pop":      dict(dance=0.72, energy=0.68, val=0.55, acou=0.15, speech=0.07, tempo=118, loud=-5.0, dur=3.4, bias=12, n=320),
    "Hip-Hop":  dict(dance=0.80, energy=0.64, val=0.50, acou=0.12, speech=0.22, tempo=100, loud=-6.0, dur=3.2, bias=10, n=300),
    "EDM":      dict(dance=0.70, energy=0.86, val=0.46, acou=0.05, speech=0.06, tempo=128, loud=-4.0, dur=3.9, bias=6,  n=240),
    "Rock":     dict(dance=0.52, energy=0.80, val=0.52, acou=0.10, speech=0.05, tempo=122, loud=-5.5, dur=4.0, bias=4,  n=240),
    "R&B":      dict(dance=0.66, energy=0.55, val=0.47, acou=0.25, speech=0.10, tempo=100, loud=-7.0, dur=3.7, bias=3,  n=200),
    "Acoustic": dict(dance=0.48, energy=0.33, val=0.44, acou=0.82, speech=0.04, tempo=110, loud=-10.5, dur=3.6, bias=-6, n=200),
}


def clip01(x):
    return np.clip(x, 0.02, 0.99)


def make():
    rows = []
    for genre, g in GENRES.items():
        n = g["n"]
        dance = clip01(RNG.normal(g["dance"], 0.11, n))
        energy = clip01(RNG.normal(g["energy"], 0.12, n))
        val = clip01(RNG.normal(g["val"], 0.16, n))
        acou = clip01(RNG.normal(g["acou"], 0.12, n))
        speech = clip01(RNG.normal(g["speech"], 0.05, n))
        tempo = np.clip(RNG.normal(g["tempo"], 12, n), 60, 200)
        loud = np.clip(RNG.normal(g["loud"], 2.0, n), -20, 0)
        dur = np.clip(RNG.normal(g["dur"], 0.6, n), 1.5, 7.0)

        # hidden popularity formula (then scaled): catchy = danceable + energetic + happy + loud,
        # penalised by heavy acousticness, lots of talking, and very long tracks.
        latent = (28 * dance + 22 * energy + 14 * val
                  - 16 * acou - 18 * speech
                  + 1.6 * (loud + 8)           # louder (closer to 0) -> more
                  - 3.0 * np.abs(dur - 3.4)    # ~3.4 min sweet spot
                  + g["bias"]
                  + RNG.normal(0, 9, n))        # randomness — taste isn't a formula
        for i in range(n):
            rows.append({
                "genre": genre,
                "danceability": round(float(dance[i]), 3),
                "energy": round(float(energy[i]), 3),
                "valence": round(float(val[i]), 3),
                "acousticness": round(float(acou[i]), 3),
                "speechiness": round(float(speech[i]), 3),
                "tempo": round(float(tempo[i]), 1),
                "loudness": round(float(loud[i]), 2),
                "duration_min": round(float(dur[i]), 2),
                "_latent": float(latent[i]),
            })

    df = pd.DataFrame(rows)
    # scale latent -> 0..100 popularity
    lo, hi = df["_latent"].quantile(0.02), df["_latent"].quantile(0.98)
    df["popularity"] = np.clip((df["_latent"] - lo) / (hi - lo) * 100, 0, 100).round(1)
    df["hit"] = (df["popularity"] >= 70).astype(int)
    df = df.drop(columns="_latent").sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "track_id", [f"T{1000+i}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = make()
    df.to_csv("songs.csv", index=False)
    print(f"Wrote songs.csv  ({len(df)} tracks, {df['hit'].mean()*100:.1f}% hits)")
    print(df.head())
