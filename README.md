# Starlink Constellation – 24/7 Live Simulation

Software-rendered (numpy + Pillow) simulation of ~10,000 Starlink satellites
orbiting a textured Earth with the Moon and a live HUD. Frames are piped into
ffmpeg and streamed over RTMP — no GPU or browser needed, so it runs on a
standard GitHub Actions runner.

## Setup
1. Create a new GitHub repo and upload everything in this folder
   (keep `assets/` and `.github/workflows/`).
2. YouTube Studio -> Go Live -> Stream -> copy the Stream URL + Stream key.
3. Repo -> Settings -> Secrets and variables -> Actions -> New secret:
   `YOUTUBE_RTMP_URL` = `rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY`
4. Actions tab -> "Starlink Live Stream" -> Run workflow.
   It then restarts itself every 5 hours via cron.

   Optional: add secret `AUDIO_URL` (direct .mp3 link, looped) for background music.
   Without it the stream carries silent audio.

## Thumbnail
`python starlink_sim.py --thumbnail thumbnail.png` (a ready one is included).

## What's on screen
Highlighted satellite (changes every 12 s), rotating "Did you know?" facts,
Moon drifting on its orbit, live UTC clock and satellites-in-view count.

## Local test
```bash
pip install numpy pillow
python starlink_sim.py --test-frame frame.png          # one PNG
python starlink_sim.py --frames 150 --out test.mp4     # 5 sec clip
```

## Tuning (env vars)
| Var | Default | Meaning |
|---|---|---|
| `FPS` | 30 | Use 24 if the runner can't keep up |
| `TIME_SCALE` | 45 | Simulation speed (x real time) |
| `X264_PRESET` | veryfast | Use `ultrafast` if CPU-limited |

Constellation shells (sum = 10,000): 53° 72x80, 43° 36x50, 70° 36x40, 97.6° 20x50.
Orbit radii are visually exaggerated so the satellites are visible around the globe.
Textures: three.js example planet textures (MIT).
