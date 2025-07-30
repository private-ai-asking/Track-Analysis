import os
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# Configure these to suit your setup:
# ─────────────────────────────────────────────────────────────────────────────
DIRECTORY   = input("Enter the directory to process: ")  # <-- change this to your folder
OUTPUT_DIR  = DIRECTORY                     # or set a different path here

VIDEO_EXT   = ".mp4"
AUDIO_EXT   = ".m4a"
# ─────────────────────────────────────────────────────────────────────────────

def list_files_by_extension(path: str, ext: str) -> dict[str, str]:
    """
    Returns a dict mapping each filename (without extension) to its full path,
    for all files in `path` ending with `ext`.
    """
    return {
        os.path.splitext(f)[0]: os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(ext)
    }

def extract_audio(input_path: str, output_path: str) -> None:
    """
    Runs ffmpeg to copy the audio stream only (no re-encoding).
    - '-vn'      disables video :contentReference[oaicite:0]{index=0}
    - '-acodec copy'  copies the audio codec as-is :contentReference[oaicite:1]{index=1}
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "copy",
        output_path
    ]
    try:
        print(f"Extracting: {os.path.basename(input_path)} → {os.path.basename(output_path)}")
        # using run(..., check=True) raises on error :contentReference[oaicite:2]{index=2}
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✔️  Done")
    except subprocess.CalledProcessError:
        print(f"❌  Failed: {input_path}")

def main():
    videos = list_files_by_extension(DIRECTORY, VIDEO_EXT)
    print(f"Found {len(videos)} '{VIDEO_EXT}' files in {DIRECTORY!r}.")

    for name, video_path in sorted(videos.items()):
        output_path = os.path.join(OUTPUT_DIR, name + AUDIO_EXT)
        if os.path.exists(output_path):
            print(f"✅  Skipping {output_path} (already exists)")
            continue
        extract_audio(video_path, output_path)

if __name__ == "__main__":
    main()
