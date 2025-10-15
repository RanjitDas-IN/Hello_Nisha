import yt_dlp
import os

# --------------------------
# Paths
# --------------------------
links_file = "Insta_reels_downloder/reel_links.txt"
output_dir = "Insta_reels_audio/m4a_format"

os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Load all reel links
# --------------------------
with open(links_file, "r") as f:
    reel_links = [line.strip() for line in f.readlines() if line.strip()]

# --------------------------
# yt-dlp options
# --------------------------
ydl_opts = {
    "format": "bestaudio",
    "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
    "quiet": False,
    "noplaylist": True,
    "ignoreerrors": True,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",   # convert to m4a if possible
        }
    ],
}

# --------------------------
# Download all audios
# --------------------------
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for idx, url in enumerate(reel_links, 1):
        print(f"[{idx}/{len(reel_links)}] Downloading: {url}")
        try:
            ydl.download([url])
        except Exception as e:
            print(f"Error downloading {url}: {e}")

print("\n✅ All audio downloads completed.")
