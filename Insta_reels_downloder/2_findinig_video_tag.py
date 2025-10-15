from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import time

# --------------------------
# Firefox profile
# --------------------------
profile_path = "/home/ranjit/.mozilla/firefox/luddqo6a.default-release"
options = Options()
options.profile = profile_path
driver = webdriver.Firefox(options=options)

# --------------------------
# Load all reel links
# --------------------------
with open("Insta_reels_downloder/reel_links.txt", "r") as f:
    reel_links = [line.strip() for line in f.readlines() if line.strip()]

video_links = []

# --------------------------
# Loop through each reel
# --------------------------
for idx, reel_url in enumerate(reel_links, 1):
    print(f"[{idx}/{len(reel_links)}] Processing: {reel_url}")
    driver.get(reel_url)
    time.sleep(5)  # wait for page to load

    video_url = None
    try:
        video_element = driver.find_element(By.TAG_NAME, "video")
        video_url = video_element.get_attribute("src")
    except:
        pass

    if video_url:
        print(f"Video URL found: {video_url}")
        video_links.append(video_url)
    else:
        print("Video tag not found.")

# --------------------------
# Save all collected video URLs
# --------------------------
with open("Insta_reels_downloder/video_elements.txt", "w") as f:
    for url in video_links:
        f.write(url + "\n")

print(f"Saved {len(video_links)} video URLs to video_elements.txt")

driver.quit()
