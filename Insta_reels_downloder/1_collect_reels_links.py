from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import time
import random

# --------------------------
# 1. Path to your existing Firefox profile
# --------------------------
profile_path = "/home/ranjit/.mozilla/firefox/luddqo6a.default-release"

# --------------------------
# 2. Firefox options
# --------------------------
options = Options()
options.profile = profile_path

# --------------------------
# 3. Initialize driver
# --------------------------
driver = webdriver.Firefox(options=options)

# --------------------------
# 4. Open target reels page
# --------------------------
target_url = "https://www.instagram.com/life_laps_official/reels/"
driver.get(target_url)
time.sleep(5)  # wait for page to load

# --------------------------
# 5. Scroll until all reels are loaded
# --------------------------
SCROLL_PAUSE_TIME = 1.5
MAX_NO_NEW_SCROLLS = 3  # stop if no new reels appear after 3 scrolls

reel_links_set = set()
no_new_count = 0

while True:
    # Find all reel links on the page
    reel_elements = driver.find_elements(By.TAG_NAME, "a")
    current_links = set(
        elem.get_attribute("href")
        for elem in reel_elements
        if elem.get_attribute("href") and "/reel/" in elem.get_attribute("href")
    )

    # Check if new links were found
    new_links = current_links - reel_links_set
    if new_links:
        reel_links_set.update(new_links)
        no_new_count = 0  # reset counter if new reels found
    else:
        no_new_count += 1

    # Stop if no new reels appear after several scrolls
    if no_new_count >= MAX_NO_NEW_SCROLLS:
        break

    # Scroll down
    driver.execute_script("window.scrollBy(0, 1000);")
    time.sleep(SCROLL_PAUSE_TIME + random.uniform(0, 1))  # random pause to mimic human

# --------------------------
# 6. Output
# --------------------------
reel_links = list(reel_links_set)
print(f"Found {len(reel_links)} reels:")
for link in reel_links:
    print(link)

# Optional: save to file
with open("Insta_reels_downloder/reel_links.txt", "w") as f:
    for link in reel_links:
        f.write(link + "\n")

# --------------------------
# 7. Close browser
# --------------------------
driver.quit()
