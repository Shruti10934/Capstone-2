import os
import shutil
import urllib.request

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

FONTS = {
    "NotoSans-Regular.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    "NotoSans-Bold.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf",
    "NotoSerif-Regular.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSerifDevanagari/NotoSerifDevanagari-Regular.ttf",
    "NotoSerif-Bold.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSerifDevanagari/NotoSerifDevanagari-Bold.ttf",
    "Kalam-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf",
    "Kalam-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Bold.ttf",
    "Kalam-Light.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Light.ttf",
    "Dekko-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/dekko/Dekko-Regular.ttf",
    "RozhaOne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/rozhaone/RozhaOne-Regular.ttf",
    "YatraOne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/yatraone/YatraOne-Regular.ttf",
    "Sahitya-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sahitya/Sahitya-Regular.ttf",
    "Sahitya-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sahitya/Sahitya-Bold.ttf",
    "Eczar-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/eczar/Eczar%5Bwght%5D.ttf",
    "Rajdhani-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/rajdhani/Rajdhani-Regular.ttf",
    "Rajdhani-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/rajdhani/Rajdhani-Bold.ttf",
}


def setup_fonts():
    os.makedirs(FONT_DIR, exist_ok=True)

    # Download missing or invalid fonts
    for name, url in FONTS.items():
        path = os.path.join(FONT_DIR, name)

        if os.path.exists(path) and os.path.getsize(path) > 5000:
            continue

        try:
            urllib.request.urlretrieve(url, path)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[WARN] {name}: {e}")
            if os.path.exists(path):
                os.remove(path)

    # Copy Windows font if available
    win_font = r"C:\Windows\Fonts\Nirmala.ttc"
    if os.path.exists(win_font):
        shutil.copy2(win_font, FONT_DIR)

    # List available fonts
    fonts = [
        f for f in os.listdir(FONT_DIR) if f.lower().endswith((".ttf", ".otf", ".ttc"))
    ]

    print(f"Fonts ready: {len(fonts)}")
    return fonts


if __name__ == "__main__":
    setup_fonts()
