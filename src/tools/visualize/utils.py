import os


def get_next_filename(folder, prefix, ext=".png"):

    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.startswith(prefix) and f.endswith(ext)
    ]

    if not existing:
        idx = 1
    else:
        nums = []

        for f in existing:
            try:
                n = int(f.split("_")[1].split(".")[0])
                nums.append(n)
            except:
                pass

        idx = max(nums) + 1 if nums else 1

    return os.path.join(folder, f"{prefix}_{idx:03d}{ext}")
