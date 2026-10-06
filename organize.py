import os, shutil, glob
R = "."
def mv(src, dst):
    if not os.path.exists(src):
        print(f"  (missing) {src}"); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print(f"  (exists, skipped) {dst}"); return
    shutil.move(src, dst); print(f"  {src}  ->  {dst}")

V = "videos"
videos = {
 # --- anime: swordsman-on-bridge series (same prompt/seed, different render modes) ---
 "anime4_5s.mp4":          f"{V}/anime/swordsman-bridge/1_reference-stage1-only_768x448_24fps_5s.mp4",
 "anime4_hd_5s.mp4":       f"{V}/anime/swordsman-bridge/2_reference-hd_1536x896_24fps_5s_47min.mp4",
 "anime4_fast1.mp4":       f"{V}/anime/swordsman-bridge/3_flash-2step_1536x896_12fpsx2_5.4s_18min.mp4",
 "anime4_fast1_s2one.mp4": f"{V}/anime/swordsman-bridge/4_FLASH_1536x896_12fpsx2_5.4s_15min_RECOMMENDED.mp4",
 "anime4_turbo.mp4":       f"{V}/anime/swordsman-bridge/5_turbo-preview_1280x704_12fpsx2_5.4s_9.5min.mp4",
 # --- anime: earlier tests ---
 "anime3_5s.mp4":          f"{V}/anime/early-tests/bamboo-forest-swordsman_768x448_24fps_5s.mp4",
 "anime2_3s.mp4":          f"{V}/anime/early-tests/ghibli-girl-yellow-raincoat_768x448_24fps_3s.mp4",
 "anime_3s.mp4":           f"{V}/anime/early-tests/cyberpunk-swordswoman-mecha-dragon_768x448_24fps_3s.mp4",
 # --- realistic ---
 "hd_4s.mp4":              f"{V}/realistic/hd-showcase/fox-snowy-forest_1280x704_24fps_4s_two-stage_20min.mp4",
 "dog_2stage.mp4":         f"{V}/realistic/tests/dog-beach_1024x640_2s_two-stage.mp4",
 "dog_3s.mp4":             f"{V}/realistic/tests/dog-beach_640x384_3s.mp4",
 "dog.mp4":                f"{V}/realistic/tests/dog-beach_384x256_1s_early-recipe.mp4",
 "fruit.mp4":              f"{V}/realistic/tests/fruit-bowl_384x256_1s_early-recipe.mp4",
 # --- failed / superseded (kept, not deleted) ---
 "out.mp4":                f"{V}/_failed-or-duplicate/first-run_black-video_NaN-bug.mp4",
 "out_nobos.mp4":          f"{V}/_failed-or-duplicate/prompt-ignored_old-man_missing-BOS-token.mp4",
 "out_1024.mp4":           f"{V}/_failed-or-duplicate/prompt-ignored_woman_early-text-bug.mp4",
 "anime3_5s_T.mp4":        f"{V}/_failed-or-duplicate/bamboo-forest-swordsman_byte-identical-duplicate.mp4",
}
print("== videos =="); [mv(a, b) for a, b in videos.items()]

print("== prompts =="); 
for a, b in {"anime4_prompt.txt": "prompts/anime_swordsman-bridge.txt", "anime3_prompt.txt": "prompts/anime_bamboo-forest-swordsman.txt",
             "anime2_prompt.txt": "prompts/anime_ghibli-girl-yellow-raincoat.txt"}.items(): mv(a, b)

print("== png frames ==");  [mv(p, f"frames/{p}") for p in sorted(glob.glob("*.png"))]
print("== latents ==");     [mv(p, f"latents/{p}") for p in sorted(glob.glob("*.pt") + glob.glob("*.pt.stage1"))]
print("== logs / timing notes ==")
for p in sorted(glob.glob("*.log") + glob.glob("*_start.txt")): mv(p, f"logs/{p}")
print("\nleft at top level:", sorted(os.listdir(R)))
