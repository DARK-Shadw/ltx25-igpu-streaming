# Ye Chen ep1 v2 - saved state (2026-10-09 ~00:00)
Render queue stopped by the director; nothing is running.

## Done
- 17 of 18 new shots rendered with QC + face strips (episodes/ye-chen-ep1-v2/qc/). Shot 27: take 1 done, take 2 stopped mid-render (re-run `python episode_run_v2.py ye-chen-ep1-v2 27`).
- Draft cut: videos/episodes/ye-chen-ep1-v2/assembly/draft1.mp4 (2:15, 29 clips + black; NO shot 27 yet; cutlist next to it). Rebuild with `python assemble_draft.py draft1` (picks/trims/delays/mutes are in the PLAN list in assemble_draft.py).

## My picks (director has NOT confirmed): {"30": "take 2 (a1)", "28": "take 1 (a0)", "16": "take 1 (a0)", "20": "take 2 (a1)", "23": "take 1 (a0)", "09": "take 2 (a1)", "12": "take 1 (a0)", "14": "take 2 (a1)", "15": "take 2 (a1)", "17": "take 2 (a1)", "26": "take 2 (a1)", "25": "take 1 (a0)", "27": "take 1 (a0)"}
Old-v1 clips reused: 01(2-step) 02(a1) 03 04 05A 05B(a1 cut 4.7 s) 06(a0 first 1.7 s, muted) 07(a0 first 1.8 s) 08(a0) 09(a0) 10(a0) 12(a0 first 2.7 s, muted).

## Open items
1. Shot 08 (crowd turns): the front rows turn their backs; currently cut at 2.2 s. Candidate for a re-render with the crowd facing the dais.
2. Shot 27 take 2 + add 27 to the draft.
3. Sound pass: model audio is mostly low rumble/near-silence; needs an ambience/music bed (director's track or synthesized), ducking under dialogue, J/L-cuts, title card text.
4. Director's notes after watching draft1 (which takes to swap, what to cut/redo).
5. Spoken-line start times: several takes speak at 0.0 s (delays applied in assemble_draft: shot 09 +0.8 s, 21 +0.6 s, 30 +1.2 s).

## Files
beat sheet: episodes/ye-chen-ep1/beat_sheet_v2.md | image prompts: gemini_prompts_v2.md | start images: episodes/ye-chen-ep1/frames_v2/ (R3 = the new standing-crowd version; R3_old_kneeling.png kept)
tooling: shot_run.py shot_qc.py (zoom/speech-onset/face strips) episode_run_v2.py assemble_draft.py stream/stitch.py (trim, audio delay, gain, frame_fn) | rules: episodes/PLAYBOOK.md
