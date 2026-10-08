# Episode production playbook (written from the Ye Chen ep1 mistakes, so the next chapter can run with far less hand-holding)

## Pipeline
1. Adapt: read the chapter, keep the beats that carry story, cut the rest; one scene = ~12-14 shots, ~60-70 s at our speed (render time ~0.22 min per generated frame at 800x448 -> 1600x896 FLASH).
2. Staging map FIRST (top view): where every character stands and which way they face; define camera families (which end of the room the camera is at). Only then write shots.
3. Design sheets (GPT-style painterly): one per character + one empty-hall sheet. Pick by style; ask Gemini/GPT for a SIMPLIFIED design if it is too fussy (bracers, tassels, fine embroidery smear in video).
4. Start frames: one per shot, 16:9, written with the dialogue/framing rules below. Review each against its shot card before any video.
5. Shot cards (`shots.json`): frame analysis, motion, camera, audio, dialogue, duration (8k+1 frames), full video prompt, QC thresholds, status.
6. Video: one shot at a time. Preview/QC -> user approval -> next. Never batch-render un-reviewed shots.
7. Finish: stitch with the drift-free audio stitcher, grade, title, music bed.

## Rules learned (each one cost us a redo)
- Paste prompts with the style text WRITTEN OUT. A bare "STYLE." placeholder was pasted literally once.
- Characters face what they react to. Decide facing from the staging map; never default to "looking at the viewer" except in a deliberate close-up gaze.
- A face only exists in the video if it is visible in frame 0. Hidden faces get invented and drift off-model. Backs/profiles only in silent shots.
- Talking shots: near-frontal speaker (<=20-30 degrees), mouth visible, head >= 1/4 frame height, one speaker per clip, lines of 2-3 s, mouth CLOSED in frame 0.
- Never write "to the left of the lens", "stirring", "low angle" in talking close-ups: they trigger a tilted key-art layout (subject on the left third, giant foreground hand). Use "static, calm, symmetrical, plain portrait-style framing, no hands visible".
- Hands: hide them or keep them simple; foreground hands become distorted blobs.
- Aura = elevation + scale + isolation + stillness among motion + one shaft of light + a crowd reacting to the person. A tiny figure at a small table is not aura.
- Crowds: faces tiny or seen from behind; slow camera only.
- Duration by content: establishing 6-7 s, reactions/inserts 4 s, one spoken line 5-6 s, longer speech 7-8 s (render time scales with frames).
- Video prompt: one paragraph, present tense, style-keeping first sentence, subject motion, camera in its own clause, dialogue in quotes with a voice description, sound last. Slow deliberate motion only; "motion smear / speed lines / quick half-circle" caused collapse.
- Motion QC bands (median frame change, real frames): <8 static for ACTION shots, 8-38 good, >38 smear. For dialogue/establishing shots use the shot card band instead.
- Verify audio sync on the finished stitch with speech timestamps against shot start times; crossfades must never shorten the track.
- ASR (faster-whisper) hallucinates "Thanks for watching!" on music-only tails: ignore it.
- HOLD-POSE shots (people seated/standing and reacting): the prompt must say they stay where they are ("remain seated exactly as in the first frame, nobody stands up or walks away, all stay in frame"), and the camera must be almost static. A pan close to a crowd slides everyone out of frame and the model adds body motion to match; "glances toward X" becomes walking toward X (ep1 shot 02 attempt 0). Check: the people in frame 0 are still there and still in place at the last frame.
- AUDIO prompt words are taken literally: "low / faint / quiet / soft" produce near-silence (ep1 shot 02: rms 0.006 = about -44 dB vs 0.14 for a "deep ceremonial horn drone"), and "whispers/murmurs" did not render as voices at all (spectral centroid 115 Hz = a low rumble, not voices). Ask for concrete audible sounds ("clearly audible crowd murmur, rustling silk, echoing hall"), and plan a post pass: per-shot loudness normalization + a continuous ambience/music bed under the whole scene. QC every shot's audio: finite, rms >= ~0.02, and for speech shots an ASR transcript.

## Story-design rules (added after the v1 director review of ye-chen-ep1)
- DESIGN THE EPISODE BEFORE PRICING IT. Write the full beat sheet from the chapter's emotional engine first (here: the fear of Gu Changge), then show the director the shot count, length and render hours for the full version and for a trimmed version, and let them choose. Never shrink the story silently to fit the render budget.
- Find the chapter's engine (who or what is feared/wanted) and make EVERY act point at it before it is shown: plant (no face) -> avoidance -> empty space -> the name makes people freeze -> the powerful stammer -> silence + a small sound -> the crowd reacts -> only then the face -> the one look.
- Anticipation: every first appearance and every spoken line gets a 0.8-1.2 s beat of silent presence. v1's dialogue shots that spoke at 0.0 s felt rushed; the ones at 0.7-0.8 s felt natural. QC speech onset 0.7-1.8 s.
- Reaction shots carry dread. They are not optional polish: cut dialogue, not reactions.
- Select the attempt by PICTURE; fix audio in the sound pass. Look at face crops myself for every dialogue and key-character shot; automatic checks cannot judge faces.
- A face the viewer must read needs >= 25% of the frame height at the 800x448 working size. A wide shot with a tiny character is a silhouette by design.
