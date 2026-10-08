# Ye Chen ep1 - edit plan (the editorial layer; shots 1-3 are LOCKED as assembly V1)

Why this exists: three well-made clips merged did not feel like an episode (no sound carrying across cuts, brightness jumps, no motivated cuts,
shot 1 re-drew the set). Every remaining shot is now planned as part of a sequence: how it enters, how it leaves, what the sound does across the cut.

## Sequence (13 shots, about 70 s)
| # | Shot | Length | Enters by | Leaves by | Sound across the cut |
|---|---|---|---|---|---|
| 1-3 | LOCKED (V1: hall, whispering gallery, Saintess) | 16.25 s | - | hard cut | bell tail of shot 3 rings into shot 4 |
| 4 | teacup insert (plants Gu Changge, face hidden) | 4.1 s | hard cut, calm-to-calm | hard cut on the first drum beat | L-cut: shot 3's bell rings over the first 0.5 s |
| 5A | Ye Chen seen from behind, facing the dais (silent) | 4.1 s | J-cut: deep drum starts 0.3 s before the picture | match on action: fist tightens -> he draws a breath | drum + wind continue into 5B |
| 5B | Ye Chen shouts (face visible) | 5.4 s | match on action from 5A | cut on his last word | echo of the shout rings 0.5 s into shot 6 |
| 6 | the Elder roars | 6.1 s | reaction cut | cut on his last word | echo of the roar over the first 0.3 s of 7 |
| 7 | crowd reaction | 4.1 s | reaction cut | hard cut as shot 8's tone begins | J-cut: the deep tone starts 0.4 s early |
| 8 | the Archbishop's aura + line | 6.75 s | J-cut on the tone | cut on his last word | reverb tail over the start of 9 |
| 9 | Ye Chen answers under pressure | 5.4 s | reverse shot | eyeline match (he looks at the dais) | pressure drone continues into 10 |
| 10 | the Saintess (what he sees) | 4.1 s | eyeline match | hard cut to silence | bell fades; the crowd murmur stops dead = tension |
| 11 | the hall bows to Gu Changge | 6.75 s | smash cut from the murmur to a hush, then a deep tone swells | continuous push, then cut on the swell | tone continues under 12 |
| 12 | Gu Changge on the throne, narration | 6.75 s | cut on the tone swell | hold 1 s after the narration, then fade out | tone resolves with the narration |

## Sound architecture (one track under the whole scene, cuts happen on top of it)
- BED: one continuous hall ambience + music pulse under all 13 shots (OPEN DECISION below). Dialogue ducks it by 6 dB.
- Levels: dialogue RMS ~0.06, ambience ~0.03, music ~0.04; every shot level-matched before mixing (V1 method).
- Model audio is used for dialogue lines and distinct sound effects (bell, drum, chime, tone). Anything the model renders as near-silence ("whispers") comes from the bed instead.

## Picture continuity
- One colour grade for the scene (V1 method: 60 % toward the scene average per channel) + the same sunbeam direction (from the left) in every frame.
- Never use 3-step refinement (it re-draws costumes/props). Dense wide shots: 800x448 1-step by default, 2-step if needed (shot 11 may need it).
- Gaze/screen direction follows the staging map in gemini_prompts.md: south-camera shots see the dais-side faces, north-camera shots see Ye Chen's face and Gu Changge's throne.

## Continuity contracts (what each shot must leave for the next)
- 3 -> 4: calm; bell still ringing. 4 -> 5A: silence, then the drum. 5A -> 5B: a clenched fist / a breath.
- 5B -> 6: the end of a shouted line (echo). 6 -> 7 -> 8: escalation then a held breath; 8 ends with the glow on.
- 9 -> 10: Ye Chen looks toward the dais. 10 -> 11: total silence. 11 -> 12: Gu Changge centered, tone swelling.

## Per-shot checks BEFORE showing a shot to the director (will be automated)
1. Frame 0 matches the start image. 2. Everyone present in frame 0 is still in place at the end (hold check). 3. Motion median inside the card's band, no sudden jumps.
4. Audio finite, average level >= 0.02, centroid not a pure low rumble unless intended. 5. Dialogue shots: ASR transcript vs the exact line, speech onset time.
6. Contact sheet + the 3 numbers above reported as facts; the director judges motion and sound by watching.

## OPEN DECISIONS (need the director)
1. Sound bed: supply a royalty-free music/ambience track, or let me test generating ambience with LTX (untested, may sound poor).
2. Title card at the start / end card? (can be added in the finishing pass)
3. Order of shots 2 and 3: currently hall -> gallery -> Saintess (kept as V1). Hall -> Saintess -> gallery would follow establishing -> subject -> reaction.
