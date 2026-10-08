# Ye Chen ep1 - BEAT SHEET v2 (the real adaptation)

Status: PLAN for the director's approval. Nothing new is rendered until you approve this and send the new start images.

## 0. Why v1 failed (honest diagnosis, so v2 does not repeat it)
1. v1 was sized to the render budget (12 shots / 70 s), so the story was cut to what fit, and the first things cut were the reaction shots that carry the dread.
2. The chapter's engine is FEAR OF GU CHANGGE. v1 only showed him in the last two shots. Everything before it never pointed at him, so the reveal had nothing to pay off.
3. Shots were planned one by one. There was no anticipation anywhere: shot 8 (the Archbishop) cut in with his speech starting at 0.0 s; shots 6 and 9 the same. The shots that felt natural (5B and 12) started speaking at 0.7-0.8 s. A real anime gives a beat to LOOK first.
4. I trusted automatic checks that cannot judge faces. Shot 11 "passed"; Gu Changge's face there is ~25 px tall at the working size and smeared.
5. My selection rule preferred the take with better AUDIO numbers over the one with the better PICTURE (shot 10 a1 beat a0). Audio is fixed in the sound pass; the picture is not.

## 1. What the chapter is really about
Everyone in the hall is afraid of a man who has not moved or spoken. Ye Chen calls him "the devil" without knowing he is in the room; the disciples whisper "the young master"; the Archbishop stammers at the word "devil". The audience should feel a presence behind every frame BEFORE it is shown.
How Gu Changge's aura is built, in order (each is a shot or a sound, never a caption):
1. He is planted without a face (a hand holding a teacup that does not ripple).
2. People will not look at him: a disciple glances back, flinches, drops his eyes.
3. Space obeys him: an empty circle of floor around his throne in a packed hall.
4. His NAME makes people freeze (the Direct Disciple's line and the faces that freeze after it).
5. The most powerful man in the room (the Archbishop) stammers and glances toward him.
6. Silence: the hall goes quiet, then one small sound, the click of a teacup.
7. The crowd turns as one and bows. The reveal is the CROWD'S reaction first, his face second.
8. He never reacts to anything. He only lifts his eyes once, at the very end, and looks at us.

## 2. Rules for this scene (carried into every prompt)
- Every spoken line is preceded by a 0.8-1.2 s beat of silent presence (the speaker looks, breathes). QC now checks the speech onset (must be 0.7-1.8 s).
- Talking heads: upright, SAME SIZE in frame, camera locked off, "his head and shoulders do not move toward the camera" (a lean/zoom makes the face morph; that is what ruined shots 6, 8 and 10).
- "Hold" shots (seated crowds): everyone stays seated and in frame, camera still (shot 7 and 10 a1 failed because people got up and left after ~2 s).
- No face a viewer must read is smaller than ~25% of the frame height (at the working size a smaller face is just smear). Wide shots with a tiny Gu are silhouettes by design (backlight, shadow), never faces.
- Concrete audible sound words only ("a clear temple bell"), never "soft/faint/low/hush".
- Every shot is checked by me frame by frame (face crops) before it is marked done.

## 3. The sequence (31 shots, ~2 min 22 s; v1 was 13 shots / 69 s)
Status key: REUSE = existing clip used as is or trimmed (no render time). NEW render from an existing image = no image work for you. NEW image needed = you generate a start frame (prompts in gemini_prompts_v2.md). Tier A = the story needs it; Tier B = polish, cut first.

| # | Beat | Source | Length | Tier | Dialogue |
|---|---|---|---|---|---|
| 01 | Grand hall, the ceremony is solemn (establishing) | REUSE as is - old shot 1 (2-step) | 6.8 s | A | - |
| 02 | Seniors whisper in the gallery (everyone is watching) | REUSE as is - old shot 2 (a1) | 4.1 s | A | - |
| 03 | The Saintess, serene, about to be given away | REUSE as is - old shot 3 | 5.4 s | A | - |
| 04 | PLANT 1: a hand with a teacup, face hidden, tea perfectly still | REUSE as is - old shot 4 | 4.1 s | A | - |
| 05 | PLANT 2: a young disciple glances back over his shoulder, flinches, snaps his eyes down, sleeves tremble | NEW image needed (N1) | 4.1 s | A | - |
| 06 | PLANT 3: the far end of the hall - a dark throne in backlight, a shape on it, and an EMPTY CIRCLE of floor nobody steps into | NEW image needed (N2) | 5.4 s | A | - |
| 07 | Ye Chen from behind, facing the dais (the room holds its breath) | REUSE as is - old shot 5A | 4.1 s | A | - |
| 08 | The whole crowd's heads whip toward the aisle (the gasp) | NEW image needed (N3) | 4.1 s | A | - |
| 09 | Ye Chen, line 1 (after a 1 s stare) | NEW render, existing image - old 5B image | 5.4 s | A | So this is the Supreme Stygian Holy Land? Oppressive. Tyrannical. |
| 10 | Ye Chen, line 2 (your chosen take) | REUSE trimmed - old 5B attempt 1, cut at 4.7 s | 4.7 s | A | You would even give your own daughter up to the devil! |
| 11 | The Saintess hears it (her reaction) | REUSE as is - old shot 10 attempt 0 (the take you liked) | 4.1 s | A | - |
| 12 | Ye Chen, line 3: 'You never asked if she was willing' | NEW render, existing image - old 5B image | 6.1 s | A | You never even asked the Saintess if she was willing! I will do her justice today! |
| 13 | The Elder: silent glare building | REUSE trimmed - old shot 6 attempt 0, first 1.7 s only | 1.7 s | B | - |
| 14 | The Elder roars (speech after a 1 s hold) | NEW render, existing image - old shot 6 image | 5.4 s | A | Blasphemous! How dare you, a puny disciple, go against the Archbishop! |
| 15 | Gossip: two disciples (the crowd thinks he is a madman) | NEW image needed (N5) | 4.1 s | B | Is Ye Chen mad? Charging in here and saying such things? |
| 16 | THE KEY LINE: the Direct Disciple, frightened, whispers the name | NEW image needed (D1) | 6.1 s | A | If he angers the young master, he could bring down the entire Holy Land. |
| 17 | At 'young master' the faces around him FREEZE: one swallows, sweat, eyes dart toward the back | NEW image needed (N6) | 4.1 s | A | - |
| 18 | Cut to the throne again: the shape has not moved, steam rises, the empty circle | NEW render, existing image - N2 image (new prompt) | 4.1 s | A | - |
| 19 | HUSH: the whole hall goes still, dust hangs in the sunbeams | NEW render, existing image - old shot 1 image | 4.1 s | B | - |
| 20 | The Archbishop, eyes closed, perfectly still ... eyes open ... he looks at us for 2 s. No words. | NEW image needed (A1) | 5.4 s | A | - |
| 21 | His aura floods the hall and he speaks | REUSE as is - old shot 8 (speech starts at 0.0 s there, so shot 20 supplies the pause) | 6.8 s | A | Ye Chen, is it? I remember you. You came to my Holy Land seeking refuge. |
| 22 | Ye Chen under the pressure | REUSE as is - old shot 9 attempt 0 (passed) | 5.4 s | A | I am simply requesting justice for the Saintess. |
| 23 | The Archbishop stammers 'T-To the devil?' and his eyes FLICK toward the back of the hall | NEW image needed (A2) | 4.1 s | A | T-To the devil? How slanderous! |
| 24 | The crowd misreads it: sneers ('he is only jealous') | REUSE trimmed - old shot 7 attempt 0, first 1.8 s only | 1.8 s | B | - |
| 25 | THE CLICK: the cup is set down on the lacquer, tea ripples; the sound rings through a hall that has stopped breathing | NEW render, existing image - old shot 4 image | 4.1 s | A | - |
| 26 | Ye Chen notices the silence; his eyes slide toward the back | NEW render, existing image - old 5B image | 4.1 s | A | - |
| 27 | The crowd turns as one and bows; the aisle opens | NEW image needed (R3) | 5.4 s | A | - |
| 28 | REVEAL: Gu Changge on the dark throne, big enough to read his face, the whole hall bowing to him | NEW image needed (R4) | 6.8 s | A | - |
| 29 | Gu Changge sips his tea, utterly indifferent | REUSE trimmed - old shot 12 attempt 0, first 2.7 s only | 2.7 s | A | - |
| 30 | He lifts his eyes and looks straight at us. One look. Narration over it. | NEW image needed (R5) | 6.1 s | A | (narrator, mouth closed) And that young man was none other than Gu Changge. |
| 31 | Cut to black, title card (post-production, no render) | REUSE as is - made in post | 2.0 s | A | - |

### Act structure
- ACT 1 (01-06, ~30 s): the ceremony, three plants of Gu Changge (no face).
- ACT 2 (07-18, ~55 s): Ye Chen breaks the silence; the Elder; the crowd; the Direct Disciple says "the young master"; faces freeze; back to the throne.
- ACT 3 (19-23, ~28 s): the Archbishop, with a hush and a silent stare before he speaks; the stammer at "devil" and the glance.
- ACT 4 (24-31, ~38 s): the misreading, the click, the silence, the crowd turns and bows, the reveal, the stare, the title.

## 4. Cuts and sound (the editing layer)
- Cut on motivated reasons only: eyeline (a glance back -> the throne), sound (the click -> the hall turns), reaction to a line. No two consecutive shots of the same character at the same size.
- J-cuts: Ye Chen's first words start 0.4 s before the picture cuts to him (07->09). L-cuts: the echo of each shout rings over the next shot.
- SILENCE is used on purpose: before the Archbishop's eyes open (19-20) and between the click and the turn (25-27). The model's per-shot sound is used for dialogue lines and distinct effects; everything it makes as "ambience" is replaced.
- Sound bed (OPEN DECISION): a continuous low hall ambience + slow string/choir pad under the whole scene, ducked under dialogue. Options: (a) you provide a royalty-free track; (b) I synthesize an ambient bed (drone, pad, bell hits, silence cuts) from code; it will sound synthetic but controlled; (c) both: yours for music, mine for the tension drops.

## 5. What is reused from v1 and what is NOT usable
| Old shot | Use in v2 | Good part | Not used because |
|---|---|---|---|
| 1 | 01 as is | all (2-step version) | - |
| 2 | 02 as is | all | - |
| 3 | 03 as is | all | - |
| 4 | 04 as is | all; its image also reused for 25 (the click) | - |
| 5A | 07 as is | all | - |
| 5B attempt 1 | 10 (cut at 4.7 s) | 0-4.7 s | after 4.7 s the face zooms in and morphs |
| 6 attempt 0 | 13 (first 1.7 s) | silent glare | from ~2 s the Elder leans toward the lens and his mouth/face morph |
| 7 attempt 0 | 24 (first 1.8 s) | sneering whisper | after ~2 s they stand up and leave |
| 8 attempt 0 | 21 as is | the aura flood, line (similarity 0.9) | starts speaking at 0.0 s; shot 20 now supplies the pause. The white-out between 2.7 and 5.3 s is heavy (may need a grade). |
| 9 attempt 0 | 22 as is | stable, passed | - |
| 10 attempt 0 | 11 | your pick | - |
| 11 | NOT used | - | Gu's face is ~25 px, smeared; crowd blocks the frame. Replaced by 27 + 28 (bigger Gu). |
| 12 attempt 0 | 29 (first 2.7 s) | the sip, face on-model | from ~4 s the face zooms in, eyes closed, off-model; also he must LOOK at us at the end, which this take never does (30 handles it). |

## 6. Cost (be honest about it)
- Total length 142 s in 31 shots. Reused existing clips: 13 clips = 54 s (no render time).
- NEW renders: 18 clips, of which 11 need a new start image from you; the others reuse an image you already made.
- Render time for the new clips (1-step at 800x448; 2-step marked): about 280 minutes = 4.7 hours (Tier A 4.2 h, Tier B 0.5 h), plus retries and any shot you reject. That is on top of what v1 cost (about 6 hours).
- Per-shot cost: 4 s = 12 min, 5.4 s = 15 min, 6 s = 17 min, 6.75 s = 18 min; +7 min for the 2-step dense shots (06, 19, 27, 28).
- Your effort: 11 start images (list and prompts in gemini_prompts_v2.md).
Order of work if you approve: (1) you send the 11 images (I check each one against its shot: size, gaze direction, face size); (2) I render the Tier A shots one by one with the new QC (speech onset, face crops, hold); (3) I show you each hard shot before moving on (05, 06, 16, 20, 23, 28, 30); (4) assemble, sound pass, title.

## 7. Tooling fixes I will make BEFORE rendering (so v1's mistakes cannot repeat)
1. Selection rule: choose the attempt with the best PICTURE; audio problems never override a better picture (fixed later in the sound pass).
2. QC adds: speech onset must be 0.7-1.8 s; face-size growth check (head scale vs frame 0); I produce face-crop contact sheets and look at them for every dialogue shot and every Gu shot.
3. Assembler: J-cuts / L-cuts (audio offset per shot), trim points per shot (already done for end trims; adding start trims), title card.
4. Prompts: the dialogue template becomes "For the first second he is silent and still, looking [where]. Then he says ... He stays upright and the same size in the frame; his head and shoulders do not move toward the camera. The camera is locked off."

## 8. Open decisions for the director
1. Scope: approve ~2:22 (31 shots) or tell me which Tier B shots to drop (13, 15, 19, 24 are the first candidates; saves ~31 min).
2. Sound bed: your track, my synthesized bed, or both.
3. Lines: I shortened Ye Chen's long speech into three lines of ~10 words so each shot stays short and the face does not drift. Tell me if you want his exact original words (the shots become 8 s and riskier).
4. Title/end card text.
