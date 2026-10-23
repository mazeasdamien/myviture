# Verdict du président (2026-10-23)

## Causes du tremblement et du traînage, classées
1. predict_ms = 12 jamais calibré (hypothèse ; la métrique est aveugle à cette erreur, vérifié myviture.cpp:943-958). 6,7 ms d'erreur = 1° à 150°/s.
2. Warp incohérent avec le lisseur (vérifié) : tw_q brut (:933) alors que l'image est rendue avec oq*q (:1014) ; dq = conj(tw_q)*late (:1410) réinjecte la correction que le lisseur absorbe (:997) → saut, retour, glissement.
3. Warp sauté sur les images de repli (vérifié) : engine_pose(target) échoue (:931) → tw_target = 0 → pas de warp (:1221).
4. Lissage = retard gardé (vérifié) : révisions < 3° absorbées, relâchées en 60 ms pendant les rotations (:1008).
5. Copie du maillage (900 000 × 24 o = 21,6 Mo) sous verrou sur le thread de rendu toutes les ~4 s (:1136-1145), puis dessiné chaque image avec l'occlusion (:1285) : vblanks ratés périodiques (à tester).
6. IMU sans horodatage par échantillon (state.hpp:90) : gigue possible de la cible (à tester).
7. Rémanence 60 Hz : ~2,5° d'étalement à 150°/s (chercher un mode 90/120 Hz).

## La pièce
- « 0 sommets » = artefact de log (:427 toutes les 2 s, maillage toutes les 4 s). Le maillage est tronqué à 900 000 sommets dans un ordre arbitraire (:265), 1,2-1,35 s par reconstruction.
- « 8 murs » = plafond de la boucle `wall < 8` (:155).
- Carte saturée à 400 000 voxels, nouvelles surfaces ignorées (:367) ; `seq % 3` rate des images (:315). Base stéréo réelle 136,1 mm.
- Représentation : sol (polygone, gravité IMU), murs (quads, angles droits), plateaux, grille 2D marchable ; TSDF creuse CPU 4-5 cm plus tard pour le vrac. Plans détectés image par image puis fusionnés.
- Affichage : rien par défaut sauf occlusion ; pendant le scan, grille légère au sol et lignes fines aux jonctions sol-mur. Jamais de voxels.
- Profondeur : census gardé, SGM 4 chemins si besoin ; pas de stéréo apprise sur l'iGPU ; mono RGB plus tard (720p max).

## Objectif final
Sol juste à ~1 cm (pieds de l'humain), marchable = sol moins obstacles 0,1-1,8 m, A* sur grille ; occlusion par plans + boîtes ; MCP de scène (planes, walkable, raycast, anchors, move_human_to) ; repère commun lunettes ↔ cellule Ned (ArUco ou 3 points) : angle mort de tout le conseil.

## Plan
1. Test chambranle + réglage predict_ms + A/B (défaut / lissage coupé / warp coupé) (0,5 j)
2. Warp cohérent avec la pose lissée, warp aussi sur les replis (0,5 j)
3. Assainir la carte : copie hors verrou, log, purge des voxels non confirmés, seq - seen >= 3, vblanks ratés par vitesse (1 j)
4. Plans robustes guidés par la gravité, sans plafond à 8 murs, vérifiés au mètre (2-3 j)
5. Grille marchable, humain qui marche, occlusion par plans (3 j)
6. Calibration repère robot + MCP de scène (2-3 j)
7. Plus tard : TSDF creuse, SGM, mono (3-5 j)

## Première chose
Ligne verticale fine sur un vrai chambranle à 2 m, secouer la tête à 1-2 Hz, ajuster predict_ms avec +/- jusqu'à ce qu'elle ne traîne ni ne devance ; trois fois (défaut, lissage coupé, warp coupé).
# Council dossier: myViture — virtual content lags/trembles in the glasses, and the room mesh/depth is sparse

## The product
Native Windows C++ / Direct3D 11 app (code: C:\Users\Damien Mazeas\Documents\myViture\native) for VITURE Luma Ultra
optical see-through (OST) AR glasses, tethered to a laptop. The developer wants virtual objects "ultra stable" when
walking and turning, a good room mesh (walls/floor/objects) for occlusion and anchoring, and reuses the VITURE SDK
wherever possible.

## Hardware (hard constraints)
- Laptop: Intel Core 5 220H, **Intel integrated GPU only** (no NVIDIA, so no CUDA/TensorRT/nvblox/InfiniTAM-GPU).
  D3D11 compute works; DirectML / OpenVINO on the iGPU are possible but untested here.
- Glasses: 2 grayscale fisheye tracking cameras 640x480 @ ~25-30 Hz (shared with VITURE's own VIO), one RGB camera
  1920x1080 MJPG @ 29 fps, IMU. Display: 3840x1200 side-by-side, 60 Hz, shown as a Windows monitor; our swap chain is
  in hardware overlay / independent flip (measured: composed 0, overlay ~120/2 s), so DWM adds no frame.
- USB bandwidth is fragile: uncompressed 1080p RGB once starved the tracking cameras (6DoF lost until replug).

## Tracking / rendering pipeline today
- Pose: VITURE engine (carina_vio.dll) `get_gl_pose(t)`, IMU-predicted; polled at ~515 Hz. Log line:
  "engine pose changes when polled at 1 kHz: latest 25 Hz, at device time 63 Hz" and "engine 21-30 ms".
- Render at 60 fps, CPU ~6 ms/frame. Prediction target = time to vblank + `predict_ms` (=12 ms, the assumed glasses
  display latency after vblank; user can change it with +/- keys but never did).
- Rotation-only timewarp: render to texture, wait until vblank - margin (3-9 ms adaptive), re-query engine pose at the
  same target time, warp with M = E^T * Delta * E.
- Tracking guard: freeze on the last good pose when the pose jumps >15 deg or >25 cm in a frame.
- Error metric (ONLY reference available, no photon sensor): the pose used vs the engine's own estimate of that same
  instant read 80 ms later. Measured during real head motion (p50/p95, degrees):
    - still: prediction 0.02-0.2 / 0.06-0.7
    - slow (<~100 deg/s): prediction 0.7-1.3 / 1.5-3.4 ; after timewarp 0.3-0.8 / 0.6-2.5
    - fast (>~100 deg/s, peaks 150-350 deg/s): prediction 1.6-2.9 / 2.7-6.4 ; after timewarp 1.1-3.0 / 2.1-7.5
  At 150 deg/s, 2-3 deg = 13-20 ms of uncompensated latency (about one display frame).
- Display calibration: VITURE factory eye poses (~9.5 deg tilt) + user FOV/IPD settings; never measured against the
  real world. Positional (translation) error and calibration error are not covered by the rotation timewarp.
- ~3.5/s "GetPoseByTimestamp failed due to imu preintegration" messages from the engine (cause unknown).
- Earlier: once the engine pose itself spun for ~12 s while walking (up to 324 deg/s, no engine error).

## What the user sees (latest report)
"The voxel mesh is ultra stable in the dashboard on the RGB video, but in the glasses there is lag and it trembles.
When I move my head fast, all the 3D struggles to follow." The dashboard RGB view is video see-through: virtual
drawn with the PAST pose at the RGB frame's capture time, so video and virtual share the delay (always coherent).
The glasses are OST: real world has zero latency, virtual must be predicted.

## Depth / room mapping today (all ours; VITURE plane/depth APIs are stubs on PC)
- Stereo from the two fisheye cams on the GPU (D3D11 compute): rectify to 320x240 pinhole (f=150 px, ~93 deg),
  census 5x5, Hamming cost, 64 disparities, 5x5 box aggregation, WTA + uniqueness + subpixel, left-right check,
  speckle removal, 3x3 median. Runs every 3rd camera frame (~8 Hz). Result is SPARSE (the dashboard depth image shows
  large holes on textureless walls, floor, ceiling).
- Accumulation: voxel hash 2.5 cm, a voxel = mean of points + hit count + up to 4 viewpoint ids + misses (ray carving).
  Confirmed when seen >= 5 times from >= 3 viewpoints. MAX_VOXELS = 400000 -> **saturated** in the last session
  (log: "400000 voxels, 8 walls, planes+mesh 152 ms (0 mesh vertices), snapshot 124 ms").
- Mesh: surface nets over confirmed voxels + smoothing, rebuilt every 4 s on CPU; MAX_MESH_VERTS 900000. Last session the
  mesh came out EMPTY (0 vertices) while the map was full. What the user sees as "the mesh" is drawn voxel cubes / points.
- Floor/planes from a height histogram; walls by RANSAC in the top view. Mesh used depth-only for occlusion.

## Options found on GitHub (verified to exist)
- Learned stereo with ONNX: HITNet (PINTO0309 model zoo, ibaiGorordo/ONNX-HITNET), CREStereo (ONNX, iter2 ~12 ms on
  RTX 3090 at 640x480), RAFT-Stereo realtime (11 ms 3090 / 24 ms 3060), NVlabs/Fast-FoundationStereo (CVPR 2026,
  16-23 ms on 3090 with TRT; zero-shot, works on mono/IR stereo; NVIDIA Open Model license; needs rectified pinhole
  input), GHUStereo / ESMStereo / LightStereo (~17-20 ms on desktop GPUs). Collection: pcb9382/StereoAlgorithms.
  All timings are on NVIDIA desktop GPUs; an Intel iGPU is likely 5-20x slower.
- Monocular depth: Depth-Anything-V2-Small (24.8M params, has metric indoor checkpoint), Any2Full / PriorDA (prompt a
  mono model with sparse metric depth), simple per-frame scale+shift least squares against sparse stereo.
- Fusion/meshing: TSDF + marching cubes — voxblox (CPU, BSD-3), Open3D VoxelBlockGrid (CPU/CUDA/SYCL, MIT),
  vdbfusion (MIT, OpenVDB), tinyvdb (header-only C, TSDF fusion + marching cubes), nvblox/InfiniTAM (CUDA).
- Latency: OST literature measures motion-to-photon with a photodiode or with laser pointers + a 30 Hz camera; the
  practical fix is late latching + correct total latency + prediction; rotation timewarp leaves translation errors.

## The end goal (stated by the user after the advisors answered)
"Occlusion objects... actually I'm not sure. I'd like to have the human I made on the web dashboard in my virtual
environment, walking around where it is possible to walk. Then I'd like an MCP for it — I already made an MCP for a
robot in another project — so two MCPs, orchestrated to assist a technician."
So the room model must serve: (a) a walkable-area map (floor minus obstacles, from floor plane + clutter heights) for a
virtual human to path-find on; (b) occlusion so the human disappears correctly behind real furniture; (c) stable
world-locked placement; (d) a queryable scene API (planes, walkable area, anchors, raycast, "move the human to X")
exposed through an MCP server, orchestrated with an existing robot MCP (two Niryo Ned arms) to assist a technician.

## The question to the council
1. Why does the virtual content lag/tremble in the glasses during fast head motion, and what concrete changes would make
   it "ultra stable"? Rank by expected gain per day of work.
2. Should the depth be improved (better classical stereo, learned stereo on the iGPU, mono depth + sparse scale,
   temporal fusion), and should the voxel mean map be replaced by TSDF + marching cubes? What exact pipeline, given an
   Intel iGPU, 8 Hz budget, and the room-scale goal (occlusion + walls + anchoring)?
3. The user also asks for reliable FLOOR and WALL detection (today: floor from a height histogram, walls by RANSAC
   on the top view of confirmed voxels; the last session reported 8 walls, never checked against reality). Should the
   room be modelled primarily as planes (floor, walls, ceiling, tables: Manhattan/plane-first, like ARKit/ARCore
   plane detection or MRUK room boxes), with the voxel/TSDF mesh only for clutter? How to detect planes robustly
   from sparse noisy stereo (per-frame plane fitting in depth, gravity from the IMU, merge over time)?
   The user adds: "showing the mesh / voxels on my screen is not a good solution" — they do not want the raw voxel
   cloud as the visual; what should the user SEE of the room in the glasses (nothing, plane outlines, a subtle grid
   on floor and walls, occlusion only), and what visual is the best test of stability?
4. What should be done FIRST on Monday?
# Five anonymous council responses (A-E)

## Response A
Q1 — Lag/tremble, ordered fixes (~1.5 days):
1. Metric cannot see predict_ms error (1 h to log, 3 h sweep). preds (myviture.cpp:950) and tw_late (:964) are compared to the engine pose at the same target, built from predict_ms=12 (state.hpp:24). At 150 deg/s, 1 ms = 0.15 deg; ~37 px/deg (1920 px / ~52 deg) -> ~5 px/ms, visible. Test: world-locked vertical line on a door frame, yaw +/-20 deg at 1 Hz (126 deg/s peak), sweep predict_ms 4->36 in 2 ms steps, pick where the line neither trails nor leads. Persist it.
2. Fallback tick ~3.5/s (2 h). When engine_pose(target) fails at :931, the frame uses our extrapolation (:922) and tw_target stays 0, so the warp is skipped for that frame (:1221). Fix: retry at target-2 ms; if it fails again, still set tw_target/tw_q from the fallback so the warp runs. Log fallback frames/s.
3. smooth_corrections (state.hpp:60) adds lag by design (1 h). :997 absorbs any revision <3 deg, including IMU-driven ones, decaying with tau 60 ms (:1008): ~0.2 deg/frame x (60/16.7) ~ 0.7 deg of extra trailing. Fix: absorb only when rate_dps<30, pass revisions through above 60 deg/s. Log |oq| p95 when fast; preds is pushed before smoothing (:950), so it misses this too.
4. Missed vblanks under fast motion (30 min): one miss = 16.7 ms late = 2.5 deg at 150 deg/s. Log missed (:1600) by head-speed bin.
Q2 — Depth: saturation is certain (assumed ~10 cm baseline, unchecked): noise at 3 m ~0.15 m; room ~72 m2 = 115k voxels single layer, ~6-voxel-thick shells -> ~700k. Fix now: evict unconfirmed voxels older than 40 frames. Replacement (3 days): own sparse TSDF, 8^3 blocks, 4 cm, projective; ~100k updates/frame on CPU at 8 Hz; marching cubes on dirty blocks; depth-only occlusion. Stereo: 4-path SGM in D3D11 compute (2 days) before any ONNX. Rate: `g_cams.seq % 3 == 0` (roommap.hpp:315) misses frames when busy -> use `seq - seen >= 3`. "0 mesh vertices" likely a logging artifact: :427 prints on every 2 s refresh but the mesh is built every 4 s (:401).
Q3 — Planes first (2 days). The wall loop caps at 8 (`wall < 8`, :155): "8 walls" = cap hit, probably shell fragments (3 cm RANSAC band vs 15 cm noise). Per-frame planes from depth normals + gravity: floor n.up>0.95, walls |n.up|<0.1; merge over time in (yaw, d), 10 deg / 15 cm; snap walls to dominant yaw +/-90. Show a faint grid on the floor polygon and wall outlines, no voxels; occlusion on. Stability test: floor grid + door-frame line.
Q4 — Monday: morning logging (fallbacks, |oq|, misses by speed, gain p50) + :427 fix, one walk session; midday smoothing A/B, predict_ms sweep; afternoon voxel eviction + seq fix.

## Response B
1. The dashboard is video see-through and cannot show the problem; the metric shares the same blindness (engine vs its own later estimate): real display latency and calibration are invisible. Two symptoms: lag/swim (2-3 deg at 150 deg/s ~ one uncounted frame; well under 1 deg slip against a real edge is noticeable) and tremble (a 25 Hz engine pose stepping under a 60 Hz display; still p95 0.7 deg is visible). Rank: predict_ms 12 is untuned (free); smooth 25 Hz steps; calibration only once tests prove it.
2. No-equipment self-tests: shake test (thin virtual line on a door frame, shake 1-2 Hz, press +/- until it neither trails nor leads, note the value); freeze test (brace head on a wall: if it shakes, it's tracking noise); turn vs walk (only walking slides -> translation/scale, timewarp is rotation only); near (0.5 m) vs far (3 m) marker; one eye at a time (eyes disagree -> per-eye calibration, IPD, 9.5 deg tilt); nod vs shake (pitch worse -> eye tilt).
3. No voxels: noisy, no real edge to compare, mesh empty anyway. Show thin lines on floor-wall seams and door frames, faint distance-faded floor grid, one test cube on a table corner; otherwise only occlusion. Walls as planes; check "8 walls" against the room. Jargon: user never touched +/-; add one "stability test" key, ms slider, "adjust until the line sticks while you shake your head".
4. Monday: 20-min self-test, then edge-line view, then decide between smoothing the 25 Hz steps, translation prediction, calibration. Depth/TSDF later; it does not affect perceived solidity during turns.

## Response C
1. The metric (myviture.cpp:943-958) measures the engine's self-consistency; wrong predict_ms/FOV/IPD/tilt score zero. If real latency is 25 ms, 150 deg/s x 13 ms = 1.95 deg missing from every number. Timewarp makes fast p95 worse (7.5 vs 6.4): (i) it warps by the raw engine delta (tw_q :933, late :1408-1410), bypassing the smoother (:973-1014) built to absorb 25 Hz visual steps, so vision updates between render and warp are re-injected as twitches ("tremble"); (ii) the engine's prediction changes at 63 Hz (15.9 ms) while render->warp <= ~10 ms, so the warp sees new data only ~63% of frames: uneven correction = judder. ~3.5/s engine_pose fails and we fall back to predicted_pose (:922 vs :931): predictor switching = micro-jumps. Rank: (a) warp delta from the smoothed pose, or A/B warp off; (b) on engine failure hold the engine's last trend; (c) measure latency, tune predict_ms; (d) slow 30 deg turn: drift even slowly = FOV/tilt.
2. Fix the map first: unconfirmed voxels only leave by carving (roommap.hpp:362), new ones dropped at 400k (:367), so noise eats the budget permanently. Empty mesh: sampling every 2nd pixel (:349) at f=150 -> 3 m pixel ~2 cm, samples ~4 cm apart, wider than 2.5 cm voxels; confirmed voxels end up isolated and fail the 2-neighbour test (:215-218). Plan: age eviction, TSDF at 4-5 cm, keep classical stereo. Learned stereo: 60-480 ms on the iGPU and trained on RGB, not fisheye IR: not now. Mono depth needs RGB on fragile USB: last.
3. Planes first, but "8 walls" with 0 mesh vertices is a red flag: RANSAC on scattered points hallucinates. Per-frame plane fits with IMU gravity, merged over time; check with a tape measure. Show occlusion only plus at most a faint floor grid. Test: one thin wireframe line on a real door-frame edge 2 m away.
4. Monday: ground truth. Phone at 240 fps slow-mo behind the lens, film the virtual line vs the real edge while turning ~150 deg/s, for warp on/off, smoother off, a few predict_ms. Lag in frames = real latency.

## Response D
1. The warp probably re-queries an unchanged pose: the engine moves at 25 Hz latest / 63 Hz at device time, so between render and warp (3-9 ms) engine_pose(target) (myviture.cpp:931) often returns the same 16 ms step — explains fast p95 worse after warp. We already receive raw 1 kHz gyro and ignore it (on_imu, :457-467). Rank: (a) gyro late-latch (1 day): integrate our own gyro from the engine pose timestamp to the warp instant, build Delta from that (ours); (b) measured predict_ms (10 ms error at 150 deg/s = 1.5 deg ~ 60 px at ~40 px/deg); (c) translation in the warp using planes as depth proxy. Make (b) a reusable tool (240 fps phone through the lens, or a "nod until it stops swimming" slider, saved per device).
2. Replace the voxel-mean map (saturated; once full it silently drops new surfaces, roommap.hpp:367). Dense TSDF on the iGPU: 6x6x3 m at 3 cm = 4M cells x 4 B = 16 MB as a D3D11 3D texture; fusing 320x240 at 8 Hz trivial; marching cubes every 1-2 s; occlusion can raycast the TSDF. "0 mesh vertices" likely a logging artifact (mesh rebuilt every other refresh, :401/:412; log prints the empty vector otherwise, :427). Fill holes with the RGB camera: Depth-Anything-V2-Small via OpenVINO/DirectML ~2 Hz on keyframes, project sparse stereo into RGB, least-squares scale+shift, fuse at low weight; skip learned stereo.
3. Plane-first room model like MRUK: floor polygon, wall quads, ceiling, table boxes with confidence/last-seen; TSDF only for clutter. Per-frame plane fit on mono+stereo depth, gravity-constrained, merged. Small API (planes, room box, anchor-to-plane, raycast) = the SDK feature VITURE lacks on PC. User sees occlusion + contact shadows; during scan a faint grid sweeps floor and walls then fades; thin glowing seam line, which is also the stability test (measurable in px).
4. Monday: gyro late-latch + predict_ms calibration first; 1 hour in parallel to confirm the mesh log artifact.

## Response E
"Wrong question": the only instrument cannot see the dominant term. Budget err = w.dt + calib + tracking + translation/dist. Time dominates: 2-3 deg at 150 deg/s = 13.3-20 ms ~ one 16.7 ms frame; at 350 deg/s one frame = 5.8 deg (matches p95 6.4-7.5). Calibration is a static offset, cannot cause tremble. Translation: 1 cm at 1 m = 0.57 deg, small. Metric compares to the engine at the ASSUMED photon time (myviture.cpp:949-957): wrong predict_ms scores zero; every 6.7 ms wrong costs 1 deg at 150 deg/s. Timewarp worsens the fast tail (6.4 -> 7.5), so the late re-query (:1408) adds noise. The engine output may not be fully predicted (pose changes at 63 Hz device time; 63 vs 60 Hz = 3 Hz beat looks like tremble — hypothesis; the `gain` stat at :939 tests full-lead extrapolation). Smoothing holds back corrections on purpose (tau 60 ms while turning, :1008, :1014): fine for video, in OST it is kept registration error. Rank: measure latency and tune predict_ms; print gain, fix shortfall with our own gyro extrapolation from on_imu (:457); A/B smooth_corrections off; check each warped present lands on its target vblank. Calibration after.
2. No learned stereo yet: 55-480 ms vs a 125 ms budget. Urgent bug is the map: 400k voxels x 6.25 cm2 = 250 m2 of surface vs ~85 m2 for a 5x4x2.5 m room, so the map is mostly unconfirmed noise; :367 then refuses new voxels -> empty mesh. Fix: evict stale unconfirmed voxels. TSDF (voxblox/tinyvdb CPU) only once it cannot starve; it won't fill textureless holes.
3. Features need: horizontal surface heights/extents (resting), vertical planes (anchoring); occlusion with 2-3 deg error is cm-wrong anyway. Plane-first: gravity, per-frame plane fits on sparse depth, merge; textureless walls are where stereo fails and plane fits need few inliers; mesh only for clutter. Show nothing by default; occlusion; optional thin outline where planes meet the floor. Test: thin virtual line on a real door frame/table edge during fast yaw.
4. Monday: no code first. Line on the door edge, shake, step predict_ms with +/- until it neither trails nor leads. Then log gain.
# Letter mapping
A = The Executor, B = The Outsider (perception / end user), C = The Contrarian, D = The Expansionist, E = The First-principles thinker

# Peer review 1 (verified myviture.cpp)
Strongest A: code claims check out — when engine_pose(target) fails at :931, tw_target stays 0 and :1221 skips the warp; preds pushed with unsmoothed q (:950) before the smoothing offset is applied (:1014); tau 60 ms when turning (:1008). Biggest blind spot B (no code: misses warp skip on fallback, raw-vs-smoothed warp, saturated map; defers depth though the map is broken now); D second (RGB holes despite fragile USB). Wrong: C/E "timewarp makes fast p95 worse (7.5 vs 6.4)" — dossier gives ranges across sessions; tw_late only filled on warped frames (:1409) while preds covers all frames (:950): not comparable. D's "same 16 ms step" is speculation (g_app.tw_deg :1413 could test it). All missed: no metric measures what was SHOWN: rendered pose is oq*q (:1014) but preds logs raw q and tw_late raw late; displayed pose with warp ~ oq*late; "after timewarp" excludes fallback frames (likely the worst); the 80 ms truth lookups (:954, :964) can also fail with preintegration errors and are dropped silently, biasing both stats.

# Peer review 2 (verified roommap.hpp, calib)
Strongest A: "0 mesh vertices" is a logging artifact — :427 logs every 2 s refresh but the mesh is rebuilt only >4 s (:401/:412); "8 walls" = `wall < 8` cap (:155); `seq % 3 == 0` (:315) skips frames. D also caught the artifact. Blind spot B. Wrong: C "confirmed voxels isolated fail 2-neighbour test -> empty mesh" and E ":367 refuses new voxels -> empty mesh": refuted by the log, lines alternate "(0 mesh vertices)" and "(900000-900024 mesh vertices)". A's ~10 cm baseline is wrong: real baseline 136.1 mm (console.log "baseline 136.1 mm", yaml t=(-0.133,0.001,-0.029)); noise at 3 m still ~0.1 m. All missed: mesh hits MAX_MESH_VERTS 900k (:38, :265) and the loop breaks in unordered_set order, dropping arbitrary room regions; each rebuild costs 1.1-1.6 s on the room thread (stereo paused ~35-40% of the time); every ~4 s the render thread copies ~21.6 MB into the vertex buffer while holding room::g.mu (myviture.cpp:1138-1145) — a likely periodic missed vblank = tremble (untested: correlate missed vblanks with mesh_version); the two cameras are toed out ~23 deg (yaml rotation 0.919), shrinking stereo overlap after rectification -> holes.

# Peer review 3 (arithmetic, end-user angle)
Strongest A (126 deg/s peak, 0.7 deg smoothing lag, 2.5 deg per missed vblank, 115k voxels all correct; hides voxels). C best diagnosis (warp bypasses smoother; only ~63% of warps see a new pose). C's phone test needs the phone taped in the eyebox and the rig turned by hand; 240 fps = 4.2 ms = 0.6 deg at 150 deg/s, adequate. Blind spot B (defers depth though the user wants floor/walls); D too big for "fast". Wrong numbers: A 37 px/deg assumes 52 deg horizontal; if 52 deg diagonal, 16:10 -> ~45 deg horizontal -> ~40 px/deg (D's), ~6 px/ms at 150 deg/s. E "translation small" wrong when walking: 1.2 m/s x 20 ms = 2.4 cm = 1.4 deg at 1 m. E "calibration cannot cause tremble" too strong: 5% FOV error at 20 deg off-centre drifts 1 deg during each turn. C/E 7.5 vs 6.4 compare unpaired ranges. All missed: sample-and-hold judder — at 60 Hz and 150 deg/s content jumps 2.5 deg (~100 px) per frame, 5.8 deg at 350 deg/s; prediction cannot remove it; try 120 Hz / low-persistence if the panel supports it. Also a day-one floor guess: standing head height minus ~0.11 m eye offset + gravity -> floor grid immediately.

# Peer review 4 (AR runtime engineer)
Strongest C: the warp delta dq = qconj(tw_q)*late (myviture.cpp:1410) uses raw engine poses, but the frame was rendered with the smoothed q = oq*q (:1009); a visual revision landing between render and warp is applied by the warp, then the next frame's smoother cancels it (:997) and re-blends over 60 ms: step, pull-back, slide = tremble by construction. Its warp-off / smoothed-delta A/B is cheap and conclusive; 240 fps through-the-lens is the only real ground truth proposed. Blind spot D: gyro late-latch is much harder than 1 day — g_imu.axis stores bare floats without per-sample timestamps (state.hpp:90, myviture.cpp:462), no IMU-to-head extrinsic, no gyro bias, and the engine already integrates that gyro; only worth it if the `gain` stat (:939) shows under-prediction. Wrong: D "same 16 ms step explains fast p95 worse after warp" — if the pose is unchanged dq is identity (:1408-1410), so warp error = prediction error, cannot be worse. All missed: IMU delivery jitter — device_now() (:146) extrapolates from the QPC time of the last IMU callback; if USB delivers IMU in bursts, every prediction target jitters by the burst gap (tremble no predict_ms fixes): measure on_imu inter-arrival gaps. Display scan-out/persistence: full-persistence 60 Hz holds each frame 16.7 ms (~2.5 deg smear at 150 deg/s); rolling scan-out makes the bottom later than the top; check 90/120 Hz modes, consider per-band warp.
