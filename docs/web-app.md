# myViture

Application web pour les lunettes **VITURE Luma Ultra**, **sans SDK ni DLL propriétaire** : tout passe par WebUSB et la webcam UVC, avec nos propres pilote, fusion IMU, profondeur stéréo, odométrie 6DoF et rendu stéréo. C'est le successeur de [myXreal](https://github.com/mazeasdamien/myXreal) (C++/ImGui pour XREAL Air 2 Ultra).

Seule dépendance externe : [MediaPipe](https://github.com/google-ai-edge/mediapipe) (open source, Apache 2.0) pour les mains, avec son modèle téléchargé au premier lancement depuis storage.googleapis.com.

## Lancer

```bash
npm install
npm run dev
```

Ouvrir **http://localhost:5180 dans Chrome ou Edge** (WebUSB nécessaire), puis « Connecter les lunettes ». Fermer SpaceWalker et arrêter le service « VitureXrRuntime » avant : un seul programme peut ouvrir le port USB des lunettes à la fois. Pour la scène 3D, les lunettes doivent être un écran séparé (Win+P → Étendre).

## Ce que fait l'appli

- **Tête** : orientation 3DoF (filtre de Mahony sur l'IMU à 1000 Hz, calibration d'usine, biais gyro suivi à l'arrêt), position 6DoF par odométrie stéréo et IMU, recentrage et mise à niveau.
- **Caméras de suivi** : stéréo 640×480 à 25 i/s, exposition automatique ou manuelle.
- **Profondeur** : rectification, census 5×5, volume de coûts agrégé, contrôle gauche-droite et filtre anti-taches sur GPU (WebGL2), environ 8 cartes/s ; occlusion du virtuel par le réel (option).
- **Mains** : MediaPipe sur les deux caméras de suivi, articulations triangulées en 3D et lissées (filtre One Euro), pincement. Source RGB possible.
- **Caméra RGB** : 1920×1080 à 30 i/s, photo, vidéo WebM. Case « MR » : la scène 3D vue depuis la caméra RGB est superposée au flux (vue à la première personne), et l'enregistrement capture le mélange.
- **Scène 3D** : passe les lunettes en 3840×1200 (3D côte à côte) par USB et ouvre une fenêtre dédiée posée sur leur écran (clic ou F11 pour le plein écran), rendu par œil. Contenu proche et lumineux, adapté à la vision directe. Pincer près d'un cube pour l'attraper, ou viser de loin et pincer. `R` recentre, `S` bascule mono/stéréo, Échap quitte (et repasse les lunettes en 2D). Un personnage animé se tient à 2 m (Rocketbox, MIT) : `1` repos, `2` parler, `3` écouter (en boucle), `4` saluer, `5` hausser les épaules, `6` réfléchir, `7` applaudir (une fois, puis retour à la boucle). `L` bascule son éclairage studio (contrasté) et l'éclairage plat de la scène.
- **Affichage** : champ vertical (28,15° par défaut), écart des yeux, position des yeux, inclinaison, prédiction de la tête par le gyroscope (30 ms par défaut), test d'ancrage avec des poteaux tous les 15° à 2 m.
- **Mains** : placées dans le repère du monde avec la pose de la tête à l'instant de l'image, puis extrapolées à l'instant d'affichage.

## Architecture

Trois Web Workers pour que l'affichage ne bloque jamais les flux USB :

| Thread | Rôle |
| --- | --- |
| page ([src/main.ts](src/main.ts)) | tableau de bord, scène 3D (three.js), caméra RGB |
| [device.worker](src/workers/device.worker.ts) | WebUSB, IMU, orientation, position inertielle, exposition auto |
| [vision.worker](src/workers/vision.worker.ts) | profondeur stéréo (GPU), odométrie, plans |
| [hands.worker](src/workers/hands.worker.ts) | MediaPipe, triangulation des mains |

## Protocole USB (35ca:1104)

Décodé à partir de captures USB ; détails dans [src/device/protocol.ts](src/device/protocol.ts) et [src/device/mcu.ts](src/device/mcu.ts).

| Endpoint | Contenu |
| --- | --- |
| `0x81` interrupt | IMU, 1000 Hz, 28 octets : compteur, horodatage (100 µs), gyro ±2000 °/s, accél ±4 g, température |
| `0x82` bulk | caméras gauche/droite 640×480, tranches de 16 Kio (en-tête `aa 8f`), plus des points caractéristiques calculés par l'appareil |
| `0x83` interrupt | vsync de l'écran, 60 Hz |
| `0x04` / `0x85` bulk | commandes `fa 55 <cmd> <flag> <len> …` : `0x95` démarrer, `0x94` arrêter, `0x90` exposition, `0x92` gain, `0xed` série, `0x80` firmware, `0xe4`/`0xe5` calibration d'usine (protobuf) |
| `0x06` / `0x87` bulk | messages MCU de 512 octets (`ff fe`, CRC-16/XMODEM) : `8` = mode d'affichage (`0x41` 2D 1920×1200, `0x42` 3D 3840×1200), luminosité, volume |

## Outils

- `npm run probe [s]` : teste le pilote WebUSB depuis Node (débits IMU et caméras, sauvegarde de la calibration et d'une image).
- `npx tsx tools/calib.ts` : affiche la calibration décodée.
- `npx tsx tools/rig.ts` : vérifie la géométrie stéréo (projection puis triangulation).
- `npx tsx tools/exposure.ts` : vérifie l'effet des commandes d'exposition et de gain.
- `npx tsx tools/display3d.ts` : passe les lunettes en 3D puis en 2D.
