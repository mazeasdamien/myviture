# myViture

**English** · [Français](#français)

A technician assistant for the **VITURE Luma Ultra** optical see-through AR glasses: a virtual human who walks in your room, stands on the real floor and shows you where things are, driven by voice or by another program (MCP).

<p align="center">
  <img src="docs/images/human-in-room-map.jpg" width="49%" alt="The virtual human in the room's 3D map">
  <img src="docs/images/room-map-top-view.jpg" width="49%" alt="The room map seen from above">
</p>
<p align="center">
  <img src="docs/images/rgb-view.jpg" width="49%" alt="The glasses' RGB camera view">
  <img src="docs/images/stereo-depth.jpg" width="49%" alt="Stereo depth: tracking camera, census, SGM">
</p>

## What it does

The native Windows app (`native/`, C++ / Direct3D 11):

- **Head tracking**: 6DoF from VITURE's engine, with our own late latching and rotation timewarp. Side-by-side 3840×1200 at 60 or 90 Hz.
- **Room map**: GPU stereo depth (census + 8-path SGM) from the two tracking cameras. Depth Anything V2 runs on the RGB camera, scaled by the stereo. Then a voxel map, a mesh for occlusion, the floor and the walkable area.
- **Virtual human**: Rocketbox character, idle and walk animations, A* paths on the floor seen. It comes to you, wanders, and walks to an object to point at it.
- **Hands**: skeleton from VITURE's hand tracking, predicted to the display. Three cubes fall on the scanned surfaces and can be grabbed and thrown.
- **Objects and voice**: name what you look at (« scan drill »), then ask « where is the drill? ». Speech goes to Whisper through the lab's Orange gateway.
- **MCP server** (JSON-RPC over HTTP): room, user, human, markers, robot-frame calibration, so a robot's MCP can orchestrate it.
- **Recording**: MP4 of the RGB camera with the virtual content over it. A dashboard (Dear ImGui) shows RGB, 3D and diagnostics views.

The first version, a WebUSB web app with its own USB driver, is in `src/`. It is documented in [docs/web-app.md](docs/web-app.md), with the decoded USB protocol.

## Status

Work on the PC version stopped in October 2026. The next step is a standalone version on the VITURE neckband (Unity SDK).

## Build and run

Windows 11, Visual Studio 2026 (C++), and VITURE SpaceWalker installed (the app loads its `carina_vio.dll`, not included here). Glasses connected, display set to Extend (Win+P).

```bat
native\fetch_deps.bat
native\build.bat
native\myviture.exe
```

`fetch_deps.bat` downloads ONNX Runtime, DirectML, the hand and Depth Anything models, and bakes the human. Voice commands need `.env.local` at the repository root with `ORANGE_GATEWAY` and `ORANGE_PROXY_SECRET`. It is never committed.

## Credits

Microsoft Rocketbox avatars (MIT), Depth Anything V2 Small (Apache 2.0), ONNX Runtime and DirectML (MIT), Dear ImGui (MIT), MediaPipe hand models via opencv_zoo (Apache 2.0). VITURE's runtime is proprietary and not part of this repository.

---

## Français

Un assistant pour technicien sur les lunettes de réalité augmentée **VITURE Luma Ultra** (optique transparente). Un humain virtuel marche dans ta pièce, se tient sur le vrai sol et te montre où sont les choses. On le pilote à la voix ou par un autre programme (MCP).

## Ce que fait l'appli

L'application Windows native (`native/`, C++ / Direct3D 11) :

- **Suivi de la tête** : 6DoF par le moteur de VITURE, avec notre late latching et notre timewarp en rotation. Côte à côte 3840×1200 à 60 ou 90 Hz.
- **Carte de la pièce** : profondeur stéréo sur GPU (census + SGM sur 8 directions) depuis les deux caméras de suivi. Depth Anything V2 tourne sur la caméra RGB, mis à l'échelle par la stéréo. Ensuite une carte de voxels, un maillage pour l'occlusion, le sol et la zone marchable.
- **Humain virtuel** : personnage Rocketbox, animations de repos et de marche, chemins A* sur le sol vu. Il vient vers toi, se promène, et va jusqu'à un objet pour le montrer du bras.
- **Mains** : squelette issu du suivi des mains de VITURE, prédit jusqu'à l'affichage. Trois cubes tombent sur les surfaces scannées ; on les attrape et on les lance.
- **Objets et voix** : nommer ce qu'on regarde (« scan perceuse »), puis demander « où est la perceuse ? ». La parole part à Whisper par la passerelle Orange du labo.
- **Serveur MCP** (JSON-RPC sur HTTP) : pièce, porteur, humain, repères, calage sur le repère du robot, pour être orchestré par le MCP d'un robot.
- **Enregistrement** : MP4 de la caméra RGB avec le virtuel par-dessus. Un tableau de bord (Dear ImGui) montre les vues RGB, 3D et diagnostics.

La première version, une application web WebUSB avec son propre pilote USB, est dans `src/`. Elle est décrite dans [docs/web-app.md](docs/web-app.md), avec le protocole USB décodé.

## État

Le travail sur la version PC s'est arrêté en octobre 2026. La suite : une version autonome sur le neckband VITURE (SDK Unity).

## Compiler et lancer

Windows 11, Visual Studio 2026 (C++) et VITURE SpaceWalker installé (l'appli charge son `carina_vio.dll`, absent de ce dépôt). Lunettes branchées, affichage en mode Étendre (Win+P).

```bat
native\fetch_deps.bat
native\build.bat
native\myviture.exe
```

`fetch_deps.bat` télécharge ONNX Runtime, DirectML, les modèles des mains et de Depth Anything, et prépare l'humain. Les commandes vocales demandent un `.env.local` à la racine du dépôt avec `ORANGE_GATEWAY` et `ORANGE_PROXY_SECRET`. Ce fichier n'est jamais commité.

## Crédits

Avatars Microsoft Rocketbox (MIT), Depth Anything V2 Small (Apache 2.0), ONNX Runtime et DirectML (MIT), Dear ImGui (MIT), modèles des mains MediaPipe via opencv_zoo (Apache 2.0). Le runtime de VITURE est propriétaire et ne fait pas partie de ce dépôt.
