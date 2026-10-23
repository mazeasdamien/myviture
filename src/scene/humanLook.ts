// Lighting helpers for the human: environment reflections, a soft ground shadow and
// light scattering in the skin. Everything works per renderer: the scene is drawn by
// several WebGL contexts (dashboard, glasses window), so nothing here is a render target.

import * as THREE from 'three';

/**
 * A small studio painted on a canvas, used as an equirectangular environment: bright
 * ceiling, warm walls with two softboxes and a window, darker floor. Each renderer turns
 * it into its own prefiltered map, which gives the skin, eyes and fabric soft reflections
 * and fills the shadows with coloured light instead of a flat ambient.
 */
export function studioEnvironment(): THREE.Texture {
  const W = 512, H = 256;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d')!;
  const v = g.createLinearGradient(0, 0, 0, H);
  v.addColorStop(0, '#f4f2ee');    // ceiling
  v.addColorStop(0.38, '#c9c0b4'); // upper walls
  v.addColorStop(0.5, '#8f877d');  // horizon
  v.addColorStop(0.62, '#5d564f'); // lower walls
  v.addColorStop(1, '#2e2a27');    // floor
  g.fillStyle = v;
  g.fillRect(0, 0, W, H);
  const box = (x: number, y: number, w: number, h: number, colour: string) => {
    g.filter = 'blur(6px)';
    g.fillStyle = colour;
    g.fillRect(x * W, y * H, w * W, h * H);
    g.filter = 'none';
  };
  box(0.18, 0.2, 0.1, 0.16, '#ffffff');  // key softbox
  box(0.62, 0.24, 0.08, 0.12, '#fff3e2'); // fill softbox, warmer
  box(0.4, 0.3, 0.16, 0.12, '#e8f1ff');  // window, cooler
  const tex = new THREE.CanvasTexture(c);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Black radial falloff for contact shadows (alpha carries the shape). */
export function shadowTexture(): THREE.Texture {
  const S = 128;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d')!;
  const r = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  r.addColorStop(0, 'rgba(0,0,0,1)');
  r.addColorStop(0.45, 'rgba(0,0,0,0.55)');
  r.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = r;
  g.fillRect(0, 0, S, S);
  return new THREE.CanvasTexture(c);
}

/** A flat shadow blob on the floor, w x d metres. */
export function shadowBlob(map: THREE.Texture, w: number, d: number, opacity: number): THREE.Mesh {
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(w, d).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ map, transparent: true, opacity, depthWrite: false }),
  );
  m.position.y = 0.002;
  m.renderOrder = -1;
  return m;
}

// The scene's lights are tuned for see-through glasses: a strong hemisphere light keeps
// every object bright, which leaves the human flat. His materials scale the two lights
// themselves (shared uniforms, so switching costs no recompile and the other objects
// keep their lighting): less ambient, more key, the environment map filling in.
export const humanLight = { ambient: { value: 1 }, key: { value: 1 } };
const UNIFORMS = '#include <common>\nuniform float humanAmbient;\nuniform float humanKey;';
const KEY = 'getDirectionalLightInfo( directionalLight, directLight );';
const AMBIENT = 'irradiance += getHemisphereLightIrradiance(';

// Wrap lighting: light reaches a little past the shadow line, more in red than in blue,
// the way it does through skin. Only where the albedo looks like skin (clearly redder
// than blue), so the black polo, grey trousers and white shoes keep plain Lambert.
const DIFFUSE = 'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - F );';
const SCATTER = `
	#ifdef SKIN_SCATTER
		vec3 skinAlbedo = material.diffuseContribution;
		float skinMask = smoothstep( 0.04, 0.14, skinAlbedo.r - skinAlbedo.b ) * smoothstep( 0.08, 0.2, skinAlbedo.r );
		vec3 skinWrap = SKIN_SCATTER * skinMask;
		vec3 skinIrradiance = saturate( ( dot( geometryNormal, directLight.direction ) + skinWrap ) / ( 1.0 + skinWrap ) ) * directLight.color;
		reflectedLight.directDiffuse += skinIrradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - F );
	#else
		${DIFFUSE}
	#endif`;

// The hair painted on the scalp is a very dark brown: the 4% specular every dielectric
// gets, strongest at grazing angles, outshone it and turned the top and back of the head
// into a grey plastic shell. No specular where the albedo looks like dark hair (dark,
// redder than blue); the black polo (neutral) and the skin (light) keep theirs.
const MATTE_HAIR = `#include <lights_physical_fragment>
	{
		vec3 a = material.diffuseColor;
		float brown = ( a.r - a.b ) / ( a.r + 0.002 );
		float hairness = ( 1.0 - smoothstep( 0.03, 0.07, dot( a, vec3( 0.2126, 0.7152, 0.0722 ) ) ) ) * smoothstep( 0.15, 0.35, brown );
		material.specularColor *= 1.0 - hairness;
		material.specularColorBlended *= 1.0 - hairness;
		material.specularF90 *= 1.0 - hairness;
	}`;
let warned = false;

/**
 * Human lighting on a standard material: the light balance above, matte painted hair, plus skin scattering
 * when `scatter` is set. A no-op (with one warning) if three's shaders changed.
 */
export function humanShading(material: THREE.MeshStandardMaterial, scatter: boolean, wrap = [0.5, 0.22, 0.14]) {
  const begin = THREE.ShaderChunk.lights_fragment_begin;
  const physical = THREE.ShaderChunk.lights_physical_pars_fragment;
  if (!begin.includes(KEY) || !begin.includes(AMBIENT) || !physical.includes(DIFFUSE) || !THREE.ShaderChunk.lights_physical_fragment.includes('specularColorBlended')) {
    if (!warned) console.warn('human: custom lighting skipped, three.js lighting shaders changed');
    warned = true;
    return;
  }
  if (scatter) material.defines = { ...material.defines, SKIN_SCATTER: `vec3( ${wrap.map(n => n.toFixed(3)).join(', ')} )` };
  material.onBeforeCompile = shader => {
    shader.uniforms.humanAmbient = humanLight.ambient;
    shader.uniforms.humanKey = humanLight.key;
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', UNIFORMS)
      .replace('#include <lights_fragment_begin>', begin.replace(KEY, `${KEY}\n\t\tdirectLight.color *= humanKey;`).replace(AMBIENT, 'irradiance += humanAmbient * getHemisphereLightIrradiance('))
      .replace('#include <lights_physical_pars_fragment>', physical.replace(DIFFUSE, SCATTER))
      .replace('#include <lights_physical_fragment>', MATTE_HAIR);
  };
  material.customProgramCacheKey = () => 'human-shading';
  material.needsUpdate = true;
}
