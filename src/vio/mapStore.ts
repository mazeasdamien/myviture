// Persistence of the environment memory (landmarks + anchors) in IndexedDB, one record
// for the room the glasses were last used in.

import type { MapData } from './envMap';

export interface Anchor {
  id: string;
  label: string;
  /** position in the map frame, metres */
  m: [number, number, number];
  created: number;
}

export interface StoredMap {
  map: MapData;
  anchors: Anchor[];
  saved: number;
}

const DB = 'myviture';
const STORE = 'maps';
const KEY = 'room';

function db(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function loadMap(): Promise<StoredMap | null> {
  try {
    const d = await db();
    return await new Promise(resolve => {
      const req = d.transaction(STORE).objectStore(STORE).get(KEY);
      req.onsuccess = () => resolve((req.result as StoredMap) ?? null);
      req.onerror = () => resolve(null);
    });
  } catch { return null; }
}

export async function saveMap(m: StoredMap): Promise<void> {
  try {
    const d = await db();
    await new Promise<void>(resolve => {
      const tx = d.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(m, KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch { /* storage blocked: memory lasts for this session only */ }
}

export async function clearMap(): Promise<void> {
  try {
    const d = await db();
    d.transaction(STORE, 'readwrite').objectStore(STORE).delete(KEY);
  } catch { /* nothing stored */ }
}
