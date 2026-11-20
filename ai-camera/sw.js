const CACHE_NAME = 'ai-camera-v2';
const ASSETS = [
  './',
  './index.html',
  './manifest.json'
];

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(
    caches.open(CACHE_NAME).then(c => c.addAll(ASSETS).catch(() => {}))
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

function saveToCache(req, res) {
  if (res && res.status === 200) {
    const clone = res.clone();
    caches.open(CACHE_NAME).then(c => c.put(req, clone));
  }
  return res;
}

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;

  // Pages: network first, so a new version on GitHub shows up right away; cache only when offline.
  if (e.request.mode === 'navigate' || e.request.destination === 'document') {
    e.respondWith(
      fetch(e.request).then(res => saveToCache(e.request, res)).catch(() => caches.match(e.request))
    );
    return;
  }

  // Library and model files: cache first (they are large and rarely change).
  e.respondWith(
    caches.match(e.request).then(cached =>
      cached || fetch(e.request).then(res => saveToCache(e.request, res))
    )
  );
});
