const CACHE_NAME = 'ai-objects-v1';
const ASSETS = ['./', './index.html', './manifest.json', './lib/tf.min.js', './lib/coco-ssd.min.js'];

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS).catch(() => {})));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

function put(req, res) {
  if (res && res.status === 200) { const c = res.clone(); caches.open(CACHE_NAME).then(x => x.put(req, c)); }
  return res;
}

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  // pages: network first, so updates on GitHub show up right away
  if (req.mode === 'navigate' || req.destination === 'document') {
    e.respondWith(fetch(req).then(r => put(req, r)).catch(() => caches.match(req)));
    return;
  }
  // libraries and model weights: cache first (large, rarely change)
  e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(r => put(req, r))));
});
