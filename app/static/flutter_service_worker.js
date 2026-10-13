// Replaces the service worker that earlier versions of the web app installed. Browsers re-check this file on every
// visit; this version empties the old cache, removes itself and reloads the page once, so the visitor gets the
// current app. The app no longer registers a service worker (flutter_app/web/flutter_bootstrap.js).
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      await caches.delete(key);
    }
    await self.registration.unregister();
    for (const client of await self.clients.matchAll({ type: 'window' })) {
      client.navigate(client.url);
    }
  })());
});
