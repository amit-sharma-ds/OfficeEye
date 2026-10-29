// OfficeEye service worker: makes the site installable as an app (PWA) and
// shows the phone alerts the office PC sends (web push). Pages and data are
// never cached - they must be live; without a connection a page says so.

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

const OFFLINE = '<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1">' +
  '<title>OfficeEye</title><body style="font-family:system-ui;background:#111827;color:#e5e7eb;' +
  'display:grid;place-items:center;height:100vh;margin:0;text-align:center"><div>' +
  '<h2>OfficeEye</h2><p>No internet connection.<br>Live pictures and records need one - try again.</p>' +
  '<button onclick="location.reload()" style="padding:8px 16px">Try again</button></div>';

self.addEventListener('fetch', (e) => {
  if (e.request.mode !== 'navigate') return;          // everything else goes straight to the network
  e.respondWith(fetch(e.request).catch(() =>
    new Response(OFFLINE, {headers: {'Content-Type': 'text/html; charset=utf-8'}})));
});

self.addEventListener('push', (e) => {
  let d = {};
  try { d = e.data ? e.data.json() : {}; } catch (err) { d = {body: e.data && e.data.text()}; }
  e.waitUntil(self.registration.showNotification(d.title || 'OfficeEye', {
    body: d.body || '',
    icon: '/static/icon-192.png',
    badge: '/static/icon-192.png',
    image: d.image || undefined,
    tag: d.tag || undefined,
    renotify: !!d.tag,
    requireInteraction: !!d.urgent,
    data: {url: d.url || '/'},
  }));
});

// Tapping an alert opens (or brings forward) the page it is about.
self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  const url = (e.notification.data && e.notification.data.url) || '/';
  e.waitUntil(self.clients.matchAll({type: 'window', includeUncontrolled: true}).then((list) => {
    for (const c of list) {
      if ('focus' in c) { c.navigate(url).catch(() => {}); return c.focus(); }
    }
    return self.clients.openWindow(url);
  }));
});
