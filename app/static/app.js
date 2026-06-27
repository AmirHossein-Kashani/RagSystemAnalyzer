// Minimal fetch + UI helpers shared across pages.

async function handle(res) {
  if (res.status === 204) return null;
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = text; }
  if (!res.ok) {
    const msg = (body && body.detail) || res.statusText || ('HTTP ' + res.status);
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return body;
}

const api = {
  get:  (url)         => fetch(url).then(handle),
  post: (url, body)   => fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handle),
  postForm: (url, fd) => fetch(url, { method: 'POST', body: fd }).then(handle),
  del:  (url)         => fetch(url, { method: 'DELETE' }).then(handle),
};

function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatDate(iso) {
  if (!iso) return '';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

let toastTimer = null;
function toast(msg, isError = false) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.className = 'toast' + (isError ? ' error' : '');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = 'toast hidden'; }, 3500);
}

// ----- Folder / multi-file upload helpers -----

const DEFAULT_UPLOAD_EXTS = ['.txt', '.md', '.pdf', '.html', '.htm'];

function isSupportedUpload(name, exts = DEFAULT_UPLOAD_EXTS) {
  const i = name.lastIndexOf('.');
  return i >= 0 && exts.includes(name.slice(i).toLowerCase());
}

function relativePathFor(file) {
  const raw = file.webkitRelativePath || file.name;
  return raw.replace(/\\/g, '/');
}

function collectFilesFromFileList(fileList) {
  return [...fileList].map((file) => ({
    file,
    relativePath: relativePathFor(file),
  }));
}

async function readAllEntries(reader) {
  const out = [];
  let batch;
  do {
    batch = await new Promise((resolve) => reader.readEntries(resolve));
    out.push(...batch);
  } while (batch.length);
  return out;
}

async function collectFilesFromEntry(entry, pathPrefix = '') {
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => {
      entry.file(resolve, reject);
    });
    const relativePath = pathPrefix + file.name;
    return [{ file, relativePath }];
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const children = await readAllEntries(reader);
    const nested = await Promise.all(
      children.map((child) => collectFilesFromEntry(child, pathPrefix + entry.name + '/'))
    );
    return nested.flat();
  }
  return [];
}

async function collectFilesFromDataTransfer(dataTransfer) {
  if (!dataTransfer) return [];

  const items = [...dataTransfer.items].filter((item) => item.kind === 'file');
  if (items.length && items.every((item) => typeof item.webkitGetAsEntry === 'function')) {
    const entries = items
      .map((item) => item.webkitGetAsEntry())
      .filter(Boolean);
    if (entries.length) {
      const nested = await Promise.all(entries.map((entry) => collectFilesFromEntry(entry)));
      return nested.flat();
    }
  }

  return collectFilesFromFileList(dataTransfer.files);
}
