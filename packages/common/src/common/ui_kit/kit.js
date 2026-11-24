// UI kit: tiny helpers shared by every package UI. Plain JS, no build step.
// Usage in a page:  <script src="/ui/kit/kit.js"></script>
(function () {
  const KEY_STORAGE = "app_api_key";

  function getKey() {
    try { return sessionStorage.getItem(KEY_STORAGE) || ""; } catch (e) { return ""; }
  }
  function setKey(value) {
    try { sessionStorage.setItem(KEY_STORAGE, value); } catch (e) { /* storage blocked */ }
  }
  function clearKey() {
    try { sessionStorage.removeItem(KEY_STORAGE); } catch (e) { /* ignore */ }
  }

  // Never show the full key: only whether one is set.
  function keyStatus() {
    return getKey() ? "key set for this session" : "no key set";
  }

  // fetch() that adds X-API-Key. Throws an Error carrying the server's error envelope.
  async function apiFetch(path, options = {}) {
    const headers = Object.assign({}, options.headers, { "X-API-Key": getKey() });
    const res = await fetch(path, Object.assign({}, options, { headers }));
    const text = await res.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (e) { body = { raw: text }; }
    if (!res.ok) {
      const err = new Error((body && body.error && body.error.message) || res.statusText);
      err.status = res.status;
      err.code = body && body.error && body.error.code;
      throw err;
    }
    return body;
  }

  // Tabs: buttons with data-tab="id" show the element with that id inside `root`.
  function initTabs(root) {
    const buttons = root.querySelectorAll("[data-tab]");
    const panels = root.querySelectorAll("[data-panel]");
    function show(id) {
      buttons.forEach((b) => b.classList.toggle("active", b.dataset.tab === id));
      panels.forEach((p) => { p.hidden = p.dataset.panel !== id; });
    }
    buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
    if (buttons.length) show(buttons[0].dataset.tab);
  }

  // Table from an array of objects. Text is set with textContent, never innerHTML.
  function renderTable(el, rows, columns) {
    el.replaceChildren();
    const head = el.createTHead().insertRow();
    columns.forEach((c) => { const th = document.createElement("th"); th.textContent = c; head.appendChild(th); });
    const body = el.createTBody();
    rows.forEach((r) => {
      const tr = body.insertRow();
      columns.forEach((c) => { tr.insertCell().textContent = r[c] == null ? "" : String(r[c]); });
    });
  }

  // Server-Sent Events over fetch, so we can send the X-API-Key header (EventSource cannot).
  async function streamSSE(path, options, onEvent) {
    const headers = Object.assign({}, options && options.headers, { "X-API-Key": getKey() });
    const res = await fetch(path, Object.assign({}, options, { headers }));
    if (!res.ok || !res.body) throw new Error("stream failed: " + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const lines = chunk.split("\n");
        const event = (lines.find((l) => l.startsWith("event:")) || "event: message").slice(6).trim();
        const data = lines.filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("\n");
        onEvent(event, data);
      }
    }
  }

  // The access-key bar for a standalone package page. Hidden when the page is embedded in
  // the main app (an iframe), which already asks for the key and shares sessionStorage.
  function mountKeyBar(container, onChange) {
    if (window.self !== window.top) { container.hidden = true; return; }
    container.className = "row";
    const label = document.createElement("label");
    label.textContent = "Access key";
    const input = document.createElement("input");
    input.type = "password";
    input.autocomplete = "off";
    input.placeholder = "APP_API_KEY";
    const save = document.createElement("button");
    save.textContent = "Save";
    const clear = document.createElement("button");
    clear.textContent = "Clear";
    const state = document.createElement("span");
    state.className = "muted";
    const refresh = () => { state.textContent = keyStatus(); };
    save.addEventListener("click", () => {
      if (input.value) setKey(input.value);
      input.value = "";   // never leave the key sitting in the page
      refresh();
      onChange();
    });
    clear.addEventListener("click", () => { clearKey(); refresh(); onChange(); });
    container.replaceChildren(label, input, save, clear, state);
    refresh();
  }

  // Human-readable error text for an Error thrown by apiFetch.
  function errorText(e) {
    if (e.status === 401) return "Enter your access key.";
    if (e.status === 503) return e.message;
    return e.message || "Request failed";
  }

  // The active DOMAIN: the collection that Chat, Ingest, Retrieval and Evals default to. It is a
  // convenience kept in sessionStorage, shared by the main page and the package pages in it.
  const DOMAIN_KEY = "app_domain";
  function getDomain() {
    try { return sessionStorage.getItem(DOMAIN_KEY) || ""; } catch (e) { return ""; }
  }
  function setDomain(value) {
    try { sessionStorage.setItem(DOMAIN_KEY, value || ""); } catch (e) { /* storage blocked */ }
    window.dispatchEvent(new CustomEvent("domainchange", { detail: value || "" }));
  }
  // Calls fn(domain) when the domain changes here, or in another page of this tab.
  function onDomainChange(fn) {
    window.addEventListener("domainchange", (e) => fn(e.detail));
    window.addEventListener("storage", (e) => { if (e.key === DOMAIN_KEY) fn(e.newValue || ""); });
  }

  window.kit = {
    getKey, setKey, clearKey, keyStatus, apiFetch, initTabs, renderTable, streamSSE,
    mountKeyBar, errorText, getDomain, setDomain, onDomainChange,
  };
})();
