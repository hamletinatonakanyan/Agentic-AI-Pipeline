// Main app shell. Loads without a key, then asks for one. Only API calls need it.
(function () {
  const $ = (id) => document.getElementById(id);

  function showKeyState() { $("key-state").textContent = kit.keyStatus(); }

  function badge(text, cls) {
    const b = document.createElement("span");
    b.className = "badge " + cls;
    b.textContent = text;
    return b;
  }

  function card(c) {
    const el = document.createElement("div");
    el.className = "card";
    const h = document.createElement("h3");
    h.textContent = c.name;
    el.appendChild(h);

    const row = document.createElement("div");
    row.className = "row";
    row.appendChild(badge(c.enabled ? "enabled" : "disabled", c.enabled ? "ok" : "warn"));
    row.appendChild(badge(c.configured ? "configured" : "not configured", c.configured ? "ok" : "warn"));
    if (c.reachable !== null) {
      row.appendChild(badge(c.reachable ? "reachable" : "unreachable", c.reachable ? "ok" : "bad"));
    }
    el.appendChild(row);

    const lines = c.missing.concat(c.detail ? [c.detail] : []);
    lines.forEach((t) => {
      const p = document.createElement("div");
      p.className = "muted";
      p.textContent = t;
      el.appendChild(p);
    });
    return el;
  }

  // ----- tabs ---------------------------------------------------------------------------
  function show(id) {
    document.querySelectorAll("#tabs [data-tab]").forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === id));
    document.querySelectorAll("main [data-panel]").forEach((p) => {
      p.hidden = p.dataset.panel !== id;
      // Load a package UI the first time its tab opens, so it sees the key already entered.
      const frame = p.querySelector("iframe[data-src]");
      if (!p.hidden && frame && !frame.src) frame.src = frame.dataset.src;
    });
  }
  window.showTab = show;  // the chat uses this to open the trace in the Observability tab
  $("tabs").addEventListener("click", (e) => {
    const b = e.target.closest("[data-tab]");
    if (b) show(b.dataset.tab);
  });

  // One tab per built component that has a UI. Rebuilt on every status load.
  function renderSections(components) {
    const open = document.querySelector("#tabs .active");
    document.querySelectorAll(".dyn").forEach((n) => n.remove());
    components.filter((c) => c.enabled && c.ui_path).forEach((c) => {
      const id = "sec-" + c.name;
      const btn = document.createElement("button");
      btn.dataset.tab = id;
      btn.className = "dyn";
      btn.textContent = c.name;
      $("tabs").appendChild(btn);

      const panel = document.createElement("section");
      panel.dataset.panel = id;
      panel.className = "dyn";
      panel.hidden = true;
      const frame = document.createElement("iframe");
      frame.className = "section";
      frame.title = c.name;
      frame.dataset.src = c.ui_path;
      panel.appendChild(frame);
      document.querySelector("main").appendChild(panel);
    });
    show(open && document.querySelector('#tabs [data-tab="' + open.dataset.tab + '"]')
      ? open.dataset.tab : "status");
  }

  // ----- the active domain (a collection) --------------------------------------------------
  async function loadDomains() {
    try {
      const cols = await kit.apiFetch("/api/ingest/collections");
      $("domain-list").replaceChildren(...cols.map((c) => new Option(c.name, c.name)));
      $("domain-note").textContent = cols.length
        ? "Existing domains: " + cols.map((c) => c.name + " (" + c.points + " chunks)").join(", ") + ". A new name is created by its first ingest."
        : "No domain yet: ingest some files into a name of your choice.";
    } catch (e) { /* ingest may be off or Qdrant down: the box still works as free text */ }
  }
  function applyDomain() {
    const v = $("domain").value.trim();
    if (v && !/^[A-Za-z0-9_.-]{1,100}$/.test(v)) { $("domain-note").textContent = "Use letters, digits, '_', '.' or '-' (up to 100)."; return; }
    kit.setDomain(v);
  }
  $("domain").value = kit.getDomain();
  $("domain").addEventListener("change", applyDomain);
  kit.onDomainChange(() => { if ($("domain").value.trim() !== kit.getDomain()) $("domain").value = kit.getDomain(); });

  async function loadStatus() {
    $("status-error").textContent = "";
    $("cards").replaceChildren();
    try {
      const data = await kit.apiFetch("/api/status");
      data.components.forEach((c) => $("cards").appendChild(card(c)));
      renderSections(data.components);
      loadDomains();
      // The chat reads the same status to know whether the agent can answer.
      document.dispatchEvent(new CustomEvent("status", { detail: data.components }));
    } catch (e) {
      $("status-error").textContent =
        e.status === 401 ? "Enter your access key above." :
        e.status === 503 ? "APP_API_KEY is not set on the server. Add it to .env and restart." :
        e.message;
    }
  }

  $("save").addEventListener("click", () => {
    const v = $("key").value;
    if (v) kit.setKey(v);
    $("key").value = "";   // never leave the key sitting in the page
    showKeyState();
    loadStatus();
  });
  $("clear").addEventListener("click", () => { kit.clearKey(); showKeyState(); loadStatus(); });
  $("refresh").addEventListener("click", loadStatus);

  showKeyState();
  loadStatus();
})();
