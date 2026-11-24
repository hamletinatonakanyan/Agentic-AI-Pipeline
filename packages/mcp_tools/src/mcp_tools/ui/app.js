// MCP UI. Plain JS. All text is set with textContent, never innerHTML.
(function () {
  const API = "/api/mcp";
  const $ = (id) => document.getElementById(id);
  let servers = [];
  let currentTools = [];

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const showError = (e) => { $("error").textContent = e ? kit.errorText(e) : ""; };
  const json = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const list = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);

  // ===== servers ===========================================================================
  async function loadStatus() {
    const s = await kit.apiFetch(API + "/status");
    $("rules").textContent = "Allowed commands: " + (s.allowed_commands.join("  |  ") || "none") +
      ". Allowed hosts: " + s.allowed_hosts.join(", ") + ". Allowed variables: " + s.allowed_env_refs.join(", ") + ".";
    $("status").textContent = s.servers + " server(s) registered";
  }

  async function loadServers() {
    servers = await kit.apiFetch(API + "/servers");
    const t = $("servers");
    t.replaceChildren();
    if (!servers.length) return;
    const head = t.createTHead().insertRow();
    ["name", "how", "state", "allowed tools", "enabled", ""].forEach((h) => head.appendChild(el("th", "", h)));
    const body = t.createTBody();
    servers.forEach((s) => {
      const tr = body.insertRow();
      tr.insertCell().textContent = s.spec.name;
      tr.insertCell().textContent = s.spec.transport === "stdio" ? s.spec.command_line || [s.spec.command].concat(s.spec.args).join(" ") : s.spec.url;
      const state = tr.insertCell();
      state.textContent = s.state + (s.error ? ": " + s.error : "") + (s.tool_names.length ? " (" + s.tool_names.length + " tools)" : "");
      if (s.state === "error") state.className = "error";
      tr.insertCell().textContent = s.spec.tool_allowlist.join(", ") || "none";
      const en = el("input"); en.type = "checkbox"; en.checked = s.spec.enabled;
      en.addEventListener("change", async () => {
        try { await kit.apiFetch(API + "/servers/" + s.spec.name + "/enabled", json("PATCH", { enabled: en.checked })); refreshAll(); }
        catch (e) { showError(e); en.checked = !en.checked; }
      });
      tr.insertCell().appendChild(en);
      const actions = tr.insertCell();
      const check = el("button", "", "check");
      check.addEventListener("click", async () => {
        check.disabled = true;
        try { await kit.apiFetch(API + "/servers/" + s.spec.name + "/check", { method: "POST" }); showError(null); await loadServers(); }
        catch (e) { showError(e); } finally { check.disabled = false; }
      });
      const del = el("button", "", "delete");
      del.addEventListener("click", async () => {
        if (!confirm("Remove " + s.spec.name + "?")) return;
        try { await kit.apiFetch(API + "/servers/" + s.spec.name, { method: "DELETE" }); refreshAll(); } catch (e) { showError(e); }
      });
      actions.append(check, " ", del);
    });
  }

  $("n-transport").addEventListener("change", () => {
    const http = $("n-transport").value === "streamable-http";
    ["n-cmd-box", "n-env-box"].forEach((id) => { $(id).hidden = http; });
    ["n-url-box", "n-hdr-box"].forEach((id) => { $(id).hidden = !http; });
  });

  $("n-go").addEventListener("click", async () => {
    try {
      const spec = { name: $("n-name").value.trim(), transport: $("n-transport").value, tool_allowlist: list($("n-allow").value) };
      if (spec.transport === "stdio") {
        const parts = $("n-cmd").value.trim().split(/\s+/).filter(Boolean);
        spec.command = parts[0] || ""; spec.args = parts.slice(1); spec.env_refs = list($("n-env").value);
      } else {
        spec.url = $("n-url").value.trim(); spec.headers_refs = {};
        const h = $("n-hdr").value.trim();
        if (h) { const [name, ref] = h.split("="); spec.headers_refs[name.trim()] = (ref || "").trim(); }
      }
      await kit.apiFetch(API + "/servers", json("POST", spec));
      showError(null); $("n-name").value = ""; await refreshAll();
    } catch (e) { showError(e); }
  });

  // ===== tools: a form from each tool's JSON Schema ====================================================
  function fillServerSelect() {
    const sel = $("t-server"), keep = sel.value;
    sel.replaceChildren();
    servers.forEach((s) => sel.appendChild(new Option(s.spec.name + (s.spec.enabled ? "" : " (disabled)"), s.spec.name)));
    if (keep && servers.some((s) => s.spec.name === keep)) sel.value = keep;
  }

  // The first non-null type of a property, so "string or null" is shown as a string field.
  function typeOf(p) {
    if (p.type) return Array.isArray(p.type) ? p.type.find((t) => t !== "null") || "string" : p.type;
    const alt = (p.anyOf || p.oneOf || []).find((a) => a.type && a.type !== "null");
    return alt ? alt.type : "string";
  }
  function propInfo(p) {
    const alt = (p.anyOf || []).find((a) => a.type && a.type !== "null") || {};
    return Object.assign({}, alt, p);
  }

  function buildForm(schema) {
    const form = el("div");
    const fields = [];
    const required = new Set(schema.required || []);
    Object.entries(schema.properties || {}).forEach(([name, raw]) => {
      const p = propInfo(raw);
      const type = typeOf(raw);
      const label = el("label", "field");
      const bits = [name + (required.has(name) ? " *" : ""), type];
      label.appendChild(el("span", "", bits.join(" | ")));
      let input;
      if (p.enum) {
        input = el("select");
        if (!required.has(name)) input.appendChild(new Option("(not set)", ""));
        p.enum.forEach((v) => input.appendChild(new Option(String(v), String(v))));
      } else if (type === "boolean") {
        input = el("input"); input.type = "checkbox";
        label.className = "field row";
      } else if (type === "array" || type === "object") {
        input = el("textarea"); input.placeholder = type === "array" ? "[1, 2, 3]" : '{"key": "value"}';
      } else {
        input = el("input");
        if (type === "integer" || type === "number") { input.type = "number"; if (type === "integer") input.step = "1"; else input.step = "any"; }
        if (p.minimum !== undefined) input.min = p.minimum;
        if (p.maximum !== undefined) input.max = p.maximum;
        if (p.default !== undefined && p.default !== null) input.placeholder = "default " + p.default;
      }
      label.appendChild(input);
      const hints = [];
      if (p.description) hints.push(p.description);
      if (p.minimum !== undefined || p.maximum !== undefined) hints.push("range " + (p.minimum ?? "") + ".." + (p.maximum ?? ""));
      if (hints.length) label.appendChild(el("span", "hint", hints.join(". ")));
      form.appendChild(label);
      fields.push({ name, type, input, required: required.has(name) });
    });
    return {
      el: form,
      read() {
        const out = {};
        fields.forEach(({ name, type, input, required: req }) => {
          if (type === "boolean") { out[name] = input.checked; return; }
          const raw = input.value.trim();
          if (!raw) { if (req) out[name] = type === "string" ? "" : undefined; return; }
          if (type === "integer") out[name] = parseInt(raw, 10);
          else if (type === "number") out[name] = parseFloat(raw);
          else if (type === "array" || type === "object") {
            try { out[name] = JSON.parse(raw); } catch (e) { throw new Error(name + " is not valid JSON."); }
          } else out[name] = raw;
        });
        return out;
      },
    };
  }

  function showResult(box, r) {
    box.replaceChildren();
    box.className = "result " + (r.is_error ? "bad" : "good");
    box.appendChild(el("div", "", (r.is_error ? "Error (" + r.error_kind + ")" : "OK") + " | " + Math.round(r.duration_ms) + " ms" + (r.truncated ? " | output was cut" : "")));
    box.appendChild(el("pre", "", r.content));
    if (r.structured) box.appendChild(el("pre", "muted", "structured: " + JSON.stringify(r.structured, null, 2)));
  }

  function toolCard(server, tool, allowed) {
    const card = el("div", "tool");
    const head = el("div", "row");
    const box = el("input"); box.type = "checkbox"; box.checked = allowed; box.dataset.tool = tool.name;
    head.appendChild(box);
    head.appendChild(el("h3", "", tool.name));
    if (tool.read_only) head.appendChild(el("span", "badge ok", "read-only"));
    card.appendChild(head);
    card.appendChild(el("p", "desc", tool.description || "(no description)"));
    const form = buildForm(tool.input_schema || {});
    card.appendChild(form.el);
    const row = el("div", "row"); row.style.marginTop = "6px";
    const go = el("button", "", "Call");
    const out = el("div");
    go.addEventListener("click", async () => {
      go.disabled = true;
      try {
        const args = form.read();
        Object.keys(args).forEach((k) => args[k] === undefined && delete args[k]);
        const r = await kit.apiFetch(API + "/servers/" + encodeURIComponent(server) + "/tools/" + encodeURIComponent(tool.name) + "/call", json("POST", { arguments: args }));
        showError(null); showResult(out, r);
      } catch (e) {
        out.className = "result bad"; out.textContent = kit.errorText(e);
      } finally { go.disabled = false; }
    });
    row.appendChild(go);
    card.appendChild(row);
    card.appendChild(out);
    return card;
  }

  async function loadTools(refresh) {
    const name = $("t-server").value;
    const box = $("tools");
    box.replaceChildren();
    $("t-info").textContent = "";
    if (!name) return;
    try {
      const spec = servers.find((s) => s.spec.name === name).spec;
      $("t-info").textContent = "connecting...";
      const listing = await kit.apiFetch(API + "/servers/" + encodeURIComponent(name) + "/tools" + (refresh ? "?refresh=true" : ""));
      $("t-info").textContent = listing.tools.length + " tool(s)";
      if (listing.errors[name]) { box.appendChild(el("p", "error", "Could not list tools: " + listing.errors[name])); return; }
      currentTools = listing.tools;
      const allowAll = spec.tool_allowlist.includes("*");
      listing.tools.forEach((t) => box.appendChild(toolCard(name, t, allowAll || spec.tool_allowlist.includes(t.name))));
      showError(null);
    } catch (e) { $("t-info").textContent = ""; showError(e); }
  }

  $("t-load").addEventListener("click", () => loadTools(false));
  $("t-refresh").addEventListener("click", () => loadTools(true));
  $("t-server").addEventListener("change", () => loadTools(false));
  $("t-save").addEventListener("click", async () => {
    const name = $("t-server").value;
    const allowed = [...document.querySelectorAll("#tools input[type=checkbox][data-tool]")].filter((c) => c.checked).map((c) => c.dataset.tool);
    try {
      await kit.apiFetch(API + "/servers/" + encodeURIComponent(name) + "/allowlist", json("PATCH", { tool_allowlist: allowed }));
      showError(null); $("t-info").textContent = "saved: " + (allowed.join(", ") || "no tools allowed");
      await loadServers();
    } catch (e) { showError(e); }
  });

  // ===== our own server ==================================================================================
  async function loadOwn() {
    const info = await kit.apiFetch(API + "/exposed");
    const box = $("own");
    box.replaceChildren();
    box.appendChild(el("p", "", "Server name: " + info.server_name));
    box.appendChild(el("p", "rules", "Over stdio, a client launches:  " + info.stdio_command));
    box.appendChild(el("p", "rules", "Over HTTP, at " + info.http_path + " on this app. It needs the X-API-Key header, like every route. To look at it with the MCP Inspector (needs Node):  npx @modelcontextprotocol/inspector"));
    box.appendChild(el("h3", "", "Tools"));
    info.tools.forEach((t) => {
      const c = el("div", "tool");
      c.appendChild(el("h3", "", t.name));
      c.appendChild(el("p", "desc", t.description));
      c.appendChild(el("pre", "hint", JSON.stringify(t.input_schema, null, 2)));
      box.appendChild(c);
    });
    box.appendChild(el("h3", "", "Resources"));
    info.resources.concat(info.resource_templates).forEach((r) => box.appendChild(el("div", "", r)));
    box.appendChild(el("h3", "", "Prompts"));
    info.prompts.forEach((p) => box.appendChild(el("div", "", p)));
  }

  async function refreshAll() {
    try {
      await loadStatus(); await loadServers(); fillServerSelect(); showError(null);
      loadOwn().catch(() => {});
    } catch (e) { showError(e); }
  }

  kit.initTabs(document.querySelector("main"));
  kit.mountKeyBar($("keybar"), refreshAll);
  refreshAll();
})();
