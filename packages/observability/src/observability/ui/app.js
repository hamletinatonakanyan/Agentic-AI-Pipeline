// Observability UI: a trace list, and for the selected trace either the RUN view (for an agent run:
// the question, the answer, usage, and every step with its details) or the plain span waterfall.
// Works standalone (/ui) and inside the main app (/ui/observability). All text via textContent.
(function () {
  const API = "/api/obs";
  const $ = (id) => document.getElementById(id);
  let current = null;       // the trace tree on show
  let scope = null;         // a conversation id the list is limited to

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : ms.toFixed(1) + " ms"; }
  function fmtTime(ts) { return new Date(ts * 1000).toLocaleString(); }
  function fmtCost(c) { return c == null ? null : "$" + (c < 0.01 ? c.toFixed(6) : c.toFixed(4)); }
  const short = (id) => (id ? String(id).slice(0, 8) : "");

  async function loadStatus() {
    try {
      const s = await kit.apiFetch(API + "/status");
      $("status").textContent = s.enabled
        ? "enabled, exporters: " + s.exporters.join(", ") + ", store: " + s.store
        : "observability is off (OBS_ENABLED=false)";
    } catch (e) { $("status").textContent = ""; }
  }

  // ===== the list ===========================================================================
  function query() {
    const p = new URLSearchParams();
    if ($("f-name").value) p.set("name", $("f-name").value);
    if ($("f-status").value) p.set("status", $("f-status").value);
    if ($("f-dur").value) p.set("min_duration_ms", $("f-dur").value);
    if (scope) p.set("session_id", scope);
    p.set("limit", "100");
    return p.toString();
  }

  function setScope(id) {
    scope = id;
    $("scope").hidden = !id;
    if (id) $("scope-text").textContent = "Showing every run of conversation " + id;
  }

  async function loadTraces() {
    $("error").textContent = "";
    try {
      const rows = await kit.apiFetch(API + "/traces?" + query());
      $("empty").hidden = rows.length > 0;
      if (scope && rows.length) $("scope-text").textContent = "Conversation " + scope + ": " + rows.length + " run(s), newest first";
      const table = $("traces");
      table.replaceChildren();
      if (!rows.length) return;
      const head = table.createTHead().insertRow();
      ["time", "what", "conversation", "duration", "status", "calls", "tokens in / out", "est. cost"].forEach((t) => head.appendChild(el("th", "", t)));
      const body = table.createTBody();
      rows.forEach((r) => {
        const tr = body.insertRow();
        tr.className = "clickable";
        tr.insertCell().textContent = fmtTime(r.start_ts);
        const q = tr.insertCell(); q.className = "q";
        if (r.input_preview) { q.appendChild(el("div", "qtext", r.input_preview)); q.appendChild(el("div", "qroot", r.root_name)); }
        else q.textContent = r.root_name;
        const c = tr.insertCell();
        if (r.session_id) {
          const b = el("button", "id", short(r.session_id));
          b.title = "Show every run of conversation " + r.session_id;
          b.addEventListener("click", (e) => { e.stopPropagation(); setScope(r.session_id); loadTraces(); });
          c.appendChild(b);
        }
        tr.insertCell().textContent = fmtMs(r.duration_ms);
        const st = tr.insertCell();
        st.appendChild(el("span", "badge " + (r.status === "error" ? "bad" : "ok"), r.status));
        tr.insertCell().textContent = r.llm_calls || "";
        tr.insertCell().textContent = r.input_tokens + " / " + r.output_tokens;
        tr.insertCell().textContent = fmtCost(r.cost_usd) || "";
        tr.addEventListener("click", () => openTrace(r.trace_id));
      });
    } catch (e) { $("error").textContent = kit.errorText(e); }
  }

  // One box for both kinds of id: a trace id opens that trace, anything else is a conversation id.
  async function find() {
    const id = $("f-find").value.trim();
    $("error").textContent = "";
    if (!id) { setScope(null); loadTraces(); return; }
    try {
      await kit.apiFetch(API + "/traces/" + encodeURIComponent(id));
      await openTrace(id);
      return;
    } catch (e) { if (e.status !== 404) { $("error").textContent = kit.errorText(e); return; } }
    setScope(id);
    await loadTraces();
    if (!$("traces").rows.length) $("error").textContent = "No trace or conversation with id " + id + ".";
  }

  // ===== helpers over the span tree ===========================================================
  function flatten(nodes, depth, out) {
    nodes.forEach((n) => { out.push({ node: n, depth }); flatten(n.children, depth + 1, out); });
    return out;
  }
  function descendants(node) { return flatten(node.children, 0, []).map((x) => x.node); }
  const A = (n, k) => (n.attributes || {})[k];

  // ===== opening a trace ======================================================================
  async function openTrace(id) {
    $("error").textContent = "";
    try {
      current = await kit.apiFetch(API + "/traces/" + encodeURIComponent(id));
    } catch (e) { $("error").textContent = kit.errorText(e); return; }
    const root = current.roots[0];
    const isRun = root.name === "agent.run";
    $("detail-title").textContent = isRun ? "Agent run" : root.name;
    const s = current.summary;
    $("detail-summary").textContent = fmtTime(s.start_ts) + " · " + fmtMs(s.duration_ms) + " · " + s.span_count + " spans · trace " + s.trace_id;
    $("views").hidden = !isRun;
    renderSpans(current);
    if (isRun) { renderRun(current); showView("run"); } else { showView("spans"); }
    $("detail").hidden = false;
    $("detail").scrollIntoView({ behavior: "smooth" });
  }

  function showView(name) {
    $("view-run").hidden = name !== "run";
    $("view-spans").hidden = name !== "spans";
    document.querySelectorAll("#views [data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  }
  $("views").addEventListener("click", (e) => { const b = e.target.closest("[data-view]"); if (b) showView(b.dataset.view); });

  // ===== the plain waterfall ==================================================================
  function renderSpans(tree) {
    const rows = flatten(tree.roots, 0, []);
    const t0 = Math.min(...rows.map((r) => r.node.start_ts));
    const t1 = Math.max(...rows.map((r) => r.node.end_ts));
    const total = Math.max((t1 - t0) * 1000, 0.001);
    const wf = $("waterfall");
    wf.replaceChildren();
    rows.forEach(({ node, depth }) => {
      const row = el("div", "wf");
      const name = el("div", "name", node.name);
      name.style.paddingLeft = depth * 14 + "px";
      name.title = node.name;
      const track = el("div", "track");
      const bar = el("div", "bar" + (node.status === "error" ? " error" : ""));
      bar.style.left = ((node.start_ts - t0) * 1000 / total * 100) + "%";
      bar.style.width = Math.max(node.duration_ms / total * 100, 0.4) + "%";
      track.appendChild(bar);
      row.append(name, track, el("div", "ms", fmtMs(node.duration_ms)));
      const extra = el("pre", "attrs");
      extra.hidden = true;
      const info = Object.assign({}, node.attributes);
      if (node.status_message) info.status_message = node.status_message;
      extra.textContent = JSON.stringify(info, null, 2);
      row.addEventListener("click", () => { extra.hidden = !extra.hidden; });
      wf.append(row, extra);
    });
  }

  // ===== the run view ==========================================================================
  const STEPS = {
    "agent.recall":     { kind: "memory", title: "Memory recall",       sub: "long-term facts" },
    "agent.load_tools": { kind: "tools",  title: "Load tools",          sub: "what the agent may use" },
    "agent.route":      { kind: "route",  title: "Decide the route",    sub: "search, tool or direct" },
    "agent.retrieve":   { kind: "search", title: "Search documents",    sub: "retrieval" },
    "agent.plan_tools": { kind: "tools",  title: "Plan tool calls",     sub: "model chooses tools" },
    "agent.tool":       { kind: "tools",  title: "Run a tool",          sub: "" },
    "agent.generate":   { kind: "write",  title: "Write the answer",    sub: "model" },
    "agent.check":      { kind: "check",  title: "Check the answer",    sub: "against the sources" },
    "agent.respond":    { kind: "cite",   title: "Citations",           sub: "link [n] to sources" },
    "agent.remember":   { kind: "memory", title: "Save facts",          sub: "long-term memory" },
  };

  function tile(value, label, cls) {
    const t = el("div", "tile " + (cls || ""));
    t.appendChild(el("div", "v", value));
    t.appendChild(el("div", "l", label));
    return t;
  }

  function renderRun(tree) {
    const root = tree.roots[0];
    const box = $("view-run");
    box.replaceChildren();
    const s = tree.summary;
    const all = flatten(tree.roots, 0, []).map((x) => x.node);

    // -- identity line: conversation and trace
    const ident = el("p", "muted");
    ident.appendChild(document.createTextNode("Conversation "));
    if (s.session_id) {
      const b = el("button", "id", s.session_id);
      b.title = "Show every run of this conversation";
      b.addEventListener("click", () => { setScope(s.session_id); loadTraces(); window.scrollTo({ top: 0, behavior: "smooth" }); });
      ident.appendChild(b);
    } else ident.appendChild(document.createTextNode("(none)"));
    ident.appendChild(document.createTextNode("  ·  status "));
    ident.appendChild(el("span", "badge " + (s.status === "error" ? "bad" : "ok"), s.status));
    box.appendChild(ident);

    // -- question and answer
    const io = el("div", "io");
    const q = el("div", "box"); q.appendChild(el("h3", "", "Question"));
    q.appendChild(el("div", "text", A(root, "input.value") || "(not recorded: AGENT_TRACE_CONTENT is off)"));
    const a = el("div", "box"); a.appendChild(el("h3", "", "Answer"));
    a.appendChild(el("div", "text", A(root, "output.value") || (A(root, "agent.status") === "interrupted" ? "(waiting for an approval)" : "(not recorded)")));
    io.append(q, a);
    box.appendChild(io);

    // -- usage
    const tools = all.filter((n) => n.name === "agent.tool").length;
    const tiles = el("div", "tiles");
    tiles.appendChild(tile(fmtMs(s.duration_ms), "total time"));
    tiles.appendChild(tile(s.llm_calls, "model calls"));
    tiles.appendChild(tile(s.input_tokens, "tokens in"));
    tiles.appendChild(tile(s.output_tokens, "tokens out"));
    tiles.appendChild(tile(fmtCost(s.cost_usd) || "no price set", s.cost_usd == null ? "set LLM_PRICE_* in .env" : "estimated cost", "cost"));
    tiles.appendChild(tile(tools, "tool calls"));
    tiles.appendChild(tile(A(root, "agent.retries") || 0, "answer retries"));
    tiles.appendChild(tile(A(root, "agent.passages") || 0, "passages used"));
    box.appendChild(tiles);

    // -- steps, in the order they started
    const t0 = root.start_ts, total = Math.max(root.duration_ms, 0.001);
    const steps = all.filter((n) => STEPS[n.name]).sort((x, y) => x.start_ts - y.start_ts);
    box.appendChild(el("h3", "", "Steps (" + steps.length + ")"));
    const list = el("div", "steps");
    steps.forEach((n, i) => list.appendChild(stepCard(n, i + 1, t0, total)));
    box.appendChild(list);
  }

  function stepCard(node, index, t0, total) {
    const def = STEPS[node.name];
    const card = el("div", "step k-" + def.kind + (node.status === "error" ? " err" : ""));
    const top = el("div", "top");
    top.appendChild(el("div", "n", index));
    const title = el("div", "title", def.title + attemptLabel(node));
    if (def.sub) title.appendChild(el("span", "sub", def.sub));
    top.appendChild(title);
    top.appendChild(el("div", "what", headline(node)));
    const track = el("div", "track");
    const bar = el("div", "bar");
    bar.style.left = Math.max(0, (node.start_ts - t0) * 1000 / total * 100) + "%";
    bar.style.width = Math.max(node.duration_ms / total * 100, 0.6) + "%";
    track.appendChild(bar);
    top.appendChild(track);
    top.appendChild(el("div", "ms", fmtMs(node.duration_ms)));
    card.appendChild(top);

    const body = el("div", "body");
    body.hidden = true;
    fillBody(body, node);
    card.appendChild(body);
    top.addEventListener("click", () => { body.hidden = !body.hidden; });
    return card;
  }

  function attemptLabel(n) {
    const attempt = A(n, "agent.attempt"), round = A(n, "agent.round");
    if (attempt && attempt > 1) return " (attempt " + attempt + ")";
    if (round && round > 1) return " (round " + round + ")";
    return "";
  }

  // The one-line summary shown on the closed card.
  function headline(n) {
    switch (n.name) {
      case "agent.recall": return (A(n, "agent.facts") || 0) + " fact(s) recalled";
      case "agent.load_tools": return (A(n, "agent.tools") || 0) + " tool(s) available";
      case "agent.route": return "→ " + (A(n, "agent.route") || "?") + (A(n, "agent.reason") ? ": " + A(n, "agent.reason") : "");
      case "agent.retrieve": return A(n, "agent.error") ? "search failed: " + A(n, "agent.error") : (A(n, "agent.passage_count") || 0) + " passage(s), mode " + (A(n, "agent.mode") || "?") + (A(n, "agent.rerank") ? " + rerank" : "");
      case "agent.plan_tools": { const c = A(n, "agent.call_names") || []; return c.length ? "calls " + c.join(", ") : "no tool needed"; }
      case "agent.tool": return (A(n, "agent.tool") || "tool") + " → " + (A(n, "agent.tool_status") || "?");
      case "agent.generate": return (A(n, "agent.answer_chars") || 0) + " characters" + (A(n, "agent.streamed") ? ", streamed" : "");
      case "agent.check": return A(n, "agent.passed") ? "all claims supported" : (A(n, "agent.unsupported") || 0) + " unsupported claim(s)";
      case "agent.respond": return (A(n, "agent.citations") || 0) + " citation(s)" + (A(n, "agent.warnings") ? ", " + A(n, "agent.warnings") + " warning(s)" : "");
      case "agent.remember": return (A(n, "agent.facts_saved") || 0) + " fact(s) saved";
      default: return "";
    }
  }

  function section(body, label, value, pre) {
    if (value === undefined || value === null || value === "" || (Array.isArray(value) && !value.length)) return;
    body.appendChild(el("h4", "", label));
    const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    body.appendChild(el(pre || typeof value !== "string" ? "pre" : "div", "val", text));
  }

  function fillBody(body, n) {
    const a = (k) => A(n, k);
    switch (n.name) {
      case "agent.route":
        section(body, "Decision", a("agent.route"));
        section(body, "Why", a("agent.reason"));
        section(body, "Question rewritten to stand alone", a("agent.standalone_question"));
        section(body, "Could search documents", String(a("agent.can_retrieve")));
        section(body, "Tools offered", a("agent.tools_offered"));
        break;
      case "agent.retrieve": {
        section(body, "Query sent to search", a("agent.query"));
        section(body, "Search settings", "mode " + (a("agent.mode") || "?") + ", rerank " + (a("agent.rerank") ? "on" : "off") + ", top " + (a("agent.top_k") || "?") + (a("agent.collection") ? ", domain " + a("agent.collection") : ""));
        section(body, "Query variants searched", a("agent.rewritten_queries"));
        section(body, "Error", a("agent.error"));
        const ps = a("agent.passages");
        if (Array.isArray(ps)) {
          body.appendChild(el("h4", "", "Passages given to the model"));
          ps.forEach((p) => {
            const d = el("div", "passage");
            d.appendChild(el("div", "head", "[" + p.n + "] " + (p.source || "source") + (p.page ? ", page " + p.page : "") + " · score " + p.score + " · " + p.chunk_id));
            d.appendChild(el("div", "val", p.text));
            body.appendChild(d);
          });
        }
        break;
      }
      case "agent.plan_tools":
        section(body, "Model note", a("agent.model_note"));
        section(body, "Tool calls requested", a("agent.call_arguments"));
        break;
      case "agent.tool":
        section(body, "Tool", a("agent.tool"));
        section(body, "Arguments", a("agent.tool_arguments"));
        section(body, "Result", a("agent.tool_result"), true);
        section(body, "Status", a("agent.tool_status") + (a("agent.tool_error_kind") ? " (" + a("agent.tool_error_kind") + ")" : "") + (a("agent.tool_approved") === true ? ", approved by a person" : a("agent.tool_approved") === false ? ", declined by a person" : ""));
        break;
      case "agent.generate":
        section(body, "Feedback from the previous check", a("agent.feedback"));
        section(body, "Answer written", a("agent.answer"));
        section(body, "Prompt size", a("agent.prompt_chars") ? a("agent.prompt_chars") + " characters" : null);
        break;
      case "agent.check":
        section(body, "Verdict", a("agent.passed") ? "passed" : "failed");
        section(body, "Claims the sources do not support", a("agent.claims"));
        break;
      case "agent.respond":
        section(body, "Citations", a("agent.citation_list"));
        break;
      case "agent.recall": section(body, "Facts recalled", a("agent.fact_texts")); break;
      case "agent.load_tools": section(body, "Tools", a("agent.tool_names")); break;
      case "agent.remember": section(body, "Facts saved", a("agent.facts")); break;
      default: break;
    }
    if (n.status === "error") section(body, "Error", n.status_message);
    // what happened underneath: model calls, search, MCP
    const below = descendants(n);
    if (below.length) {
      const sub = el("div", "subspans");
      sub.appendChild(el("h4", "", "Underneath"));
      below.forEach((d) => {
        const bits = [d.name, fmtMs(d.duration_ms)];
        const tin = A(d, "gen_ai.usage.input_tokens");
        if (tin !== undefined) bits.push(tin + " in / " + (A(d, "gen_ai.usage.output_tokens") || 0) + " out tokens");
        if (A(d, "llm.reasoning_tokens")) bits.push(A(d, "llm.reasoning_tokens") + " reasoning");
        if (A(d, "llm.cost_usd") !== undefined) bits.push(fmtCost(A(d, "llm.cost_usd")));
        sub.appendChild(el("div", "", bits.join(" · ")));
      });
      body.appendChild(sub);
    }
  }

  // ===== links and start ========================================================================
  // /ui/observability/#trace=<id> opens a trace, #conversation=<id> lists a conversation.
  function openFromHash() {
    const m = /^#(trace|conversation)=([A-Za-z0-9_.@-]{1,100})$/.exec(location.hash);
    if (!m) return;
    if (m[1] === "trace") openTrace(m[2]);
    else { setScope(m[2]); loadTraces(); }
  }
  window.addEventListener("hashchange", openFromHash);

  function refresh() { loadStatus(); loadTraces(); openFromHash(); }

  $("reload").addEventListener("click", loadTraces);
  $("find-go").addEventListener("click", find);
  $("f-find").addEventListener("keydown", (e) => { if (e.key === "Enter") find(); });
  $("scope-clear").addEventListener("click", () => { setScope(null); $("f-find").value = ""; loadTraces(); });
  ["f-name", "f-status", "f-dur"].forEach((id) => $(id).addEventListener("change", loadTraces));
  kit.mountKeyBar($("keybar"), refresh);
  refresh();
})();
