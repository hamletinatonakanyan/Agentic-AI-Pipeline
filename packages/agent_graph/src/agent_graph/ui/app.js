// Agent UI. Plain JS. All text is set with textContent, never innerHTML.
(function () {
  const API = "/api/agent";
  const $ = (id) => document.getElementById(id);
  let threadId = null;
  let running = false;
  let answerEl = null;

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const showError = (e) => { $("error").textContent = e ? kit.errorText(e) : ""; };
  const post = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const userId = () => $("user-id").value.trim() || null;

  // ===== threads ============================================================================
  async function loadThreads() {
    const q = userId() ? "?user_id=" + encodeURIComponent(userId()) : "";
    const threads = await kit.apiFetch(API + "/threads" + q);
    const box = $("thread-list");
    box.replaceChildren();
    threads.forEach((t) => {
      const b = el("button", t.thread_id === threadId ? "active" : "", t.title || "(new conversation)");
      b.title = t.thread_id + (t.user_id ? " · " + t.user_id : "");
      b.addEventListener("click", () => openThread(t.thread_id));
      box.appendChild(b);
    });
  }

  async function newThread() {
    const t = await kit.apiFetch(API + "/threads", post({ user_id: userId() }));
    await openThread(t.thread_id);
  }

  async function openThread(id) {
    threadId = id;
    const t = await kit.apiFetch(API + "/threads/" + id);
    $("transcript").replaceChildren();
    t.history.forEach((m) => addMessage(m.role, m.content));
    $("approvals").replaceChildren();
    const st = await kit.apiFetch(API + "/threads/" + id + "/state");
    if (st.interrupts.length) showApproval(st.interrupts[0]);
    await loadThreads();
    loadState();
  }

  // ===== transcript =========================================================================
  function addMessage(role, text) {
    const box = el("div", "msg " + (role === "user" ? "user" : "assistant"));
    box.appendChild(el("div", "who", role === "user" ? "You" : "Agent"));
    const body = el("div", "", text);
    box.appendChild(body);
    $("transcript").appendChild(box);
    return { box, body };
  }

  function options() {
    return {
      retrieval: { enabled: $("o-ret").checked, mode: $("o-mode").value, rewrite: $("o-rewrite").value, rerank: $("o-rerank").checked },
      tools: { enabled: $("o-tools").checked },
      check: $("o-check").checked,
      use_cache: $("o-cache").checked ? null : false,
      remember: $("o-remember").checked ? true : null,
    };
  }

  function logLine(text, cls) {
    const line = el("div", cls || "", text);
    $("log").appendChild(line);
    $("log").scrollTop = $("log").scrollHeight;
  }

  // ===== running ============================================================================
  async function send() {
    const message = $("message").value.trim();
    if (!message || running) return;
    if (!threadId) await newThread();
    $("message").value = "";
    addMessage("user", message);
    await stream("/messages/stream", { message: message, user_id: userId(), options: options() });
  }

  async function stream(path, body) {
    running = true;
    showError(null);
    $("busy").textContent = "working...";
    $("log").replaceChildren();
    $("approvals").replaceChildren();
    resetGraph();
    const live = addMessage("assistant", "");
    answerEl = live;
    try {
      await kit.streamSSE(API + "/threads/" + threadId + path, post(body), (_, data) => handle(JSON.parse(data), live));
    } catch (e) {
      showError(e);
      if (!live.body.textContent) live.box.remove();
    } finally {
      running = false;
      $("busy").textContent = "";
      await loadThreads();
      loadState();
    }
  }

  function handle(ev, live) {
    const d = ev.data || {};
    switch (ev.type) {
      case "run_started": logLine("run started" + (d.trace_id ? " (trace " + d.trace_id + ")" : "")); break;
      case "node_start": logLine("▶ " + d.node + (d.subgraph ? " (in answer)" : "")); markNode(d.node, "active"); break;
      case "node_end": logLine("■ " + d.node + (d.error ? " (failed)" : "")); markNode(d.node, d.error ? "" : "done"); break;
      case "token": live.body.textContent += d.text; break;
      case "answer_replaced": live.body.textContent = ""; logLine("self-check failed: rewriting the answer", "error"); break;
      case "tool_call": logLine("tool " + d.name + " -> " + d.status + ": " + (d.content || "").slice(0, 120)); break;
      case "citation": logLine("cites [" + d.marker + "] " + d.source_name + (d.page ? " p." + d.page : "")); break;
      case "interrupt": showApproval(d); break;
      case "error": showError({ message: d.message, status: 0 }); logLine("error: " + d.code, "error"); break;
      case "done": finish(d, live); break;
    }
  }

  function finish(r, live) {
    if (r.status === "interrupted") { live.box.remove(); return; }
    live.body.textContent = r.answer || "";
    const meta = [];
    if (r.route) meta.push("route: " + r.route);
    if (r.check) meta.push(r.check.skipped ? "check: skipped" : "check: " + (r.check.passed ? "passed" : "FAILED") + " (" + r.check.attempts + " attempt(s))");
    if (r.timings.ttft_ms != null) meta.push("first token " + Math.round(r.timings.ttft_ms) + " ms");
    meta.push("total " + Math.round(r.timings.total_ms) + " ms");
    live.box.appendChild(el("div", "meta", meta.join(" · ")));
    if (r.citations.length) {
      const c = el("div", "meta", "Sources: " + r.citations.map((x) => "[" + x.marker + "] " + x.source_name + (x.page ? " p." + x.page : "")).join("; "));
      live.box.appendChild(c);
    }
    if (r.check && !r.check.skipped && !r.check.passed) {
      live.box.appendChild(el("div", "meta error", "Unsupported claims: " + r.check.unsupported_claims.join("; ")));
    }
    r.warnings.forEach((w) => live.box.appendChild(el("div", "meta error", "⚠ " + w)));
  }

  // ===== approvals ==========================================================================
  function showApproval(pending) {
    const box = $("approvals");
    box.replaceChildren();
    const panel = el("div", "approval");
    panel.appendChild(el("strong", "", "The agent wants to run " + pending.calls.length + " tool call(s). Allow?"));
    const choices = {};
    pending.calls.forEach((c) => {
      choices[c.call_id] = false;
      const row = el("div");
      row.appendChild(el("div", "", c.tool));
      row.appendChild(el("pre", "", JSON.stringify(c.arguments, null, 2)));
      row.appendChild(el("div", "meta", "Why you are asked: " + c.reason));
      const sel = el("select");
      [["decline", "Decline"], ["allow", "Allow"]].forEach(([v, t]) => { const o = el("option", "", t); o.value = v; sel.appendChild(o); });
      sel.addEventListener("change", () => { choices[c.call_id] = sel.value === "allow"; });
      row.appendChild(sel);
      panel.appendChild(row);
    });
    const go = el("button", "", "Continue");
    go.addEventListener("click", () => stream("/resume/stream", { decisions: choices }));
    panel.appendChild(go);
    box.appendChild(panel);
  }

  // ===== graph ==============================================================================
  const POS = {
    __start__: [60, 40], begin: [190, 40], recall: [320, 40], load_tools: [450, 40], route: [580, 40],
    retrieve: [440, 150], plan_tools: [740, 150], approve: [740, 260], run_tool: [740, 370],
    collect_tools: [580, 370], answer: [440, 260], respond: [440, 380], remember: [440, 470], __end__: [60, 470],
  };
  let nodeEls = {};

  async function drawGraph() {
    const g = await kit.apiFetch(API + "/graph");
    const svg = $("graph");
    svg.replaceChildren();
    const ns = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs) => { const n = document.createElementNS(ns, tag); Object.keys(attrs || {}).forEach((k) => n.setAttribute(k, attrs[k])); return n; };
    const defs = mk("defs");
    const marker = mk("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto" });
    marker.appendChild(mk("path", { d: "M0,0 L10,5 L0,10 z" }));
    defs.appendChild(marker); svg.appendChild(defs);
    const W = 110, H = 34;
    g.edges.forEach((e) => {
      const a = POS[e.source], b = POS[e.target];
      if (!a || !b) return;
      const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
      const sx = a[0] + (dx / len) * (Math.abs(dx) > Math.abs(dy) ? W / 2 : 0), sy = a[1] + (dy / len) * (Math.abs(dx) > Math.abs(dy) ? 0 : H / 2);
      const tx = b[0] - (dx / len) * (Math.abs(dx) > Math.abs(dy) ? W / 2 : 0), ty = b[1] - (dy / len) * (Math.abs(dx) > Math.abs(dy) ? 0 : H / 2);
      svg.appendChild(mk("line", { x1: sx, y1: sy, x2: tx, y2: ty, class: "edge" + (e.conditional ? " cond" : ""), "marker-end": "url(#arrow)" }));
    });
    nodeEls = {};
    g.nodes.forEach((n) => {
      const p = POS[n.id]; if (!p) return;
      const grp = mk("g", { class: "node " + n.kind, transform: "translate(" + (p[0] - W / 2) + "," + (p[1] - H / 2) + ")" });
      grp.appendChild(mk("rect", { width: W, height: H }));
      const t = mk("text", { x: W / 2, y: H / 2 }); t.textContent = n.label; grp.appendChild(t);
      svg.appendChild(grp);
      nodeEls[n.id] = grp;
    });
  }
  function markNode(name, state) {
    const n = nodeEls[name]; if (!n) return;
    n.classList.remove("active", "done");
    if (state) n.classList.add(state);
  }
  function resetGraph() { Object.keys(nodeEls).forEach((k) => markNode(k, "")); }

  // ===== state and history ==================================================================
  async function loadState() {
    if (!threadId) { $("state-json").textContent = "Open a conversation first."; $("history").replaceChildren(); return; }
    try {
      const st = await kit.apiFetch(API + "/threads/" + threadId + "/state");
      $("state-note").textContent = st.next.length ? "paused or running before: " + st.next.join(", ") : "finished";
      $("state-json").textContent = JSON.stringify(st.values, null, 2);
      const hist = await kit.apiFetch(API + "/threads/" + threadId + "/history");
      const t = $("history");
      t.replaceChildren();
      const head = t.createTHead().insertRow();
      ["step", "what", "next", "checkpoint", ""].forEach((h) => head.appendChild(el("th", "", h)));
      const body = t.createTBody();
      hist.forEach((h) => {
        const tr = body.insertRow();
        tr.insertCell().textContent = h.step == null ? "" : h.step;
        tr.insertCell().textContent = h.summary;
        tr.insertCell().textContent = h.next.join(", ");
        tr.insertCell().textContent = h.checkpoint_id.slice(0, 12);
        const b = el("button", "", "Fork from here");
        b.addEventListener("click", () => fork(h.checkpoint_id));
        tr.insertCell().appendChild(b);
      });
    } catch (e) { showError(e); }
  }

  async function fork(checkpointId) {
    if (running) return;
    const message = $("fork-message").value.trim() || null;
    document.querySelector('[data-tab="chat"]').click();
    if (message) addMessage("user", message);
    await stream("/fork/stream", { checkpoint_id: checkpointId, message: message, options: options() });
  }

  // ===== memory =============================================================================
  async function loadFacts() {
    const user = $("m-user").value.trim();
    if (!user) return;
    const q = $("m-query").value.trim();
    const facts = await kit.apiFetch(API + "/memory/" + encodeURIComponent(user) + (q ? "?query=" + encodeURIComponent(q) : ""));
    const t = $("facts");
    t.replaceChildren();
    const head = t.createTHead().insertRow();
    ["fact", q ? "score" : "", ""].forEach((h) => head.appendChild(el("th", "", h)));
    const body = t.createTBody();
    facts.forEach((f) => {
      const tr = body.insertRow();
      tr.insertCell().textContent = f.text;
      tr.insertCell().textContent = f.score == null ? "" : f.score.toFixed(2);
      const b = el("button", "", "Delete");
      b.addEventListener("click", async () => {
        try { await kit.apiFetch(API + "/memory/" + encodeURIComponent(user) + "/" + f.id, { method: "DELETE" }); loadFacts(); } catch (e) { showError(e); }
      });
      tr.insertCell().appendChild(b);
    });
  }

  async function addFact() {
    const user = $("m-user").value.trim(), text = $("m-text").value.trim();
    if (!user || !text) return;
    try { await kit.apiFetch(API + "/memory/" + encodeURIComponent(user), post({ text: text })); $("m-text").value = ""; loadFacts(); } catch (e) { showError(e); }
  }

  // ===== reasoning switch ===================================================================
  async function loadReasoning() {
    const r = await kit.apiFetch(API + "/reasoning");
    $("o-reasoning").checked = r.use_reasoning;
    $("reasoning-note").textContent = r.overridden ? "(changed here, lasts until restart)" : "";
  }
  async function setReasoning() {
    try {
      const r = await kit.apiFetch(API + "/reasoning", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ use_reasoning: $("o-reasoning").checked }) });
      $("reasoning-note").textContent = "(" + r.effort + " effort, changed here, lasts until restart)";
    } catch (e) { showError(e); }
  }

  // ===== start ==============================================================================
  async function refresh() {
    showError(null);
    try {
      const s = await kit.apiFetch(API + "/status");
      $("status").textContent = "approval: " + s.approval + " · tool rounds ≤ " + s.max_tool_rounds + " · check retries ≤ " + s.max_check_retries +
        " · caches " + (s.use_cache ? "on" : "off") + " · remember " + (s.auto_remember ? "on" : "off") + " · " + s.tools_available + " tool(s)";
      await Promise.all([loadThreads(), drawGraph(), loadReasoning()]);
    } catch (e) { showError(e); }
  }

  kit.initTabs($("tabs").parentElement);
  kit.mountKeyBar($("keybar"), refresh);
  $("new-thread").addEventListener("click", () => newThread().catch(showError));
  $("send").addEventListener("click", () => send().catch(showError));
  $("message").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send().catch(showError); });
  $("user-id").addEventListener("change", () => loadThreads().catch(showError));
  $("o-reasoning").addEventListener("change", setReasoning);
  $("reload-state").addEventListener("click", loadState);
  $("m-load").addEventListener("click", () => loadFacts().catch(showError));
  $("m-add").addEventListener("click", addFact);
  refresh();
})();
