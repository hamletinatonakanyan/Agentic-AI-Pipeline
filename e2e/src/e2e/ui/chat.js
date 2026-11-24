// The chat. Plain JS. All text is set with textContent, never innerHTML.
// It talks to the agent's own API (/api/agent): there is no separate chat backend.
(function () {
  const API = "/api/agent";
  const $ = (id) => document.getElementById(id);
  let threadId = null;
  let running = false;
  let agentReady = false;
  let hasTraceTab = false;

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const post = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const showError = (e) => { $("c-error").textContent = e ? (e.status === 401 ? "Enter your access key." : (e.message || String(e))) : ""; };

  // The user id is a convenience, not a secret: it only selects whose memory is used.
  function savedUser() { try { return localStorage.getItem("chat_user") || ""; } catch (e) { return ""; } }
  function saveUser(v) { try { localStorage.setItem("chat_user", v); } catch (e) { /* storage blocked */ } }
  const userId = () => $("c-user").value.trim() || null;

  // ===== status: can the agent answer at all? ==============================================
  document.addEventListener("status", (ev) => {
    const comps = ev.detail;
    const agent = comps.find((c) => c.name === "agent");
    const why = [];
    if (!agent || !agent.enabled) why.push("The agent is turned off (E2E_ENABLE_AGENT).");
    else if (!agent.configured) why.push("The agent needs: " + agent.missing.join(", ") + ". Add it to .env and restart.");
    const ret = comps.find((c) => c.name === "retrieval");
    const qd = comps.find((c) => c.name === "qdrant");
    if (qd && qd.reachable === false) why.push("Qdrant is not reachable, so searching documents will fail. Run `make up`.");
    else if (ret && !ret.enabled) why.push("Retrieval is turned off, so the chat cannot search documents.");
    agentReady = !!agent && agent.enabled;
    hasTraceTab = comps.some((c) => c.name === "observability" && c.enabled && c.ui_path);
    $("chat-blocked").hidden = why.length === 0;
    $("chat-blocked").textContent = why.join(" ");
    $("c-send").disabled = !agentReady || (agent && !agent.configured);
    if (agentReady) { loadThreads().catch(() => {}); loadReasoning().catch(() => {}); }
  });

  // ===== threads ============================================================================
  async function loadThreads() {
    const q = userId() ? "?user_id=" + encodeURIComponent(userId()) : "";
    let threads = await kit.apiFetch(API + "/threads" + q);
    const domain = kit.getDomain();
    if ($("c-only-domain").checked && domain) threads = threads.filter((t) => !t.collection || t.collection === domain);
    const box = $("c-threads");
    box.replaceChildren();
    threads.forEach((t) => {
      const b = el("button", t.thread_id === threadId ? "active" : "", t.title || "(new conversation)");
      if (t.collection) b.appendChild(el("span", "tchip", t.collection));
      b.title = t.thread_id + (t.collection ? " · domain " + t.collection : "");
      b.addEventListener("click", () => openThread(t.thread_id).catch(showError));
      box.appendChild(b);
    });
  }

  async function newThread() {
    const t = await kit.apiFetch(API + "/threads", post({ user_id: userId() }));
    await openThread(t.thread_id);
  }

  async function openThread(id) {
    threadId = id;
    showError(null);
    const t = await kit.apiFetch(API + "/threads/" + id);
    $("c-transcript").replaceChildren();
    t.history.forEach((m) => {
      const live = message(m.role, m.content);
      if (m.run) decorate(live, m.run.response);   // sources, tool calls, timings and trace, as they were
    });
    if (t.collection && t.collection !== kit.getDomain()) kit.setDomain(t.collection);  // a conversation keeps its domain
    $("c-approvals").replaceChildren();
    const st = await kit.apiFetch(API + "/threads/" + id + "/state");
    if (st.interrupts.length) approval(st.interrupts[0]);
    loadThreads().catch(() => {});
  }

  // ===== messages ===========================================================================
  function message(role, text) {
    const box = el("div", "cmsg " + (role === "user" ? "user" : "assistant"));
    box.appendChild(el("div", "who", role === "user" ? "You" : "Agent"));
    const body = el("div", "", text);
    box.appendChild(body);
    $("c-transcript").appendChild(box);
    box.scrollIntoView({ block: "end" });
    return { box, body };
  }

  // Turns "[1]" in the answer into a small button that opens that source. Text nodes only.
  function renderAnswer(body, text, citations, passages, sourcesBox) {
    body.replaceChildren();
    const known = new Set(citations.map((c) => c.marker));
    text.split(/(\[\d+\])/).forEach((part) => {
      const m = /^\[(\d+)\]$/.exec(part);
      const n = m ? Number(m[1]) : 0;
      if (m && known.has(n)) {
        const b = el("button", "cite", String(n));
        const p = passages[n - 1];
        b.title = p ? (p.source_name || "source") + (p.page ? ", page " + p.page : "") + ": " + p.text.slice(0, 300) : "";
        b.addEventListener("click", () => {
          sourcesBox.open = true;
          sourcesBox.querySelectorAll(".src").forEach((s) => s.classList.toggle("hit", s.dataset.n === String(n)));
        });
        body.appendChild(b);
      } else if (part) {
        body.appendChild(document.createTextNode(part));
      }
    });
  }

  function sourcesPanel(citations, passages) {
    if (!citations.length) return null;
    const d = el("details", "sources");
    d.appendChild(el("summary", "", "Sources (" + citations.length + ")"));
    citations.forEach((c) => {
      const p = passages[c.marker - 1];
      const row = el("div", "src");
      row.dataset.n = String(c.marker);
      row.appendChild(el("div", "", "[" + c.marker + "] " + (c.source_name || "source") + (c.page ? ", page " + c.page : "")));
      if (p) row.appendChild(el("pre", "", p.text.slice(0, 700)));
      d.appendChild(row);
    });
    return d;
  }

  function toolCalls(calls) {
    const out = [];
    calls.forEach((c) => {
      const d = el("details", "toolcall");
      const s = el("summary");
      s.appendChild(el("span", "badge " + (c.status === "ok" ? "ok" : c.status === "declined" ? "warn" : "bad"), c.status));
      s.appendChild(document.createTextNode(c.name));
      d.appendChild(s);
      d.appendChild(el("pre", "", "arguments: " + JSON.stringify(c.arguments)));
      d.appendChild(el("pre", "", c.content));
      out.push(d);
    });
    return out;
  }

  function finish(r, live) {
    if (r.status === "interrupted") { live.box.remove(); return; }
    decorate(live, r);
  }

  // Everything under an answer: sources, tool calls, timings, and links to its trace.
  function decorate(live, r) {
    const box = live.box;
    const src = sourcesPanel(r.citations, r.passages);
    renderAnswer(live.body, r.answer || "", r.citations, r.passages, src || el("details"));
    toolCalls(r.tool_calls).forEach((d) => box.appendChild(d));
    if (src) box.appendChild(src);
    const t = r.timings, parts = [];
    if (r.route) parts.push(r.route);
    if (t.ttft_ms != null) parts.push("first token " + Math.round(t.ttft_ms) + " ms");
    if (t.retrieval_ms != null) parts.push("search " + Math.round(t.retrieval_ms) + " ms");
    parts.push("total " + Math.round(t.total_ms) + " ms");
    const meta = el("div", "meta", parts.join(" · "));
    if (r.trace_id) {
      meta.appendChild(document.createTextNode(" · "));
      if (hasTraceTab) {
        const b = el("button", "cite", "trace " + r.trace_id.slice(0, 8));
        b.style.verticalAlign = "baseline";
        b.addEventListener("click", () => openObs("#trace=" + encodeURIComponent(r.trace_id)));
        meta.appendChild(b);
      } else meta.appendChild(document.createTextNode("trace " + r.trace_id.slice(0, 8)));
    }
    box.appendChild(meta);
    const metaSpan = meta;
    if (r.trace_id && hasTraceTab) {
      const links = el("div", "links");
      const conv = el("button", "", "all runs of this conversation");
      conv.title = "Open every run of this conversation in Observability";
      conv.addEventListener("click", () => openObs("#conversation=" + encodeURIComponent(r.thread_id)));
      links.appendChild(conv);
      box.appendChild(links);
      // tokens and an estimated cost, from the trace (only when observability is on)
      kit.apiFetch("/api/obs/traces/" + encodeURIComponent(r.trace_id)).then((tr) => {
        const s = tr.summary, bits = [s.llm_calls + " model call(s)", s.input_tokens + " in / " + s.output_tokens + " out tokens"];
        if (s.cost_usd != null) bits.push("≈ $" + (s.cost_usd < 0.01 ? s.cost_usd.toFixed(6) : s.cost_usd.toFixed(4)));
        metaSpan.appendChild(document.createTextNode(" · " + bits.join(" · ")));
      }).catch(() => {});
    }
    if (r.check && !r.check.skipped && !r.check.passed) {
      box.appendChild(el("div", "meta warn", "The self-check found claims the sources do not support: " + r.check.unsupported_claims.join("; ")));
    }
    r.warnings.forEach((w) => box.appendChild(el("div", "meta warn", "⚠ " + w)));
  }

  // Open the Observability tab at a trace (#trace=<id>) or a conversation (#conversation=<id>).
  function openObs(hash) {
    const frame = document.querySelector('iframe[title="observability"]');
    if (!frame || !window.showTab) return;
    window.showTab("sec-observability");
    frame.src = "/ui/observability/" + hash;
  }

  // ===== approval ===========================================================================
  function approval(pending) {
    const box = $("c-approvals");
    box.replaceChildren();
    const panel = el("div", "capproval");
    panel.appendChild(el("strong", "", "The agent wants to run " + pending.calls.length + " tool call(s). Allow?"));
    const choices = {};
    pending.calls.forEach((c) => {
      choices[c.call_id] = false;
      const row = el("div");
      row.appendChild(el("div", "", c.tool));
      row.appendChild(el("pre", "", JSON.stringify(c.arguments, null, 2)));
      row.appendChild(el("div", "muted", "Why you are asked: " + c.reason));
      const sel = el("select");
      [["decline", "Decline"], ["allow", "Allow"]].forEach(([v, t]) => { const o = el("option", "", t); o.value = v; sel.appendChild(o); });
      sel.addEventListener("change", () => { choices[c.call_id] = sel.value === "allow"; });
      row.appendChild(sel);
      panel.appendChild(row);
    });
    const go = el("button", "", "Continue");
    go.addEventListener("click", () => run("/resume/stream", { decisions: choices }));
    panel.appendChild(go);
    box.appendChild(panel);
  }

  // ===== running ============================================================================
  function options() {
    const topk = parseInt($("c-topk").value, 10);
    return {
      retrieval: {
        enabled: $("c-ret").checked, mode: $("c-mode").value, rerank: $("c-rerank").checked,
        collection: $("c-coll").value.trim() || kit.getDomain() || null, top_k: topk > 0 ? topk : null,
      },
      tools: { enabled: $("c-tools").checked },
    };
  }

  async function send() {
    const text = $("c-message").value.trim();
    if (!text || running) return;
    if (!threadId) await newThread();
    $("c-message").value = "";
    message("user", text);
    await run("/messages/stream", { message: text, user_id: userId(), options: options() });
  }

  async function run(path, body) {
    running = true;
    showError(null);
    $("c-busy").textContent = "working...";
    $("c-approvals").replaceChildren();
    const live = message("assistant", "");
    let streamed = false;
    try {
      await kit.streamSSE(API + "/threads/" + threadId + path, post(body), (_, data) => {
        const ev = JSON.parse(data), d = ev.data || {};
        if (ev.type === "node_start" && d.node && !streamed) $("c-busy").textContent = d.node.replace(/_/g, " ") + "...";
        else if (ev.type === "token") { streamed = true; live.body.textContent += d.text; }
        else if (ev.type === "answer_replaced") { live.body.textContent = ""; $("c-busy").textContent = "double-checking the answer..."; }
        else if (ev.type === "interrupt") approval(d);
        else if (ev.type === "error") showError({ message: d.message });
        else if (ev.type === "done") finish(d, live);
      });
    } catch (e) {
      showError(e);
    } finally {
      if (!live.body.textContent && !live.box.querySelector(".meta")) live.box.remove();
      running = false;
      $("c-busy").textContent = "";
      loadThreads().catch(() => {});
    }
  }

  // ===== the global reasoning switch =========================================================
  async function loadReasoning() {
    const r = await kit.apiFetch(API + "/reasoning");
    $("c-reasoning").checked = r.use_reasoning;
    $("c-reasoning-note").textContent = r.overridden ? "changed here, lasts until restart" : "";
  }
  async function setReasoning() {
    try {
      const r = await kit.apiFetch(API + "/reasoning", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ use_reasoning: $("c-reasoning").checked }) });
      $("c-reasoning-note").textContent = r.effort + " effort, lasts until restart";
    } catch (e) { showError(e); }
  }

  // The domain shows as the placeholder, and a conversation's domain follows the page.
  function showDomain() { $("c-coll").placeholder = kit.getDomain() || "docs (default)"; }
  kit.onDomainChange(() => { showDomain(); loadThreads().catch(() => {}); });
  $("c-only-domain").addEventListener("change", () => loadThreads().catch(showError));
  showDomain();

  $("c-user").value = savedUser();
  $("c-user").addEventListener("change", () => { saveUser($("c-user").value.trim()); threadId = null; $("c-transcript").replaceChildren(); loadThreads().catch(showError); });
  $("c-new").addEventListener("click", () => newThread().catch(showError));
  $("c-send").addEventListener("click", () => send().catch(showError));
  $("c-message").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send().catch(showError); });
  $("c-reasoning").addEventListener("change", setReasoning);
})();
