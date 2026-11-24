// Retrieval UI: one query, one or two sets of options, results side by side.
// Plain JS. All text is set with textContent, never innerHTML.
(function () {
  const API = "/api/retrieval";
  const $ = (id) => document.getElementById(id);
  let strategies = null;
  const panels = [];

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function showError(e) { $("error").textContent = e ? kit.errorText(e) : ""; }
  const fmt = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms");

  // ----- an options panel ------------------------------------------------------------------
  function select(values, current) {
    const s = el("select");
    values.forEach((v) => s.appendChild(new Option(v, v)));
    s.value = current;
    return s;
  }
  function field(label, input, inline) {
    const l = el("label", inline ? "inline" : "");
    if (inline) { l.appendChild(input); l.appendChild(el("span", "", label)); }
    else { l.appendChild(el("span", "", label)); l.appendChild(input); }
    return l;
  }
  function slider(min, max, value) {
    const wrap = el("div", "row");
    const r = el("input"); r.type = "range"; r.min = min; r.max = max; r.value = value; r.style.flex = "1";
    const out = el("span", "muted", String(value));
    r.addEventListener("input", () => { out.textContent = r.value; });
    wrap.append(r, out);
    return { wrap, input: r };
  }
  function parseValue(raw) {
    const t = raw.trim();
    try { return JSON.parse(t); } catch (e) { return t; }   // 2025, true, [1,2], {"gte": 3}, or text
  }

  function makePanel(title, defaults) {
    const box = el("div", "panel");
    box.appendChild(el("h3", "", title));
    const mode = select(strategies.modes, defaults.mode);
    const rewrite = select(strategies.rewrite_strategies, defaults.rewrite);
    const rerank = el("input"); rerank.type = "checkbox"; rerank.checked = defaults.rerank;
    const model = select(strategies.rerank_models.map((m) => m.name), strategies.defaults.rerank_model);
    if (!strategies.rerank_models.some((m) => m.name === strategies.defaults.rerank_model)) {
      model.appendChild(new Option(strategies.defaults.rerank_model, strategies.defaults.rerank_model));
      model.value = strategies.defaults.rerank_model;
    }
    const topK = slider(1, 30, strategies.defaults.top_k);
    const cands = slider(1, 200, strategies.defaults.candidates);
    const filters = el("div", "filters");
    const add = el("button", "", "+ filter");
    add.type = "button";
    function addRow() {
      const row = el("div", "frow");
      const k = el("input"); k.placeholder = "field, e.g. source_name"; 
      const v = el("input"); v.placeholder = 'value, e.g. a.pdf, 2025, [1,2], {"gte": 3}';
      const x = el("button", "", "x"); x.type = "button";
      x.addEventListener("click", () => row.remove());
      row.append(k, v, x);
      filters.appendChild(row);
    }
    add.addEventListener("click", addRow);
    box.append(
      field("Search mode", mode), field("Query rewriting (uses the LLM)", rewrite),
      field("Rerank with a cross-encoder", rerank, true), field("Rerank model", model),
      field("top_k: results to return", topK.wrap), field("candidates: fetched before reranking", cands.wrap),
      el("div", "muted", "Filters (all must match)"), filters, add,
    );
    $("panels").appendChild(box);
    return {
      box,
      read() {
        const f = {};
        filters.querySelectorAll(".frow").forEach((row) => {
          const [k, v] = row.querySelectorAll("input");
          if (k.value.trim() && v.value.trim()) f[k.value.trim()] = parseValue(v.value);
        });
        const topKv = parseInt(topK.input.value, 10);
        return {
          mode: mode.value, rewrite: rewrite.value, rerank: rerank.checked,
          rerank_model: rerank.checked ? model.value : null,
          top_k: topKv, candidates: Math.max(parseInt(cands.input.value, 10), topKv), filters: f,
        };
      },
    };
  }

  // ----- running ---------------------------------------------------------------------------------
  async function runOne(opts) {
    const body = Object.assign({ query: $("query").value.trim(), collection: $("collection").value || null }, opts);
    return kit.apiFetch(API + "/retrieve", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  }

  function hitCard(h, other) {
    const card = el("div", "hit");
    if (other) {
      const twin = other.hits.find((o) => o.chunk_id === h.chunk_id);
      card.classList.add(twin ? "both" : "only");
    }
    const meta = el("div", "meta");
    meta.appendChild(el("b", "", "#" + h.rank));
    if (other) {
      const twin = other.hits.find((o) => o.chunk_id === h.chunk_id);
      meta.appendChild(el("span", "", twin ? "also in the other (#" + twin.rank + ")" : "only here"));
    }
    meta.appendChild(el("span", "", "score " + h.score.toFixed(4)));
    if (h.rerank_score !== null) meta.appendChild(el("span", "", "rerank " + h.rerank_score.toFixed(3)));
    const p = h.payload;
    meta.appendChild(el("span", "", (p.source_name || "?") + (p.page ? " p." + p.page : "") + (p.label ? " (" + p.label + ")" : "")));
    if (p.section_path) meta.appendChild(el("span", "", p.section_path));
    card.appendChild(meta);
    const text = h.text.length > 320 ? h.text.slice(0, 320) + "..." : h.text;
    const body = el("div", "body", text);
    if (h.text.length > 320) {
      body.style.cursor = "pointer"; body.title = "click to expand";
      let open = false;
      body.addEventListener("click", () => { open = !open; body.textContent = open ? h.text : text; });
    }
    card.appendChild(body);
    return card;
  }

  function resultCard(title, res, other) {
    const card = el("div", "result");
    card.appendChild(el("h3", "", title));
    const o = res.timings;
    card.appendChild(el("div", "timings",
      "rewrite " + fmt(o.rewrite_ms) + " | search " + fmt(o.retrieve_ms) + " | rerank " + fmt(o.rerank_ms) +
      " | total " + fmt(o.total_ms) + " | " + res.mode + (res.rerank ? " + rerank" : "") +
      (res.rewrite !== "none" ? " + " + res.rewrite : "")));
    if (res.rewritten_queries.length > 1) {
      const q = el("div", "queries");
      res.rewritten_queries.forEach((x) => q.appendChild(el("div", "", x)));
      card.appendChild(q);
    }
    res.warnings.forEach((w) => card.appendChild(el("div", "warn", w)));
    if (!res.hits.length) card.appendChild(el("p", "muted", "No results."));
    res.hits.forEach((h) => card.appendChild(hitCard(h, other)));
    if (res.trace_id) card.appendChild(el("div", "muted", "trace " + res.trace_id));
    return card;
  }

  async function run() {
    showError(null);
    if (!$("query").value.trim()) return;
    const compare = $("compare").checked;
    const active = compare ? panels : [panels[0]];
    const out = $("results");
    out.replaceChildren(el("p", "muted", "searching..."));
    $("run").disabled = true;
    const settled = await Promise.allSettled(active.map((p) => runOne(p.read())));
    $("run").disabled = false;
    out.replaceChildren();
    settled.forEach((s, i) => {
      const title = compare ? "Settings " + "AB"[i] : "Results";
      if (s.status === "rejected") {
        const c = el("div", "result"); c.appendChild(el("h3", "", title));
        c.appendChild(el("p", "error", kit.errorText(s.reason)));
        out.appendChild(c);
        return;
      }
      const other = compare && settled[1 - i].status === "fulfilled" ? settled[1 - i].value : null;
      out.appendChild(resultCard(title, s.value, other));
    });
  }

  // ----- setup ---------------------------------------------------------------------------------------
  function showSpec() {
    const c = (strategies.collections || []).find((x) => x.name === $("collection").value);
    $("spec").textContent = c && c.spec
      ? c.points + " chunks | " + c.spec.embedding_model + " | BM25 language: " + c.spec.sparse_language
      : (c ? "This collection was not created by ingest." : "");
  }

  async function load() {
    try {
      strategies = await kit.apiFetch(API + "/strategies");
    } catch (e) { showError(e); $("run").disabled = true; return; }
    showError(null);
    $("status").textContent = "Default rerank model: " + strategies.defaults.rerank_model +
      " (downloads the first time it is used)";
    const sel = $("collection");
    const keep = sel.value || kit.getDomain() || strategies.defaults.collection;
    sel.replaceChildren();
    strategies.collections.forEach((c) => sel.appendChild(new Option(c.name, c.name)));
    if (strategies.collections.some((c) => c.name === keep)) sel.value = keep;
    showSpec();
    if (!panels.length) {
      panels.push(makePanel("Settings A", { mode: "hybrid", rewrite: "none", rerank: false }));
      panels.push(makePanel("Settings B", { mode: "hybrid", rewrite: "none", rerank: true }));
      panels[1].box.hidden = true;
    }
    $("run").disabled = !strategies.collections.length;
    if (!strategies.collections.length) showError(new Error("No collections yet. Ingest some files first."));
  }

  $("compare").addEventListener("change", () => { panels[1].box.hidden = !$("compare").checked; });
  $("collection").addEventListener("change", showSpec);
  // The main page's Domain box picks the collection here too.
  kit.onDomainChange((domain) => {
    if (domain && [...$("collection").options].some((o) => o.value === domain)) { $("collection").value = domain; showSpec(); }
  });
  $("run").addEventListener("click", run);
  $("query").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  kit.mountKeyBar($("keybar"), load);
  load();
})();
