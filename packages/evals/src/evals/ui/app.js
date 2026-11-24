// Evals UI. Plain JS. All text is set with textContent, never innerHTML.
(function () {
  const API = "/api/evals";
  const $ = (id) => document.getElementById(id);
  let targets = null;
  let reviewing = null;

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const showError = (e) => { $("error").textContent = e ? kit.errorText(e) : ""; };
  const fmtTime = (ts) => new Date(ts * 1000).toLocaleString();
  const fmtNum = (v) => (v === null || v === undefined ? "-" : Number(v).toFixed(3));
  const fmtMs = (v) => (v === null || v === undefined ? "-" : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const json = (method, body, extra) => ({
    method, headers: Object.assign({ "Content-Type": "application/json" }, extra), body: JSON.stringify(body),
  });

  function table(target, columns, rows, onRow) {
    target.replaceChildren();
    if (!rows.length) return;
    const head = target.createTHead().insertRow();
    columns.forEach((c) => head.appendChild(el("th", "", c.label)));
    const body = target.createTBody();
    rows.forEach((r) => {
      const tr = body.insertRow();
      columns.forEach((c) => {
        const td = tr.insertCell();
        const v = c.get(r, tr);
        if (v instanceof Node) td.appendChild(v); else td.textContent = v === null || v === undefined ? "" : String(v);
      });
      if (onRow) { tr.classList.add("clickable"); tr.addEventListener("click", () => onRow(r)); }
    });
  }

  async function pollJob(id, onProgress) {
    for (;;) {
      const job = await kit.apiFetch(API + "/jobs/" + encodeURIComponent(id));
      onProgress(job);
      if (job.state === "completed" || job.state === "failed") return job;
      await sleep(1000);
    }
  }

  // ===== datasets ==============================================================================
  async function loadDatasets() {
    const rows = await kit.apiFetch(API + "/datasets");
    table($("datasets"), [
      { label: "name", get: (d) => d.name },
      { label: "status", get: (d) => d.status },
      { label: "accepted", get: (d) => d.counts.accepted },
      { label: "draft", get: (d) => d.counts.draft },
      { label: "rejected", get: (d) => d.counts.rejected },
      { label: "created", get: (d) => fmtTime(d.created_at) },
      { label: "", get: (d) => {
        const b = el("button", "", "delete");
        b.addEventListener("click", async (e) => {
          e.stopPropagation();
          if (!confirm("Delete dataset " + d.name + "?")) return;
          try { await kit.apiFetch(API + "/datasets/" + d.id, { method: "DELETE" }); if (reviewing === d.id) closeReview(); refreshAll(); }
          catch (err) { showError(err); }
        });
        return b;
      } },
    ], rows, (d) => openReview(d.id));
    return rows;
  }

  function closeReview() { reviewing = null; $("review-box").hidden = true; }

  async function openReview(id) {
    reviewing = id;
    const ds = await kit.apiFetch(API + "/datasets/" + encodeURIComponent(id));
    $("review-box").hidden = false;
    $("review-title").textContent = "Review: " + ds.name;
    $("review-counts").textContent = ds.counts.accepted + " accepted, " + ds.counts.draft + " draft, " + ds.counts.rejected + " rejected. Only accepted questions are used in runs.";
    const box = $("items");
    box.replaceChildren();
    ds.items.forEach((item) => box.appendChild(itemCard(ds.id, item)));
  }

  function itemCard(datasetId, item) {
    const card = el("div", "item " + item.status);
    const q = el("textarea"); q.value = item.question;
    const a = el("textarea"); a.value = item.reference_answer;
    card.appendChild(el("div", "muted", item.status + " | id " + item.id +
      (item.relevant_chunk_ids.length ? " | evidence chunk " + item.relevant_chunk_ids.join(", ") : " | no evidence chunk")));
    card.appendChild(el("label", "muted", "Question"));
    card.appendChild(q);
    card.appendChild(el("label", "muted", "Reference answer (needed for completeness and answer accuracy)"));
    card.appendChild(a);
    card.appendChild(el("label", "muted", "Relevant chunk ids, comma-separated (needed for recall, precision, hit rate, MRR, nDCG, source accuracy)"));
    const rel = el("input"); rel.value = item.relevant_chunk_ids.join(", "); rel.style.width = "100%";
    card.appendChild(rel);
    item.evidence.forEach((ev) => {
      card.appendChild(el("div", "muted", "Evidence: " + (ev.source_name || "?") + (ev.page ? " p." + ev.page : "")));
      card.appendChild(el("div", "evidence", ev.text));
    });
    const row = el("div", "row");
    async function patch(body) {
      try { await kit.apiFetch(API + "/datasets/" + datasetId + "/items/" + item.id, json("PATCH", body)); showError(null); await openReview(datasetId); loadDatasets(); }
      catch (e) { showError(e); }
    }
    const edits = () => ({
      question: q.value.trim(), reference_answer: a.value,
      relevant_chunk_ids: rel.value.split(",").map((x) => x.trim()).filter(Boolean),
    });
    [["Accept", "accepted"], ["Reject", "rejected"]].forEach(([label, status]) => {
      const b = el("button", "", label);
      b.addEventListener("click", () => patch(Object.assign(edits(), { status })));
      row.appendChild(b);
    });
    const save = el("button", "", "Save edits");
    save.addEventListener("click", () => patch(edits()));
    row.appendChild(save);
    card.appendChild(row);
    return card;
  }

  $("accept-all").addEventListener("click", async () => {
    if (!reviewing) return;
    try {
      const ds = await kit.apiFetch(API + "/datasets/" + reviewing);
      for (const it of ds.items.filter((i) => i.status === "draft")) {
        await kit.apiFetch(API + "/datasets/" + reviewing + "/items/" + it.id, json("PATCH", { status: "accepted" }));
      }
      await openReview(reviewing); loadDatasets();
    } catch (e) { showError(e); }
  });

  $("up-go").addEventListener("click", async () => {
    const f = $("up-file").files[0];
    if (!f) { showError(new Error("Choose a file first.")); return; }
    try {
      const form = new FormData();
      form.append("file", f);
      if ($("up-name").value.trim()) form.append("name", $("up-name").value.trim());
      const ds = await kit.apiFetch(API + "/datasets", { method: "POST", body: form });
      showError(null); $("up-file").value = ""; await refreshAll(); openReview(ds.id);
    } catch (e) {
      const lines = e.details && e.details.lines ? " Lines: " + Object.entries(e.details.lines).map(([n, m]) => n + " (" + m + ")").join("; ") : "";
      showError(new Error(kit.errorText(e) + lines));
    }
  });

  $("gen-go").addEventListener("click", async () => {
    try {
      $("gen-go").disabled = true;
      const body = {
        collection: $("gen-collection").value, n: parseInt($("gen-n").value, 10) || 10,
        min_chars: parseInt($("gen-min").value, 10) || 0, name: $("gen-name").value.trim() || null,
      };
      const accepted = await kit.apiFetch(API + "/datasets/generate", json("POST", body, { "Idempotency-Key": crypto.randomUUID() }));
      const job = await pollJob(accepted.job_id, (j) => { $("gen-state").textContent = "drafting " + j.progress + " / " + j.total + "..."; });
      if (job.state === "failed") { $("gen-state").textContent = ""; showError(new Error(job.error)); return; }
      $("gen-state").textContent = "drafted " + job.result.drafted + (job.result.failed ? " (" + job.result.failed + " failed)" : "");
      showError(null); await refreshAll(); openReview(job.result.dataset_id);
    } catch (e) { showError(e); } finally { $("gen-go").disabled = false; }
  });

  // ===== runs ==============================================================================
  function targetBody() {
    if ($("r-kind").value === "http") {
      const cfg = { kind: "http", url: $("r-url").value.trim(), headers_refs: {} };
      const h = $("r-header").value.trim();
      if (h) { const [name, ref] = h.split("="); cfg.headers_refs[name.trim()] = (ref || "").trim(); }
      return cfg;
    }
    let options = {};
    const raw = $("r-options").value.trim();
    if (raw) { try { options = JSON.parse(raw); } catch (e) { throw new Error("Target options are not valid JSON."); } }
    // The active domain is the default collection of a target, unless the options name one.
    if (kit.getDomain() && !("collection" in options) && ["retrieval", "agent", "retrieval_http"].includes($("r-preset").value)) {
      options = Object.assign({ collection: kit.getDomain() }, options);
    }
    return { kind: "preset", name: $("r-preset").value, options };
  }
  function runBody() {
    const ks = $("r-ks").value.split(",").map((x) => parseInt(x, 10)).filter((x) => x > 0);
    return {
      dataset_id: $("r-dataset").value, target: targetBody(),
      config: { ks: ks.length ? ks : null, metrics: chosenMetrics() },
    };
  }

  // ===== the metric picker ==================================================================
  const GROUPS = { retrieval: "Retrieval: did the search find the right chunks?", answer: "Answer quality (judged by the LLM)", citations: "Citations" };
  const FIELD_LABEL = { relevant_chunk_ids: "relevant_chunk_ids", reference_answer: "reference_answer" };
  let catalog = [];
  let picked = new Set();
  let coverage = null;   // { total, relevant_chunk_ids, reference_answer } for the chosen dataset

  function chosenMetrics() {
    const all = catalog.map((m) => m.id);
    const sel = all.filter((id) => picked.has(id));
    if (!sel.length) throw new Error("Choose at least one metric.");
    return sel.length === all.length ? null : sel;   // null = every metric
  }

  async function loadCatalog() {
    catalog = await kit.apiFetch(API + "/metrics");
    if (!picked.size) picked = new Set(catalog.map((m) => m.id));
    renderMetrics();
  }

  async function loadCoverage() {
    coverage = null;
    const id = $("r-dataset").value;
    if (id) {
      try {
        const ds = await kit.apiFetch(API + "/datasets/" + encodeURIComponent(id));
        const items = ds.items.filter((i) => i.status === "accepted");
        coverage = {
          total: items.length,
          relevant_chunk_ids: items.filter((i) => i.relevant_chunk_ids.length).length,
          reference_answer: items.filter((i) => (i.reference_answer || "").trim()).length,
        };
      } catch (e) { coverage = null; }
    }
    renderMetrics();
  }

  function needChip(field) {
    if (!coverage) return el("span", "chip", FIELD_LABEL[field]);
    const n = coverage[field], t = coverage.total;
    const c = el("span", "chip " + (n === t ? "ok" : n === 0 ? "none" : "part"), FIELD_LABEL[field] + ": " + n + " of " + t + " questions");
    return c;
  }

  function blockedBy(m) {
    if (!coverage || !coverage.total) return [];
    return m.needs_dataset.filter((f) => coverage[f] === 0);
  }

  function renderMetrics() {
    const box = $("r-metrics");
    box.replaceChildren();
    $("r-coverage").textContent = coverage ? "Selected dataset: " + coverage.total + " accepted question(s)." : "";
    ["retrieval", "answer", "citations"].forEach((group) => {
      const items = catalog.filter((m) => m.group === group);
      if (!items.length) return;
      const head = el("div", "mgroup");
      head.appendChild(el("span", "", GROUPS[group]));
      [["all", true], ["none", false]].forEach(([label, on]) => {
        const b = el("button", "", label);
        b.type = "button";
        b.addEventListener("click", () => { items.forEach((m) => (on ? picked.add(m.id) : picked.delete(m.id))); renderMetrics(); refreshEstimate(); });
        head.appendChild(b);
      });
      box.appendChild(head);
      items.forEach((m) => {
        const blocked = picked.has(m.id) && blockedBy(m).length > 0;
        const row = el("label", "metric" + (blocked ? " blocked" : ""));
        const cb = el("input"); cb.type = "checkbox"; cb.checked = picked.has(m.id);
        cb.addEventListener("change", () => { cb.checked ? picked.add(m.id) : picked.delete(m.id); renderMetrics(); refreshEstimate(); });
        row.appendChild(cb);
        const body = el("div");
        body.appendChild(el("div", "mname", m.label));
        body.appendChild(el("div", "mdesc", m.description));
        const needs = el("div", "needs");
        if (m.judged) needs.appendChild(el("span", "chip judge", "LLM judge: uses model calls"));
        m.needs_dataset.forEach((f) => needs.appendChild(needChip(f)));
        if (m.needs_target.length) needs.appendChild(el("span", "muted", "target must return: " + m.needs_target.join(", ")));
        if (!m.needs_dataset.length) needs.appendChild(el("span", "muted", "no extra dataset fields"));
        body.appendChild(needs);
        row.appendChild(body);
        box.appendChild(row);
      });
    });
    const problems = catalog.filter((m) => picked.has(m.id) && blockedBy(m).length).map((m) => m.label + " (no question has " + blockedBy(m).join(" or ") + ")");
    $("r-metric-warn").textContent = problems.length ? "These cannot be computed for this dataset: " + problems.join("; ") + ". Add the missing fields in the dataset review, or untick them." : "";
  }

  async function refreshEstimate() {
    $("r-estimate").textContent = "";
    if (!$("r-dataset").value || !picked.size) return;
    try {
      const est = await kit.apiFetch(API + "/runs/estimate", json("POST", runBody()));
      $("r-estimate").textContent = est.questions + " questions. " + est.note +
        (est.max_judge_calls ? " At most " + est.max_judge_calls + " LLM calls." : "");
    } catch (e) { $("r-estimate").textContent = ""; }
  }

  $("r-kind").addEventListener("change", () => {
    const http = $("r-kind").value === "http";
    $("r-http-box").hidden = !http; $("r-preset-box").hidden = http; $("r-options-box").hidden = http;
    refreshEstimate();
  });
  ["r-dataset", "r-preset"].forEach((id) => $(id).addEventListener("change", refreshEstimate));
  $("r-dataset").addEventListener("change", loadCoverage);
  $("r-options").addEventListener("blur", refreshEstimate);
  kit.onDomainChange((domain) => {
    const sel = $("gen-collection");
    if (domain && [...sel.options].some((o) => o.value === domain)) sel.value = domain;
    $("domain-note").textContent = domain ? "Domain \"" + domain + "\" is used as the target's collection unless the options set one." : "";
  });

  $("r-start").addEventListener("click", async () => {
    try {
      $("r-start").disabled = true;
      const accepted = await kit.apiFetch(API + "/runs", json("POST", runBody(), { "Idempotency-Key": crypto.randomUUID() }));
      showError(null); loadRuns();
      const job = await pollJob(accepted.job_id, (j) => { $("r-state").textContent = j.state + " " + j.progress + " / " + j.total; });
      $("r-state").textContent = job.state === "failed" ? "failed" : "done";
      if (job.state === "failed") showError(new Error(job.error));
      await loadRuns(); openRun(accepted.job_id);
    } catch (e) { showError(e); } finally { $("r-start").disabled = false; }
  });

  async function loadRuns() {
    const rows = await kit.apiFetch(API + "/runs");
    table($("runs"), [
      { label: "run", get: (r) => r.id },
      { label: "dataset", get: (r) => r.dataset_name },
      { label: "target", get: (r) => r.target },
      { label: "state", get: (r) => r.state },
      { label: "progress", get: (r) => r.progress + " / " + r.total },
      { label: "started", get: (r) => fmtTime(r.created_at) },
    ], rows, (r) => openRun(r.id));
    fillRunSelects(rows);
    return rows;
  }

  function fillRunSelects(rows) {
    ["c-a", "c-b"].forEach((id, i) => {
      const sel = $(id), keep = sel.value;
      sel.replaceChildren();
      rows.filter((r) => r.state === "completed").forEach((r) => sel.appendChild(new Option(r.target + " | " + r.dataset_name + " | " + r.id, r.id)));
      if (keep && [...sel.options].some((o) => o.value === keep)) sel.value = keep;
      else if (sel.options.length > i) sel.selectedIndex = i;
    });
  }

  async function openRun(id) {
    const run = await kit.apiFetch(API + "/runs/" + encodeURIComponent(id));
    $("run-box").hidden = false;
    $("run-title").textContent = "Run " + run.id + ": " + run.state;
    const c = run.config || {};
    $("run-config").textContent = "dataset " + run.dataset_name + " | target " + run.target + " | k " + (c.ks || []).join(",") +
      " | judge " + (c.judge ? c.judge.model + " (prompt v" + c.judge.prompt_version + ")" : "-");
    const box = $("run-report");
    box.replaceChildren();
    if (run.error) box.appendChild(el("p", "error", run.error));
    if (!run.report) { if (run.state !== "failed") box.appendChild(el("p", "muted", "No report yet.")); renderQuestions(box, run); return; }
    const rep = run.report;
    const stats = el("div", "stats");
    [["questions", rep.questions], ["errors", rep.errors], ["judge calls", rep.judge_calls]].forEach(([k, v]) => {
      const d = el("div"); d.appendChild(el("b", "", String(v))); d.appendChild(el("span", "muted", k)); stats.appendChild(d);
    });
    box.appendChild(stats);
    const mt = el("table");
    table(mt, [
      { label: "metric", get: (m) => m.name }, { label: "mean", get: (m) => fmtNum(m.mean) },
      { label: "std", get: (m) => fmtNum(m.std) }, { label: "questions", get: (m) => m.n },
    ], Object.entries(rep.metrics).map(([name, m]) => Object.assign({ name }, m)));
    box.appendChild(mt);
    Object.entries(rep.not_computed).forEach(([k, why]) => box.appendChild(el("div", "warn", "Not computed: " + k + " (" + why + ")")));
    const lat = Object.entries(rep.latency);
    if (lat.length) {
      box.appendChild(el("h3", "", "Latency"));
      const lt = el("table");
      table(lt, [
        { label: "", get: (l) => l.name }, { label: "p50", get: (l) => fmtMs(l.p50) }, { label: "p95", get: (l) => fmtMs(l.p95) }, { label: "n", get: (l) => l.n },
      ], lat.map(([name, l]) => Object.assign({ name }, l)));
      box.appendChild(lt);
    }
    renderQuestions(box, run);
  }

  function idList(ids, relevant) {
    const wrap = el("div");
    (ids || []).forEach((i) => wrap.appendChild(el("span", "id" + (relevant.includes(i) ? " rel" : ""), i.slice(-8) + " ")));
    if (!ids || !ids.length) wrap.textContent = "-";
    return wrap;
  }

  function renderQuestions(box, run) {
    if (!run.results.length) return;
    box.appendChild(el("h3", "", "Per question (click a row for details)"));
    const t = el("table");
    const keys = run.report ? Object.keys(run.report.metrics).filter((k) => /^(recall@|mrr|faithfulness|answer_accuracy)/.test(k)).slice(0, 4) : [];
    table(t, [
      { label: "question", get: (r) => r.question.slice(0, 80) },
      ...keys.map((k) => ({ label: k, get: (r) => fmtNum(r.metrics[k]) })),
      { label: "", get: (r) => (r.error ? "error" : "") },
    ], run.results, null);
    [...t.tBodies[0].rows].forEach((tr, i) => {
      tr.classList.add("clickable");
      let detail = null;
      tr.addEventListener("click", () => {
        if (detail) { detail.remove(); detail = null; return; }
        detail = t.tBodies[0].insertRow(tr.rowIndex);
        const td = detail.insertCell(); td.colSpan = tr.cells.length;
        td.appendChild(questionDetail(run.results[i]));
      });
    });
    box.appendChild(t);
  }

  function questionDetail(r) {
    const d = el("div", "qdetail");
    const add = (label, node) => { d.appendChild(el("b", "", label)); d.appendChild(node); };
    if (r.error) add("Error", el("pre", "error", r.error));
    add("Question", el("pre", "", r.question));
    if (r.reference_answer) add("Reference answer", el("pre", "", r.reference_answer));
    if (r.answer !== null && r.answer !== undefined) add("Answer", el("pre", "", r.answer));
    if (r.retrieved_chunk_ids) { add("Retrieved (green = relevant)", idList(r.retrieved_chunk_ids, r.relevant_chunk_ids)); }
    if (r.relevant_chunk_ids.length) add("Relevant chunks", idList(r.relevant_chunk_ids, r.relevant_chunk_ids));
    if (r.cited_chunk_ids) add("Cited", idList(r.cited_chunk_ids, r.relevant_chunk_ids));
    add("Metrics", el("pre", "", Object.entries(r.metrics).map(([k, v]) => k + ": " + fmtNum(v)).join("\n") || "-"));
    Object.entries(r.not_computed).forEach(([k, why]) => d.appendChild(el("div", "warn", "Not computed: " + k + " (" + why + ")")));
    if (r.faithfulness) add("Judge: claims (supported?)", el("pre", "", r.faithfulness.claims.map((c) => (c.supported ? "[yes] " : "[NO]  ") + c.text + (c.evidence && c.evidence !== "none" ? "  -- " + c.evidence : "")).join("\n") || "no claims"));
    if (r.completeness) add("Judge: reference points (covered?)", el("pre", "", r.completeness.points.map((p) => (p.covered ? "[yes] " : "[NO]  ") + p.point).join("\n") || "no points"));
    if (r.accuracy) add("Judge: accuracy " + fmtNum(r.accuracy.score), el("pre", "", r.accuracy.rationale));
    return d;
  }

  // ===== compare ==============================================================================
  $("c-go").addEventListener("click", async () => {
    try {
      const cmp = await kit.apiFetch(API + "/runs/compare?a=" + encodeURIComponent($("c-a").value) + "&b=" + encodeURIComponent($("c-b").value));
      showError(null);
      const out = $("c-out");
      out.replaceChildren();
      out.appendChild(el("p", "muted", "A: " + cmp.a.target + " | B: " + cmp.b.target));
      cmp.warnings.forEach((w) => out.appendChild(el("div", "warn", w)));
      const t = el("table");
      table(t, [
        { label: "metric", get: (m) => m.metric }, { label: "A", get: (m) => fmtNum(m.a) }, { label: "B", get: (m) => fmtNum(m.b) },
        { label: "change", get: (m) => {
          if (m.delta === null) return "-";
          const s = el("span", m.delta > 1e-9 ? "pos" : m.delta < -1e-9 ? "neg" : "", (m.delta > 0 ? "+" : "") + m.delta.toFixed(3));
          return s;
        } },
      ], cmp.metrics);
      out.appendChild(t);
      if (cmp.changed_questions.length) {
        out.appendChild(el("h3", "", "Questions that changed the most"));
        const ct = el("table");
        table(ct, [
          { label: "question", get: (c) => c.question.slice(0, 70) }, { label: "metric", get: (c) => c.metric },
          { label: "A", get: (c) => fmtNum(c.a) }, { label: "B", get: (c) => fmtNum(c.b) },
          { label: "change", get: (c) => el("span", c.delta > 0 ? "pos" : "neg", (c.delta > 0 ? "+" : "") + c.delta.toFixed(3)) },
        ], cmp.changed_questions);
        out.appendChild(ct);
      }
    } catch (e) { showError(e); }
  });

  // ===== setup ===============================================================================
  async function refreshAll() {
    try {
      targets = await kit.apiFetch(API + "/targets");
    } catch (e) { showError(e); return; }
    showError(null);
    $("status").textContent = "Judge: " + targets.judge_model + " (prompt v" + targets.judge_prompt_version + ")";
    $("r-hosts").textContent = targets.allowed_target_hosts.join(", ");
    $("r-ks").placeholder = targets.default_ks.join(",");
    const preset = $("r-preset"), keep = preset.value;
    preset.replaceChildren();
    targets.presets.forEach((p) => preset.appendChild(new Option(p, p)));
    if (keep) preset.value = keep;
    try {
      const cols = await kit.apiFetch(API + "/collections");
      const sel = $("gen-collection"); sel.replaceChildren();
      cols.forEach((c) => sel.appendChild(new Option(c, c)));
      if (cols.includes(kit.getDomain())) sel.value = kit.getDomain();
    } catch (e) { /* Qdrant may be down: drafting is then unavailable, the rest still works */ }
    const datasets = await loadDatasets();
    const ds = $("r-dataset"), prev = ds.value;
    ds.replaceChildren();
    datasets.filter((d) => d.counts.accepted > 0).forEach((d) => ds.appendChild(new Option(d.name + " (" + d.counts.accepted + " accepted)", d.id)));
    if (prev && [...ds.options].some((o) => o.value === prev)) ds.value = prev;
    await loadRuns();
    try { if (!catalog.length) await loadCatalog(); } catch (e) { showError(e); }
    await loadCoverage();
    refreshEstimate();
  }

  kit.initTabs(document.querySelector("main"));
  kit.mountKeyBar($("keybar"), refreshAll);
  refreshAll();
})();
