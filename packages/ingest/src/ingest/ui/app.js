// Ingest UI. Plain JS. All text is set with textContent, never innerHTML.
(function () {
  const API = "/api/ingest";
  const $ = (id) => document.getElementById(id);
  let files = [];
  let status = null;

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function showError(e) { $("error").textContent = e ? kit.errorText(e) : ""; }
  function fmtTime(ts) { return new Date(ts * 1000).toLocaleString(); }
  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms"; }

  // ----- status and options --------------------------------------------------------------
  async function loadStatus() {
    try {
      status = await kit.apiFetch(API + "/status");
    } catch (e) { showError(e); return; }
    showError(null);
    const parts = ["embedding: " + status.embed_model, "PDF parser: " + status.pdf_parser];
    if (status.pdf_parser === "llamaparse" && !status.llamaparse_configured) {
      parts.push("LLAMA_CLOUD_API_KEY not set, so scanned PDF pages will be skipped");
    }
    $("status").textContent = parts.join(" | ");
    $("formats").textContent = "Supported: " + status.supported_extensions.join(" ") +
      " (max " + status.max_upload_mb + " MB each)";
    $("file-input").accept = status.supported_extensions.join(",");
    if (!$("o-collection").value) {
      $("o-collection").value = kit.getDomain() || status.default_collection;
      $("o-strategy").value = status.chunk_strategy;
      $("o-size").value = status.chunk_size;
      $("o-overlap").value = status.chunk_overlap;
    }
    const lang = $("o-language");
    if (!lang.options.length) {
      status.languages.forEach((l) => lang.appendChild(new Option(l, l)));
      lang.value = status.sparse_language;
    }
    $("path-box").hidden = status.allowed_paths.length === 0;
    $("path-help").hidden = status.allowed_paths.length === 0;
    showStrategyHelp();
    loadCollections();
  }

  function options() {
    const o = {
      collection: $("o-collection").value.trim() || null,
      chunking: {
        strategy: $("o-strategy").value,
        chunk_size: parseInt($("o-size").value, 10),
        overlap: parseInt($("o-overlap").value, 10),
      },
      sparse_language: $("o-language").value,
      metadata: {},
    };
    const raw = $("o-metadata").value.trim();
    if (raw) {
      try { o.metadata = JSON.parse(raw); } catch (e) { throw new Error("Metadata is not valid JSON."); }
      if (o.metadata === null || typeof o.metadata !== "object" || Array.isArray(o.metadata)) {
        throw new Error("Metadata must be a JSON object, like {\"key\": \"value\"}.");
      }
    }
    return o;
  }

  // ----- file list -------------------------------------------------------------------------------
  function renderFiles() {
    const ul = $("files");
    ul.replaceChildren();
    files.forEach((f, i) => {
      const li = el("li");
      li.appendChild(el("span", "", f.name + "  (" + (f.size / 1024).toFixed(1) + " KB)"));
      const rm = el("button", "", "remove");
      rm.addEventListener("click", () => { files.splice(i, 1); renderFiles(); });
      li.appendChild(rm);
      ul.appendChild(li);
    });
    $("preview").disabled = files.length === 0;
    $("run").disabled = files.length === 0;
  }
  function addFiles(list) { files = files.concat(Array.from(list)); renderFiles(); }

  $("pick").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

  function busy(text) { $("busy").textContent = text || ""; }

  // ----- preview -----------------------------------------------------------------------------------
  $("preview").addEventListener("click", async () => {
    showError(null);
    try {
      const form = new FormData();
      form.append("file", files[0]);
      form.append("options", JSON.stringify(options()));
      busy("parsing and chunking...");
      const r = await kit.apiFetch(API + "/chunks/preview", { method: "POST", body: form });
      renderPreview(r);
    } catch (e) { showError(e); } finally { busy(""); }
  });

  function renderPreview(r) {
    $("preview-box").hidden = false;
    const s = r.stats;
    $("preview-info").textContent =
      r.source_name + ": " + r.chunks.length + " chunks | pages local " + s.pages_local +
      ", from cache " + s.pages_cached + ", skipped " + s.pages_skipped +
      (r.trace_id ? " | trace " + r.trace_id : "");
    $("preview-warnings").replaceChildren(...r.warnings.map((w) => el("div", "", w)));
    const box = $("chunks");
    box.replaceChildren();
    r.chunks.forEach((c, i) => {
      const card = el("div", "chunk " + (i % 2 ? "b" : "a"));
      const head = el("div", "head");
      head.appendChild(el("b", "", "#" + c.index));
      head.appendChild(el("span", "", c.tokens + " tokens"));
      head.appendChild(el("span", "", "page " + c.page_no + (c.label ? " (" + c.label + ")" : "")));
      if (c.section_path) head.appendChild(el("span", "", c.section_path));
      head.appendChild(el("span", "", "chars " + c.char_start + "-" + c.char_end));
      const prev = r.chunks[i - 1];
      if (prev && prev.page_no === c.page_no && c.char_start < prev.char_end) {
        head.appendChild(el("span", "", "overlaps previous by " + (prev.char_end - c.char_start) + " chars"));
      }
      card.appendChild(head);
      card.appendChild(el("pre", "", c.text));
      box.appendChild(card);
    });
  }

  // ----- running -----------------------------------------------------------------------------------
  async function startJob(path, init) {
    showError(null);
    busy("starting...");
    init.headers = Object.assign({}, init.headers, { "Idempotency-Key": crypto.randomUUID() });
    const job = await kit.apiFetch(path, init);
    $("report-box").hidden = false;
    $("report").replaceChildren();
    await pollJob(job.job_id);
  }

  async function pollJob(id) {
    for (;;) {
      const job = await kit.apiFetch(API + "/jobs/" + encodeURIComponent(id));
      $("job-state").textContent = "Job " + id + ": " + job.state;
      busy(job.state === "queued" || job.state === "running" ? "working..." : "");
      if (job.state === "completed") { renderReport(job.report); loadCollections(); return; }
      if (job.state === "failed") {
        $("report").replaceChildren(el("p", "error", job.error || "The job failed."));
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  function renderReport(r) {
    const box = $("report");
    box.replaceChildren();
    const stats = el("div", "grid-stats");
    [["added", r.added], ["updated", r.updated], ["skipped", r.skipped], ["deleted chunks", r.deleted],
      ["chunks embedded", r.chunks_written], ["chunks reused", r.chunks_reused],
      ["pages local", r.pages_local],
      ["pages to LlamaParser", r.pages_llamaparse], ["pages from cache", r.pages_cached],
      ["pages skipped", r.pages_skipped], ["duration", fmtMs(r.duration_ms)]].forEach(([k, v]) => {
      const d = el("div");
      d.appendChild(el("b", "", String(v)));
      d.appendChild(el("span", "muted", k));
      stats.appendChild(d);
    });
    box.appendChild(stats);
    if (r.trace_id) box.appendChild(el("p", "muted", "trace " + r.trace_id + " (see the Observability tab)"));
    r.warnings.forEach((w) => box.appendChild(el("div", "warn", w)));
    r.failed.forEach((f) => box.appendChild(el("div", "error", f.source + ": " + f.error)));
    if (r.files.length) {
      const t = el("table");
      kit.renderTable(t, r.files.map((f) => ({
        file: f.source, result: f.status, chunks: f.chunks, reused: f.reused })),
      ["file", "result", "chunks", "reused"]);
      box.appendChild(t);
    }
  }

  $("run").addEventListener("click", async () => {
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      form.append("options", JSON.stringify(options()));
      await startJob(API + "/files", { method: "POST", body: form });
      files = [];
      renderFiles();
    } catch (e) { showError(e); busy(""); }
  });

  $("run-path").addEventListener("click", async () => {
    try {
      const path = $("path").value.trim();
      if (!path) throw new Error("Enter a path.");
      await startJob(API + "/path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, options: options() }),
      });
    } catch (e) { showError(e); busy(""); }
  });

  // ----- collections and sources ------------------------------------------------------------------
  async function loadCollections() {
    let cols = [];
    try {
      cols = await kit.apiFetch(API + "/collections");
    } catch (e) {
      $("c-info").textContent = e.status === 503 ? e.message : "";
      return;
    }
    const sel = $("c-select");
    const keep = sel.value || ($("o-collection").value || "");
    $("o-collection-list").replaceChildren(...cols.map((c) => new Option(c.name, c.name)));
    $("c-delete-all").disabled = cols.length === 0;
    sel.replaceChildren();
    cols.forEach((c) => sel.appendChild(new Option(c.name + " (" + c.points + " chunks)", c.name)));
    if (cols.some((c) => c.name === keep)) sel.value = keep;
    const cur = cols.find((c) => c.name === sel.value);
    $("c-info").textContent = cur && cur.spec
      ? cur.spec.embedding_model + ", " + cur.spec.dense_dim + " dims, BM25 language: " + cur.spec.sparse_language
      : (cols.length ? "" : "No collections yet.");
    loadSources();
  }

  async function loadSources() {
    const name = $("c-select").value;
    const table = $("sources");
    table.replaceChildren();
    if (!name) return;
    try {
      const rows = await kit.apiFetch(API + "/collections/" + encodeURIComponent(name) + "/sources");
      if (!rows.length) { table.appendChild(el("caption", "muted", "No sources recorded for this collection.")); return; }
      const head = table.createTHead().insertRow();
      ["source", "chunks", "pages", "parsers", "hash", "last ingested", ""].forEach((t) => head.appendChild(el("th", "", t)));
      const body = table.createTBody();
      rows.forEach((s) => {
        const tr = body.insertRow();
        const parsers = Object.entries(s.parsers).map(([k, v]) => k + " " + v).join(", ");
        [s.source_name, s.chunk_count, s.pages, parsers, s.content_hash.slice(0, 10), fmtTime(s.last_ingested_at)]
          .forEach((v) => { tr.insertCell().textContent = String(v); });
        const del = el("button", "danger", "delete");
        del.addEventListener("click", async () => {
          if (!confirm("Delete " + s.source_name + " and its " + s.chunk_count + " chunks?")) return;
          try {
            await kit.apiFetch(API + "/collections/" + encodeURIComponent(name) + "/sources/" + encodeURIComponent(s.source_id), { method: "DELETE" });
            loadCollections();
          } catch (e) { showError(e); }
        });
        tr.insertCell().appendChild(del);
      });
    } catch (e) { showError(e); }
  }
  $("c-select").addEventListener("change", loadSources);

  // Delete a whole collection. The server needs the name repeated, so a slip cannot do it.
  $("c-delete-all").addEventListener("click", async () => {
    const name = $("c-select").value;
    if (!name) return;
    const typed = prompt(
      "This deletes ALL chunks of the collection and forgets every file ingested into it. " +
      "It cannot be undone (you can ingest the files again).\n\nType the collection name to confirm: " + name);
    if (typed === null) return;
    if (typed !== name) { showError(new Error("The name did not match, so nothing was deleted.")); return; }
    try {
      const r = await kit.apiFetch(API + "/collections/" + encodeURIComponent(name) + "?confirm=" + encodeURIComponent(typed), { method: "DELETE" });
      showError(null);
      $("c-info").textContent = "Deleted " + r.collection + ": " + r.chunks_deleted + " chunks, " + r.sources_forgotten + " file record(s).";
      await loadCollections();
    } catch (e) { showError(e); }
  });

  const STRATEGY_HELP = {
    recursive: "recursive: splits at paragraphs, then sentences, then words, keeping chunks near the size you set. A good default for most text.",
    fixed: "fixed: cuts every N tokens regardless of structure. Simple and predictable, but can cut mid-sentence.",
    markdown: "markdown: like recursive, but never crosses a # heading, so a chunk never mixes two sections. Best for .md files. Tables (csv, xlsx) are always split by rows, repeating the header.",
  };
  function showStrategyHelp() { $("strategy-help").textContent = STRATEGY_HELP[$("o-strategy").value] || ""; }
  $("o-strategy").addEventListener("change", showStrategyHelp);

  // The main page's Domain box sets the collection used here.
  kit.onDomainChange((domain) => {
    if (domain) { $("o-collection").value = domain; if ([...$("c-select").options].some((o) => o.value === domain)) { $("c-select").value = domain; loadSources(); } }
  });
  $("c-reload").addEventListener("click", loadCollections);

  kit.mountKeyBar($("keybar"), loadStatus);
  renderFiles();
  loadStatus();
})();
