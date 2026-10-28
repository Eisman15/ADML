(function () {
  const LETTERS = ["A", "B", "C", "D"];
  const EXAM_SIZE = 50;
  const EXAM_MINUTES = 60;
  const MODES = ["mock", "week", "exam", "notes"];

  const $ = sel => document.querySelector(sel);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v == null) continue;
      if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids) if (kid != null) n.append(kid);
    return n;
  };
  const shuffle = arr => {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem("ifn680:" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("ifn680:" + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } }
  };
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = s => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>");

  // Light markdown for the study notes (see notes.js header for the syntax).
  function renderMd(md) {
    const out = [];
    let list = null;
    const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of md.trim().split("\n")) {
      const line = raw.trim();
      let m;
      if (!line) { close(); continue; }
      if ((m = line.match(/^(#{2,3}) (.*)/))) { close(); out.push(`<h${m[1].length + 1}>${inline(m[2])}</h${m[1].length + 1}>`); }
      else if ((m = line.match(/^- (.*)/))) { if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(m[1])}</li>`); }
      else if ((m = line.match(/^\d+\. (.*)/))) { if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(m[1])}</li>`); }
      else if ((m = line.match(/^> (.*)/))) { close(); out.push(`<p class="tip">${inline(m[1])}</p>`); }
      else { close(); out.push(`<p>${inline(line)}</p>`); }
    }
    close();
    return out.join("");
  }

  // Attach each in-depth explanation to its question (longest matching question prefix wins).
  QUESTIONS.forEach(q => {
    let best = null;
    for (const d of DEEP) if (q.q.startsWith(d[0]) && (!best || d[0].length > best[0].length)) best = d;
    if (best) q.deep = { text: best[1], notes: best[2] };
  });

  const isOfficial = q => !!q.src && !/Mock-style/i.test(q.src);

  const state = {
    mode: store.get("mode", "mock"),
    week: store.get("week", "all"),
    noteWeek: store.get("noteWeek", 1),
    shuffleQ: store.get("shuffleQ", false),
    items: [],      // [{ qi, order }]
    pos: 0,
    answers: {},    // pos -> displayed option index
    submitted: false,
    deadline: null,
    timer: null
  };
  if (!MODES.includes(state.mode)) state.mode = "mock";

  function makeItem(qi) {
    const q = QUESTIONS[qi];
    return { qi, order: q.fixed ? [0, 1, 2, 3] : shuffle([0, 1, 2, 3]) };
  }

  function buildSet() {
    clearInterval(state.timer);
    state.deadline = null;
    state.pos = 0;
    state.answers = {};
    state.submitted = false;
    let idx = QUESTIONS.map((_, i) => i);
    if (state.mode === "mock") {
      idx = idx.filter(i => isMock(QUESTIONS[i]));
      // Official practice exam / Week 12 mock quiz questions first, then mock-style, each by week.
      idx.sort((a, b) => (isOfficial(QUESTIONS[b]) - isOfficial(QUESTIONS[a])) || (QUESTIONS[a].w - QUESTIONS[b].w));
      if (state.shuffleQ) idx = shuffle(idx);
    } else if (state.mode === "week") {
      if (state.week !== "all") idx = idx.filter(i => QUESTIONS[i].w === Number(state.week));
      idx.sort((a, b) => QUESTIONS[a].w - QUESTIONS[b].w);
      if (state.shuffleQ) idx = shuffle(idx);
    } else if (state.mode === "exam") {
      const official = shuffle(idx.filter(i => isOfficial(QUESTIONS[i])));
      const rest = shuffle(idx.filter(i => !isOfficial(QUESTIONS[i])));
      idx = shuffle(official.concat(rest).slice(0, EXAM_SIZE));
      state.deadline = Date.now() + EXAM_MINUTES * 60000;
      state.timer = setInterval(() => {
        if (Date.now() >= state.deadline) submitExam(); else renderStatus();
      }, 1000);
    } else {
      idx = [];
    }
    state.items = idx.map(makeItem);
    render();
  }

  const instantFeedback = () => state.mode !== "exam" || state.submitted;
  const correctDisplayIndex = item => item.order.indexOf(QUESTIONS[item.qi].a);
  const isCorrect = p => state.answers[p] === correctDisplayIndex(state.items[p]);

  function answer(i) {
    const p = state.pos;
    if (state.mode === "notes" || p < 0 || state.submitted) return;
    if (instantFeedback() && state.answers[p] !== undefined) return; // locked once revealed
    state.answers[p] = i;
    render();
  }

  function go(delta) {
    const n = state.pos + delta;
    if (n >= 0 && n < state.items.length) { state.pos = n; render(); scrollToCard(); }
  }

  function scrollToCard() {
    const card = $("#card");
    if (card.getBoundingClientRect().top < 0) card.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function submitExam() {
    clearInterval(state.timer);
    state.submitted = true;
    state.pos = -1; // results view
    render();
  }

  function setMode(mode) {
    state.mode = mode;
    store.set("mode", mode);
    buildSet();
    window.scrollTo(0, 0);
  }

  function chip(label, active, onclick, title) {
    return el("button", { class: "chip" + (active ? " active" : ""), onclick, title }, label);
  }

  function renderControls() {
    const c = $("#controls");
    c.innerHTML = "";
    if (state.mode === "notes") {
      const row = el("div", { class: "chips" });
      for (const w of Object.keys(WEEKS)) {
        row.append(chip(`W${w}`, Number(state.noteWeek) === Number(w),
          () => { state.noteWeek = Number(w); store.set("noteWeek", state.noteWeek); render(); window.scrollTo(0, 0); }, WEEKS[w]));
      }
      c.append(row);
      return;
    }
    if (state.mode === "week") {
      const row = el("div", { class: "chips" });
      for (const w of ["all", ...Object.keys(WEEKS)]) {
        const count = w === "all" ? QUESTIONS.length : QUESTIONS.filter(q => q.w === Number(w)).length;
        row.append(chip(w === "all" ? `All ${count}` : `W${w} · ${count}`, String(state.week) === String(w),
          () => { state.week = w; store.set("week", w); buildSet(); }, w === "all" ? "All weeks" : WEEKS[w]));
      }
      c.append(row);
    }
    if (state.mode === "mock") {
      c.append(el("p", { class: "hint" },
        "Official practice exam and Week 12 mock quiz questions come first, then extra questions in the same style."));
    }
    const bar = el("div", { class: "ctlbar" });
    if (state.mode === "exam") {
      bar.append(el("p", { class: "hint" }, `${EXAM_SIZE} random questions, ${EXAM_MINUTES} min (≈1 min each). Answers are shown after you submit.`));
      bar.append(el("button", { class: "btn ghost sm", onclick: buildSet }, "New exam"));
    } else {
      const order = el("div", { class: "seg", role: "group", "aria-label": "Question order" });
      [["By week", false], ["Random", true]].forEach(([label, rnd]) => order.append(el("button", {
        class: state.shuffleQ === rnd ? "on" : "",
        onclick: () => { state.shuffleQ = rnd; store.set("shuffleQ", rnd); buildSet(); }
      }, label)));
      bar.append(order, el("button", { class: "btn ghost sm", onclick: buildSet }, "Restart"));
    }
    c.append(bar);
  }

  function renderStatus() {
    const s = $("#status");
    s.innerHTML = "";
    if (state.mode === "notes") return;
    const total = state.items.length;
    const answered = Object.keys(state.answers).length;
    const correct = Object.keys(state.answers).filter(p => isCorrect(Number(p))).length;
    const row = el("div", { class: "statrow" });
    row.append(el("span", {}, "Answered ", el("b", {}, `${answered}/${total}`)));
    if (instantFeedback()) {
      row.append(el("span", {}, "Correct ", el("b", {}, `${correct}`), answered ? ` (${Math.round(100 * correct / answered)}%)` : ""));
    }
    if (state.mode === "exam" && !state.submitted && state.deadline) {
      const left = Math.max(0, state.deadline - Date.now());
      const m = Math.floor(left / 60000), sec = Math.floor(left / 1000) % 60;
      row.append(el("span", { class: "timer" }, "⏱ ", el("b", {}, `${m}:${String(sec).padStart(2, "0")}`)));
    }
    s.append(row, el("div", { class: "bar" }, el("i", { style: `width:${total ? 100 * answered / total : 0}%` })));
  }

  function renderNotes(card) {
    const w = Number(state.noteWeek);
    const note = NOTES.find(n => n.w === w);
    const count = QUESTIONS.filter(q => q.w === w).length;
    card.append(
      el("div", { class: "meta" }, el("span", { class: "badge" }, `Week ${w}`)),
      el("h2", { class: "chapter" }, WEEKS[w]),
      el("div", { class: "notes", html: note ? renderMd(note.md) : "<p>No notes yet.</p>" }),
      el("div", { class: "actions notes-actions" },
        el("button", { class: "btn ghost", disabled: w === 1, onclick: () => { state.noteWeek = w - 1; store.set("noteWeek", w - 1); render(); window.scrollTo(0, 0); } }, "← Prev"),
        el("button", { class: "btn", onclick: () => { state.week = String(w); store.set("week", state.week); setMode("week"); } }, `Practise Week ${w} (${count})`),
        el("button", { class: "btn ghost", disabled: w === 12, onclick: () => { state.noteWeek = w + 1; store.set("noteWeek", w + 1); render(); window.scrollTo(0, 0); } }, "Next →")));
  }

  function renderExplanation(card, item, q, picked, right) {
    const ok = picked === right;
    card.append(el("div", { class: "explain " + (ok ? "ok" : "no") },
      el("b", {}, ok ? "Correct. " : `Incorrect. The answer is ${LETTERS[right]}. `), q.e));
    if (!q.deep) return;
    const why = el("ul", { class: "why" });
    item.order.forEach((oi, di) => {
      const mark = di === right ? "✓" : "✗";
      why.append(el("li", { class: di === right ? "good" : "" },
        el("span", { class: "wl" }, `${LETTERS[di]} ${mark}`), el("span", {}, q.deep.notes[oi])));
    });
    card.append(el("details", { class: "deep", open: true },
      el("summary", {}, "In-depth explanation"),
      el("h4", {}, "Why each option is right or wrong"), why,
      el("h4", {}, "Concept & working"), el("p", {}, q.deep.text)));
  }

  function renderCard() {
    const card = $("#card");
    card.innerHTML = "";
    if (state.mode === "notes") { renderNotes(card); return; }
    if (!state.items.length) { card.append("No questions for this selection."); return; }
    if (state.pos === -1) { renderResults(card); return; }

    const p = state.pos, item = state.items[p], q = QUESTIONS[item.qi];
    const picked = state.answers[p];
    const reveal = instantFeedback() && picked !== undefined;
    const right = correctDisplayIndex(item);

    const meta = el("div", { class: "meta" },
      el("span", { class: "qnum" }, `Q${p + 1}/${state.items.length}`),
      el("span", { class: "badge" }, `Week ${q.w}`),
      el("span", { class: "badge style" }, q.t));
    if (q.src) meta.append(el("span", { class: "badge " + (isOfficial(q) ? "mock" : "style") }, isOfficial(q) ? q.src : "Mock-style"));
    card.append(meta, el("p", { class: "weekname" }, WEEKS[q.w]), el("p", { class: "qtext" }, q.q));

    const opts = el("div", { class: "opts" });
    item.order.forEach((oi, di) => {
      let cls = "opt";
      if (reveal) { if (di === right) cls += " right"; else if (di === picked) cls += " wrong"; }
      else if (di === picked) cls += " picked";
      opts.append(el("button", { class: cls, onclick: () => answer(di), disabled: reveal || state.submitted },
        el("span", { class: "l" }, LETTERS[di]), el("span", {}, q.o[oi])));
    });
    card.append(opts);

    if (reveal) renderExplanation(card, item, q, picked, right);

    const actions = el("div", { class: "actions" },
      el("button", { class: "btn ghost", onclick: () => go(-1), disabled: p === 0 }, "← Prev"));
    if (state.mode === "exam" && !state.submitted) actions.append(el("button", { class: "btn ghost", onclick: submitExam }, "Submit"));
    if (state.mode === "exam" && state.submitted) actions.append(el("button", { class: "btn ghost", onclick: () => { state.pos = -1; render(); } }, "Results"));
    actions.append(el("button", { class: "btn", onclick: () => go(1), disabled: p === state.items.length - 1 }, "Next →"));
    card.append(actions);
  }

  function renderResults(card) {
    const total = state.items.length;
    const correct = state.items.filter((_, p) => isCorrect(p)).length;
    const byWeek = {};
    state.items.forEach((it, p) => {
      const w = QUESTIONS[it.qi].w;
      byWeek[w] = byWeek[w] || { n: 0, ok: 0 };
      byWeek[w].n++;
      if (isCorrect(p)) byWeek[w].ok++;
    });
    const tbody = el("tbody");
    Object.keys(byWeek).sort((a, b) => a - b).forEach(w => {
      const r = byWeek[w];
      tbody.append(el("tr", {}, el("td", {}, `W${w} ${WEEKS[w]}`), el("td", {}, `${r.ok}/${r.n}`),
        el("td", {}, `${Math.round(100 * r.ok / r.n)}%`)));
    });
    card.append(el("div", { class: "result" },
      el("h2", {}, `Score ${correct}/${total} (${Math.round(100 * correct / total)}%)`),
      el("p", {}, "Revise your weakest weeks in Study Notes. Tap a question number to review its answer and in-depth explanation."),
      el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Week"), el("th", {}, "Correct"), el("th", {}, "%"))), tbody),
      el("div", { class: "actions" },
        el("button", { class: "btn ghost", onclick: () => { state.pos = 0; render(); } }, "Review answers"),
        el("button", { class: "btn", onclick: buildSet }, "New exam"))));
  }

  function renderNav() {
    const nav = $("#nav");
    nav.innerHTML = "";
    if (state.mode === "notes" || !state.items.length) return;
    const grid = el("div", { class: "grid" });
    state.items.forEach((_, p) => {
      let cls = p === state.pos ? "cur" : "";
      if (state.answers[p] !== undefined) cls += instantFeedback() ? (isCorrect(p) ? " ok" : " no") : " ans";
      else if (state.submitted) cls += " no";
      grid.append(el("button", { class: cls, title: `Week ${QUESTIONS[state.items[p].qi].w}`, onclick: () => { state.pos = p; render(); scrollToCard(); } }, String(p + 1)));
    });
    const open = store.get("navOpen", window.matchMedia("(min-width: 720px)").matches);
    const det = el("details", { class: "jump", open }, el("summary", {}, "Jump to question"), grid);
    det.addEventListener("toggle", () => store.set("navOpen", det.open));
    nav.append(det);
  }

  function render() {
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.mode === state.mode));
    renderControls();
    renderStatus();
    renderCard();
    renderNav();
  }

  document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
    if (state.mode !== t.dataset.mode) setMode(t.dataset.mode);
  }));

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "ArrowRight") return go(1);
    if (e.key === "ArrowLeft") return go(-1);
    const k = e.key.toUpperCase();
    const n = "1234".includes(k) ? "1234".indexOf(k) : LETTERS.indexOf(k);
    if (n >= 0 && k.length === 1) answer(n);
  });

  buildSet();
})();
