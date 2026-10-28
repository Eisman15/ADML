(function () {
  const LETTERS = ["A", "B", "C", "D"];
  const EXAM_SIZE = 50;
  const EXAM_MINUTES = 60;

  const $ = sel => document.querySelector(sel);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
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

  const isOfficial = q => !!q.src && !/Mock-style/i.test(q.src);

  const state = {
    mode: store.get("mode", "mock"),
    week: store.get("week", "all"),
    shuffleQ: store.get("shuffleQ", false),
    items: [],      // [{ qi, order }]
    pos: 0,
    answers: {},    // pos -> displayed option index
    submitted: false,
    deadline: null,
    timer: null
  };

  function makeItem(qi) {
    const q = QUESTIONS[qi];
    const order = q.fixed ? [0, 1, 2, 3] : shuffle([0, 1, 2, 3]);
    return { qi, order };
  }

  function buildSet() {
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
    } else {
      const official = shuffle(idx.filter(i => isOfficial(QUESTIONS[i])));
      const rest = shuffle(idx.filter(i => !isOfficial(QUESTIONS[i])));
      idx = shuffle(official.concat(rest).slice(0, EXAM_SIZE));
    }
    state.items = idx.map(makeItem);
    state.pos = 0;
    state.answers = {};
    state.submitted = false;
    clearInterval(state.timer);
    state.deadline = null;
    if (state.mode === "exam") {
      state.deadline = Date.now() + EXAM_MINUTES * 60000;
      state.timer = setInterval(() => {
        if (Date.now() >= state.deadline) submitExam(); else renderStatus();
      }, 1000);
    }
    render();
  }

  const instantFeedback = () => state.mode !== "exam" || state.submitted;
  const correctDisplayIndex = item => item.order.indexOf(QUESTIONS[item.qi].a);
  const isCorrect = p => state.answers[p] === correctDisplayIndex(state.items[p]);

  function answer(i) {
    const p = state.pos;
    if (instantFeedback() && state.answers[p] !== undefined) return; // locked once revealed
    if (state.submitted) return;
    state.answers[p] = i;
    render();
  }

  function go(delta) {
    const n = state.pos + delta;
    if (n >= 0 && n < state.items.length) { state.pos = n; render(); }
  }

  function submitExam() {
    clearInterval(state.timer);
    state.submitted = true;
    state.pos = -1; // results view
    render();
  }

  function renderControls() {
    const c = $("#controls");
    c.innerHTML = "";
    if (state.mode === "week") {
      const weeks = ["all", ...Object.keys(WEEKS)];
      for (const w of weeks) {
        const count = w === "all" ? QUESTIONS.length : QUESTIONS.filter(q => q.w === Number(w)).length;
        c.append(el("button", {
          class: "chip" + (String(state.week) === String(w) ? " active" : ""),
          title: w === "all" ? "All weeks" : WEEKS[w],
          onclick: () => { state.week = w; store.set("week", w); buildSet(); }
        }, w === "all" ? `All (${count})` : `Week ${w} (${count})`));
      }
    }
    if (state.mode === "mock") {
      c.append(el("span", { class: "sub" },
        "The practice exam and Week 12 mock quiz questions come first, then extra questions in the same style (statement combinations, shape and parameter calculations, \"which is NOT true\")."));
    }
    if (state.mode === "exam") {
      c.append(el("span", { class: "sub" },
        `${EXAM_SIZE} random questions in ${EXAM_MINUTES} minutes, about 1 minute per question like the real exam. Answers are shown after you submit.`));
      c.append(el("button", { class: "btn ghost", onclick: buildSet }, "New exam"));
    } else {
      const order = el("span", { class: "toggle" }, "Order:");
      [["By week", false], ["Random", true]].forEach(([label, rnd]) => order.append(el("button", {
        class: "chip" + (state.shuffleQ === rnd ? " active" : ""),
        onclick: () => { state.shuffleQ = rnd; store.set("shuffleQ", rnd); buildSet(); }
      }, label)));
      c.append(order);
      c.append(el("button", { class: "btn ghost", onclick: buildSet }, "Restart"));
    }
  }

  function renderStatus() {
    const s = $("#status");
    const total = state.items.length;
    const answered = Object.keys(state.answers).length;
    const correct = Object.keys(state.answers).filter(p => isCorrect(Number(p))).length;
    s.innerHTML = "";
    s.append(el("span", {}, "Answered ", el("b", {}, `${answered}/${total}`)));
    if (instantFeedback()) {
      s.append(el("span", {}, "Correct ", el("b", {}, `${correct}`),
        answered ? ` (${Math.round(100 * correct / answered)}%)` : ""));
    }
    if (state.mode === "exam" && !state.submitted && state.deadline) {
      const left = Math.max(0, state.deadline - Date.now());
      const m = Math.floor(left / 60000), sec = Math.floor(left / 1000) % 60;
      s.append(el("span", {}, "Time left ", el("b", {}, `${m}:${String(sec).padStart(2, "0")}`)));
    }
    const bar = el("div", { class: "bar" }, el("i", { style: `width:${total ? 100 * answered / total : 0}%` }));
    s.append(bar);
  }

  function renderCard() {
    const card = $("#card");
    card.innerHTML = "";
    if (!state.items.length) { card.append("No questions for this selection."); return; }
    if (state.pos === -1) { renderResults(card); return; }

    const p = state.pos, item = state.items[p], q = QUESTIONS[item.qi];
    const picked = state.answers[p];
    const reveal = instantFeedback() && picked !== undefined;
    const right = correctDisplayIndex(item);

    const meta = el("div", { class: "meta" },
      el("span", { class: "qnum" }, `Question ${p + 1} of ${state.items.length}`),
      el("span", { class: "badge" }, `Week ${q.w}: ${WEEKS[q.w]}`),
      el("span", { class: "badge style" }, q.t));
    if (q.src) meta.append(el("span", { class: "badge " + (isOfficial(q) ? "mock" : "style") }, isOfficial(q) ? q.src : "Mock-style"));
    card.append(meta, el("p", { class: "qtext" }, q.q));

    const opts = el("div", { class: "opts" });
    item.order.forEach((oi, di) => {
      let cls = "opt";
      if (reveal) { if (di === right) cls += " right"; else if (di === picked) cls += " wrong"; }
      else if (di === picked) cls += " picked";
      const b = el("button", { class: cls, onclick: () => answer(di) },
        el("span", { class: "l" }, LETTERS[di]), el("span", {}, q.o[oi]));
      if (reveal || state.submitted) b.disabled = true;
      opts.append(b);
    });
    card.append(opts);

    if (reveal) {
      const ok = picked === right;
      card.append(el("div", { class: "explain " + (ok ? "ok" : "no") },
        el("b", {}, ok ? "Correct. " : `Incorrect. The answer is ${LETTERS[right]}. `), q.e));
    }

    const actions = el("div", { class: "actions" },
      el("button", { class: "btn ghost", onclick: () => go(-1), ...(p === 0 ? { disabled: "" } : {}) }, "← Previous"));
    if (state.mode === "exam" && !state.submitted) {
      actions.append(el("button", { class: "btn ghost", onclick: () => submitExam() }, "Submit exam"));
    }
    if (state.mode === "exam" && state.submitted) {
      actions.append(el("button", { class: "btn ghost", onclick: () => { state.pos = -1; render(); } }, "Back to results"));
    }
    actions.append(el("button", { class: "btn", onclick: () => go(1), ...(p === state.items.length - 1 ? { disabled: "" } : {}) }, "Next →"));
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
      tbody.append(el("tr", {}, el("td", {}, `Week ${w}: ${WEEKS[w]}`), el("td", {}, `${r.ok}/${r.n}`),
        el("td", {}, `${Math.round(100 * r.ok / r.n)}%`)));
    });
    card.append(el("div", { class: "result" },
      el("h2", {}, `Score: ${correct}/${total} (${Math.round(100 * correct / total)}%)`),
      el("p", {}, "Weakest weeks are the ones to revise. Click any question number below to review its answer and explanation."),
      el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Week"), el("th", {}, "Correct"), el("th", {}, "%"))), tbody),
      el("div", { class: "actions" },
        el("button", { class: "btn ghost", onclick: () => { state.pos = 0; render(); } }, "Review answers"),
        el("button", { class: "btn", onclick: buildSet }, "New exam"))));
  }

  function renderNav() {
    const nav = $("#nav");
    nav.innerHTML = "";
    state.items.forEach((_, p) => {
      let cls = p === state.pos ? "cur" : "";
      if (state.answers[p] !== undefined) cls += instantFeedback() ? (isCorrect(p) ? " ok" : " no") : " ans";
      else if (state.submitted) cls += " no";
      nav.append(el("button", { class: cls, title: `Week ${QUESTIONS[state.items[p].qi].w}`, onclick: () => { state.pos = p; render(); } }, String(p + 1)));
    });
  }

  function render() {
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.mode === state.mode));
    renderControls();
    renderStatus();
    renderCard();
    renderNav();
  }

  document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
    if (state.mode === t.dataset.mode) return;
    state.mode = t.dataset.mode;
    store.set("mode", state.mode);
    buildSet();
  }));

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || state.pos === -1) return;
    const k = e.key.toUpperCase();
    const n = "1234".indexOf(k) >= 0 ? "1234".indexOf(k) : LETTERS.indexOf(k);
    if (n >= 0 && k.length === 1) answer(n);
    else if (e.key === "ArrowRight") go(1);
    else if (e.key === "ArrowLeft") go(-1);
  });

  if (!["mock", "week", "exam"].includes(state.mode)) state.mode = "mock";
  buildSet();
})();
