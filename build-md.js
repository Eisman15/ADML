// Generates QUESTIONS.md and NOTES.md from the site data:  node build-md.js
const fs = require("fs");
const path = require("path");

const files = ["questions.js", "explanations-1.js", "explanations-2.js", "explanations-3.js", "notes.js"];
const src = files.map(f => fs.readFileSync(path.join(__dirname, f), "utf8")).join("\n");
const { WEEKS, QUESTIONS, DEEP, NOTES, isMock } = new Function(src + "; return { WEEKS, QUESTIONS, DEEP, NOTES, isMock };")();

const L = ["A", "B", "C", "D"];
const isOfficial = q => !!q.src && !/Mock-style/i.test(q.src);

// Attach in-depth explanations (longest matching question prefix wins) and report gaps.
const used = new Set();
QUESTIONS.forEach(q => {
  let best = null;
  for (const d of DEEP) if (q.q.startsWith(d[0]) && (!best || d[0].length > best[0].length)) best = d;
  if (best) { q.deep = best; used.add(best); } else console.warn("No in-depth explanation for:", q.q.slice(0, 70));
});
DEEP.filter(d => !used.has(d)).forEach(d => console.warn("Unused explanation:", d[0]));

function block(q, n) {
  const lines = [];
  const tag = q.src ? ` · _${q.src}_` : "";
  lines.push(`**Q${n}. [Week ${q.w}: ${q.t}]**${tag}`, "");
  lines.push(q.q.replace(/\n/g, "  \n"), "");
  q.o.forEach((o, i) => lines.push(`- ${L[i]}. ${o}`));
  lines.push("", `<details><summary>Answer</summary>`, "", `**${L[q.a]}.** ${q.e}`, "");
  if (q.deep) {
    lines.push("Why each option:", "");
    q.deep[2].forEach((note, i) => lines.push(`- ${L[i]} ${i === q.a ? "✓" : "✗"} ${note}`));
    lines.push("", `In depth: ${q.deep[1]}`, "");
  }
  lines.push(`</details>`, "");
  return lines.join("\n");
}

const out = [];
out.push("# IFN680 Exam Practice: Question Bank", "");
out.push(`${QUESTIONS.length} multiple choice questions generated from the IFN680 lectures (weeks 1–12), using the weekly review questions and the practice exam / Week 12 mock quiz as a guide. Every answer has an in-depth explanation.`, "");
out.push("Interactive version: https://eisman15.github.io/ifn680-exam-practice/ · Study notes: [NOTES.md](NOTES.md)", "");
out.push("## Contents", "", "- [Week 12 Mock Quiz](#week-12-mock-quiz)");
for (const w of Object.keys(WEEKS)) out.push(`- [Week ${w}: ${WEEKS[w]}](#week-${w})`);
out.push("");

let n = 0;
const mock = QUESTIONS.filter(isMock).sort((a, b) => (isOfficial(b) - isOfficial(a)) || (a.w - b.w));
out.push("## Week 12 Mock Quiz", "", "Official practice exam / Week 12 mock quiz questions first, then mock-style questions in the same format.", "");
mock.forEach(q => out.push(block(q, ++n)));

for (const w of Object.keys(WEEKS)) {
  out.push(`<a id="week-${w}"></a>`, "", `## Week ${w}: ${WEEKS[w]}`, "");
  QUESTIONS.filter(q => q.w === Number(w) && !isMock(q)).forEach(q => out.push(block(q, ++n)));
}
fs.writeFileSync(path.join(__dirname, "QUESTIONS.md"), out.join("\n"));

// NOTES.md: shift note headings down one level under each week heading.
const notes = ["# IFN680 Study Notes", "", "One chapter per week: key topics, formulas and worked computations.", ""];
for (const w of Object.keys(WEEKS)) notes.push(`- [Week ${w}: ${WEEKS[w]}](#week-${w})`);
notes.push("");
for (const note of NOTES) {
  notes.push(`<a id="week-${note.w}"></a>`, "", `## Week ${note.w}: ${WEEKS[note.w]}`, "");
  notes.push(note.md.trim().replace(/^(#{2,3}) /gm, "#$1 "), "");
}
fs.writeFileSync(path.join(__dirname, "NOTES.md"), notes.join("\n"));

// Bump the ?v= cache-busting version in index.html so browsers fetch the new files.
const stamp = new Date().toISOString().replace(/\D/g, "").slice(0, 12);
const indexPath = path.join(__dirname, "index.html");
fs.writeFileSync(indexPath, fs.readFileSync(indexPath, "utf8").replace(/\?v=\d+"/g, `?v=${stamp}"`));

console.log(`Wrote QUESTIONS.md (${n} questions, ${used.size} in-depth explanations) and NOTES.md (${NOTES.length} weeks)`);
