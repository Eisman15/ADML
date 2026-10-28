// Generates QUESTIONS.md from questions.js:  node build-md.js
const fs = require("fs");
const path = require("path");

const src = fs.readFileSync(path.join(__dirname, "questions.js"), "utf8");
const { WEEKS, QUESTIONS, isMock } = new Function(src + "; return { WEEKS, QUESTIONS, isMock };")();

const L = ["A", "B", "C", "D"];
const isOfficial = q => !!q.src && !/Mock-style/i.test(q.src);

function block(q, n) {
  const lines = [];
  const tag = q.src ? ` · _${q.src}_` : "";
  lines.push(`**Q${n}. [Week ${q.w}: ${q.t}]**${tag}`, "");
  lines.push(q.q.replace(/\n/g, "  \n"), "");
  q.o.forEach((o, i) => lines.push(`- ${L[i]}. ${o}`));
  lines.push("", `<details><summary>Answer</summary>`, "", `**${L[q.a]}.** ${q.e}`, "", `</details>`, "");
  return lines.join("\n");
}

const out = [];
out.push("# IFN680 Exam Practice: Question Bank", "");
out.push(`${QUESTIONS.length} multiple choice questions generated from the IFN680 lectures (weeks 1–12), using the weekly review questions and the practice exam / Week 12 mock quiz as a guide.`, "");
out.push("Interactive version: open `index.html` (GitHub Pages).", "");
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
console.log(`Wrote QUESTIONS.md with ${n} questions`);
