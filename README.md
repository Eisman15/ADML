# IFN680 Exam Practice

Multiple choice practice site for IFN680 Advanced Machine Learning, covering lectures 1–12.

- **Week 12 Mock Quiz**: the practice exam and Week 12 mock quiz questions, plus extra questions in the same style.
- **Practice by Week**: filter by week (1–12) and order questions by week or randomly. Each answer is checked immediately with an explanation.
- **Exam Simulation**: 50 random questions in 60 minutes, then a score broken down by week.
- **Study Notes**: one chapter per week with key topics, formulas and worked computations ([NOTES.md](NOTES.md)).

Every question shows its week and topic. The full bank is also available as Markdown in [QUESTIONS.md](QUESTIONS.md).

## Editing questions

Questions live in `questions.js`, in-depth explanations in `explanations-*.js`, study notes in `notes.js`. After editing, regenerate the Markdown copies:

```
node build-md.js
```

This also bumps the `?v=` version on the links in `index.html`, so browsers load the new files instead of cached ones.
