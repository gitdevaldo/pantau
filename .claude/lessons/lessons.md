# Lessons Learned

> This file is loaded as FIRST PRIORITY before every task. Every mistake, feedback, and correction goes here.
> Format: Case → Problem → Should Be → Goal

---

## L001: Give complete commands with full context
- **Case:** User asked for commands to run training on another server that has nothing installed.
- **Problem:** Gave `git pull` instead of `git clone` — the server doesn't have the repo. Gave incomplete commands missing Python/pip install. Took 5 back-and-forth corrections to get it right.
- **Should Be:** Think about the FULL context before answering. New server = no repo, no Python, no packages. Give the complete set of commands from scratch, properly separated, in one response.
- **Goal:** Never make the user repeat themselves. Get it right the first time by thinking through the full situation.

---

## L002: Don't give one-liner commands for multi-step processes
- **Case:** User needed multiple setup steps (install Python, clone repo, install deps, run training).
- **Problem:** Chained everything into one unreadable line with `&&`.
- **Should Be:** Separate commands into numbered steps, each on its own line with a comment explaining what it does.
- **Goal:** Commands should be clear, readable, and copy-pasteable one at a time.

---

## L003: Think before responding — don't make user correct you repeatedly
- **Case:** Multiple instances of giving wrong/incomplete answers that required 3-5 rounds of correction.
- **Problem:** Responding too fast without fully considering the user's situation, environment, and context.
- **Should Be:** Before answering, ask internally: "What server? What's installed? What state is the repo in? What does the user already have?" Then give a complete, correct answer.
- **Goal:** Maximum 1 correction per task. If the user has to correct you more than once on the same thing, that's a failure.

---

## L004: Write parallel code from the start when possible
- **Case:** Wrote sequential for-loop for K-Fold CV when folds are independent and could run in parallel.
- **Problem:** User had to wait 17 hours for something that could run in 3.5 hours. Only added parallelism after user asked why it's slow.
- **Should Be:** When writing code that processes independent items (folds, batches, files), default to parallel execution. Think about performance from the start.
- **Goal:** Always consider parallelism when designing loops over independent work units.

---

## L005: Verify push with git fetch after every push
- **Case:** Said "pushed" but `git status` showed "ahead of origin by 3 commits" — caused confusion.
- **Problem:** Used explicit push URL which doesn't update local remote tracking refs. `git status` then shows stale info.
- **Should Be:** After every push, run `git fetch origin` so `origin/main` ref is current, then verify with `git log --oneline origin/main -1`. Or use `git push origin main` with proper remote URL set.
- **Goal:** After saying "pushed", the state must be verifiable. No ambiguity.
