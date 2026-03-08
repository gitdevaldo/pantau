# Lessons Learned

> **⚠️ This file is the FIRST PRIORITY — read before every task, every to-do, every instruction.**
> These are hard-won rules from past mistakes. Each lesson is a PRINCIPLE to follow,
> not just a case study. Violating any of these is a blocking issue.
> Updated after each correction from the user.

<!--
  MANDATORY: AI agents MUST read this file at the start of every session and every task.
  Format: Principle title → Rule → Why → How to verify → Origin
-->

---

## Principle 1: Give Complete Commands With Full Context — Think Before Responding

**Rule**: Before giving any command, think through the FULL environment: What server? What's installed? Is the repo cloned? What OS? What package manager? Then give the complete set of commands from scratch in one response. Never assume prior state.

**Why**: Incomplete commands force the user into 3-5 rounds of corrections. Each correction wastes time and builds frustration. A senior engineer gets it right the first time.

**How to verify**:
1. Before writing any command, mentally list: OS, installed tools, repo state, directory location, auth/credentials needed.
2. If ANY of these are unknown, ask once — don't guess.
3. Review the full command set before sending. Walk through it step by step: "If I paste these on a blank server, will they work?"

**Origin**: User asked for commands to run training on another server. Gave `git pull` (repo not cloned), missed Python install, missed pip install. Took 5 corrections to get right.

---

## Principle 2: Multi-Step Commands Must Be Separated and Numbered

**Rule**: Never chain multi-step processes into one line with `&&`. Separate into numbered steps, each on its own line with a comment explaining what it does. Each step must be independently copy-pasteable.

**Why**: One-liner commands are unreadable, hard to debug when one step fails, and impossible to run partially. Users need to see what each step does.

**How to verify**:
1. Count the number of distinct operations in your command.
2. If more than 2 → split into numbered steps.
3. Each step gets a `# comment` explaining its purpose.

**Origin**: Chained `apt install && pip install && git clone && nohup python3` into one unreadable line when user needed to set up a fresh server.

---

## Principle 3: Maximum 1 Correction Per Task — Think First, Respond Second

**Rule**: Before every response, pause and ask: "What server? What's installed? What state is the repo in? What does the user already have? What could go wrong?" If the user has to correct you more than once on the same task, that's a failure.

**Why**: Repeated corrections destroy trust and waste the user's time. The user is not your debugger.

**How to verify**:
1. Re-read the user's message and the conversation context.
2. Identify ALL assumptions you're making.
3. Verify each assumption against known facts (environment, prior messages, copilot-instructions).
4. If uncertain about anything, ask ONE clarifying question — don't guess.

**Origin**: Multiple instances of giving wrong/incomplete answers that required 3-5 rounds of correction across different tasks.

---

## Principle 4: Default to Parallel Execution for Independent Work Units

**Rule**: When writing code that processes independent items (folds, batches, files, API calls), default to parallel execution using `joblib.Parallel`, `multiprocessing`, or equivalent. Never write a sequential `for` loop for independent work without justification.

**Why**: Sequential processing of independent work wastes compute. A 5x speedup (17h → 3.5h) should never require the user to ask "why aren't we running these in parallel?"

**How to verify**:
1. For every loop you write, ask: "Are iterations independent?"
2. If yes → use parallel execution by default.
3. Check available cores (`os.cpu_count()`) and memory to confirm feasibility.
4. Add `n_jobs` parameter for configurability.

**Origin**: Wrote sequential for-loop for 5-Fold CV (45 independent retrains). User waited 17 hours for something that takes 3.5 hours with parallelism. Only fixed after user asked why.

---

## Principle 5: Verify Push With git fetch After Every Push

**Rule**: After every `git push`, immediately run `git fetch origin` and verify with `git --no-pager log --oneline origin/main -1`. Use `git push origin main` (not explicit URL) to ensure tracking refs update. Never say "pushed" without verification.

**Why**: Using explicit push URLs doesn't update local remote tracking refs. `git status` then shows "ahead of origin" even though the push succeeded, creating confusion.

**How to execute** (every time, no exceptions):
1. `git push origin main`
2. `git fetch origin`
3. `git --no-pager log --oneline origin/main -1`
4. Confirm the latest commit hash matches HEAD.

**Origin**: Said "pushed" but `git status` showed "ahead of origin by 3 commits". User thought push failed. Tracking ref was stale because explicit URL was used instead of named remote.

---

## Principle 6: Commit and Push After EVERY Change — No Exceptions

**Rule**: Every file creation, modification, or deletion must be followed by `git add` → `git commit` → `git push` → verify. This is Rule 1 in copilot-instructions.md. Never say "done" without having pushed.

**Why**: Unpushed changes are invisible to other servers and collaborators. The user expects that "done" means "committed, pushed, and verifiable on GitHub."

**How to execute**:
1. Make the change.
2. `git add <files>`
3. `git commit -m "type: description"` (with Co-authored-by trailer)
4. `git push origin main`
5. `git fetch origin && git --no-pager log --oneline origin/main -1`
6. THEN say "done."

**Origin**: Copilot instructions Rule 1 states "Commit and push after every change, even small ones." This must be followed without exception.

---

## Principle 7: Load Lessons and Skills Before Every Task

**Rule**: Before each task, to-do item, or user instruction:
1. Read `.claude/task/lessons.md` (this file) — FIRST PRIORITY.
2. Scan `.claude/skills/` and load skills relevant to the current task type.

**Why**: Lessons prevent repeating past mistakes. Skills contain domain-specific procedures. Skipping either leads to non-standard approaches and rework.

**How to verify**:
1. At the start of each message: read lessons.md.
2. Identify task type → find matching skills → read their SKILL.md.
3. Example: committing → load `git-commit` + `conventional-commit`. Writing PRD → load `prd`.

**Origin**: User explicitly requested this system after repeated mistakes that lessons would have prevented.

---

## Principle 8: Validate Hardcoded Values Against All Datasets Before Training

**Rule**: When writing training code with hardcoded hyperparameter grids, verify that the values make sense for ALL datasets the code will run on — not just the one you're currently looking at. Auto-derive data-dependent values from the dataset itself.

**Why**: Hardcoding contamination grid as `[0.10, 0.15, 0.20]` worked for GAN data (21.6% fraud) but is wrong for parametric data (15% fraud). The user ran parametric training on another server with these wrong values — wasted hours of compute. This should have been caught when the grid was first written.

**How to verify**:
1. Before hardcoding any data-dependent value, ask: "Does this hold for parametric AND GAN AND future datasets?"
2. If the answer is no → derive it from the data (e.g., `fraud_rate = df["label"].mean()`).
3. Before giving the user a training command, verify the code's assumptions match the target dataset.

**Origin**: Contamination grid was hardcoded at `[0.10, 0.15, 0.20]`. User ran parametric training (15% fraud) on another server. Only discovered the mismatch when user asked about it — after training was already running.

---

## Principle 9: New Server = Full Setup Commands — Never Assume Anything Exists

**Rule**: When the context is "other server" or any new/fresh server, ALWAYS include full setup commands: git config (user.email, user.name), dependency installation, directory creation, etc. Never assume git identity, packages, or environment are configured.

**Why**: New servers have no git identity, no pip packages, no project-specific config. Giving a `git commit` command without `git config` first causes an immediate error and wastes the user's time.

**How to verify**: Before giving commands for another server, mentally check: (1) Is git configured? (2) Are dependencies installed? (3) Does the repo exist? (4) Are environment variables set? If ANY of these are uncertain, include the setup commands.

**Origin**: User pushed parametric training results from a new server — `git commit` failed because git identity was not configured. This is the second time incomplete commands were given for a new server (first was missing apt install + git clone + pip install).
