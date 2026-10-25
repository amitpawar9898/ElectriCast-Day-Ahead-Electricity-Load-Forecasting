You are improving ElectriCast in this repo. You have no memory of earlier
iterations; the files are your memory. Read, in this order:
  1. PLAN.md        (the checklist, your source of truth)
  2. PROGRESS.md    (log of past iterations; create it if missing)
  3. DECISIONS.md   (choices already made; create it if missing)
  4. PROJECT_AUDIT.md (facts about the original code; verify line numbers
     before editing, since earlier iterations may have moved things)

THIS ITERATION
1. Pick the FIRST unchecked item in PLAN.md. Do only that item. If it is
   too big for one pass, split it into sub-items in PLAN.md (1.4a, 1.4b),
   do the first, and stop.
2. Before editing, state in 2-3 lines what you will change and how you will
   know it worked.
3. Make the change. Keep the diff small and focused.
4. VERIFY, with real evidence and not by reading your own code:
   - Backend: run pytest (if tests exist) and the relevant script; paste the
     key output lines.
   - Frontend: run `npm run build` in frontend/; it must pass with no errors.
     For UI items, also start the app and confirm the specific behavior
     (fetch the API route with curl, or describe what the page renders).
   - Training/eval items: run it and record the actual numbers.
   If verification fails, fix it and re-verify. Spend at most 3 fix attempts.
5. If it passes: tick the box in PLAN.md, append to PROGRESS.md
   (date, item id, what changed, evidence, any numbers), and make one git
   commit: "<item id>: <short summary>".
6. If it still fails after 3 attempts: do NOT tick it. Write a BLOCKED entry
   in PROGRESS.md (what you tried, the exact error, your best guess), commit
   nothing broken (git stash or revert), mark the item "[!]" in PLAN.md,
   and move on to the next item next time.
7. Stop. One item per iteration.

RULES
- Units are Wh internally; display kWh or label Wh. Timezone is IST.
- Every model is judged under the same protocol, same windows, same
  information. Never give one model data another can't see.
- Numbers shown in the UI come from backend JSON; never hardcode a result.
- Never present a placeholder or untrained model as a result.
- Never delete files under data/, and never rewrite git history.
- Do not install packages that aren't in requirements.txt/package.json
  without adding them there (pinned) and noting it in DECISIONS.md.
- If a choice is ambiguous, pick the more conservative option, log one line
  in DECISIONS.md, and continue. Do not stop to ask unless the action is
  destructive.
- Be honest in RESULTS and PROGRESS: if a model loses to a baseline, record
  that plainly.

COMPLETION
When every item in PLAN.md is [x] or [!], write FINAL_REPORT.md (what was
done, results table with baselines vs models, every [!] item with its reason,
remaining risks), commit it, and print exactly this line on its own:
ALL_DONE