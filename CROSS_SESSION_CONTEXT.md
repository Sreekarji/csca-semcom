# CSCA-SemCom — Cross-Session Context
## Both chats read this file for full context.

## PROJECT STRUCTURE
- D:\MP2 = Original paper-faithful implementation (show Dr. Joshi this)
- D:\MP2-working = NEW working implementation with fixes (create this)
- D:\Downloads\success\ = Arena AI's diagnostic report + fix code
- GitHub: github.com/Sreekarji/csca-semcom

## STUDENT
- Entering 5th sem ECE, CGPA 9.09
- Dr. Sandeep Joshi, BITS Pilani
- Hardware: RTX 4050 6GB, Python 3.12, Windows

## 7 CRITICAL GAPS (from Arena AI diagnostic)
1. BW-only action (paper uses BW+relay+MCS jointly)
2. No hard negative reward for unmet intent
3. 1 task/CSCA (paper uses 10-20)
4. Wrong log-prob (DDPO-SF vs paper's Eq 28-29)
5. TD bootstrap vs MC return
6. Global graph needed (one HAN, one DDPM for whole network)
7. MCS selection contradictory (deterministic MIM vs learned DDPM)

## CURRENT RESULTS
All methods produce identical ISR ~0.25. No differentiation.

## FILES IN D:\Downloads\success\
- CSCA-SemCom Diagnostic Report v2 —.txt (220KB) — full analysis
- gap_analysis.md (37KB) — detailed discrepancies
- IMPLEMENTATION_GUIDE.md (8.5KB) — step-by-step fix guide
- cscqi_hard_negative.py (7.7KB) — hard negative CSCQI fix
- ddpm_policy_fixed.py (14.8KB) — expanded action space + paper's log-prob
- oracle_gap_test.py (8.9KB) — oracle gap verification
- train_han_mlp.py (12KB) — minimum viable HAN+MLP training

## WHAT TO DO
1. Original project (this chat): keep D:\MP2 as paper-faithful
2. New project (other chat): create D:\MP2-working with fixes from D:\Downloads\success\
3. Both share channel model, graph builder, evaluation code
4. New project documents deviations from paper honestly

## ARENA AI PROMPT (for new chat)
Upload files from D:\Downloads\success\ to Arena AI (arena.ai/agent).
Combine .py files into one .txt for upload.
Ask for: methodology first, then code. Research web, repos, papers.
