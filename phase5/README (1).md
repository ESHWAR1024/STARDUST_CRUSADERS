# Phase 5 — Synthetic Data Generation Engine

Part of the **Automated Bank Statement Analysis System** built for CIDECODE 2026 (CID Karnataka).

---

## Purpose

This engine generates a realistic synthetic banking dataset that:

- Has fraud **genuinely hidden** inside realistic financial noise — not toy isolated examples. A judge skimming the raw transaction table should not be able to spot the fraud ring by eye.
- Stresses the **Phase 6 multi-format ingestion pipeline** by exporting statements in six different bank layouts, three file formats (CSV / Excel / scanned PNG), with different column names, date formats, and header structures per bank.
- Produces **cleanly separated ground truth** (fraud labels in a separate folder never seen by the investigation system) so detection performance can be measured with precision/recall against a known answer key.

---

## Quick Start

```bash
pip install faker pandas numpy openpyxl pillow
python generate.py                        # 500 accounts, writes to ./output/
python generate.py --n-accounts 1000 --out-dir data/run1
```

All randomness is seeded (`RANDOM_SEED = 42` in `config.py`) — the same run always produces the same dataset, which is important for demo reproducibility.

---

## Output Structure

```
output/
├── master_transactions_clean.csv       ← unified schema, NO fraud labels
├── accounts_master.csv                 ← account roster (all personas + fraud roles)
├── sample_bank_statements/             ← "as received" per-bank statement exports
│   ├── statement_ACC000XXX_HDFC.csv
│   ├── statement_ACC000XXX_ICICI.xlsx
│   ├── statement_ACC000XXX_SBI_scanned.png
│   └── ...
└── ground_truth/                       ← EVALUATION ONLY — never fed to the investigation system
    ├── account_labels.csv              ← is_fraud, fraud_role, fraud_ring_id per account
    ├── transaction_labels.csv          ← is_fraud, fraud_pattern per transaction
    └── ring_summary.csv               ← ring_id, typology, participating account list
```

---

## Data Generation Methodology

### Population Simulation

The engine builds a population of **N accounts** (default 500) split across five economic personas plus a hidden fraud-role cohort (~6% of total).

| Persona | Mix | Income Model | Key Behaviors |
|---|---|---|---|
| salaried | 38% | Monthly NEFT salary (log-uniform ₹22K–₹180K) | Salary + rent + EMI + daily UPI |
| family | 22% | Monthly household income (₹28K–₹220K) | Income + rent + utilities + household spend |
| student | 18% | Monthly IMPS allowance (₹3K–₹15K) | Allowance + frequent small UPI |
| shopkeeper | 12% | Customer collections (Poisson 6.5/day, ₹100–₹6K each) | High-frequency inbound + supplier payments |
| business | 10% | Customer collections (Poisson 5/day, ₹200–₹9K each) | Same as shopkeeper, higher amounts |

Fraud-role accounts (mule / collector / fresh\_layering\_node) are pre-planned with exact counts from the ring layout before entity generation, so every ring always gets its full allocation.

### Probability Distributions

**Opening balances**: Log-uniform per persona (e.g. salaried ₹15K–₹2.5L). Log-uniform is used because real account balances are heavy-tailed — a few accounts hold much more than the median.

**Salaries / income**: Log-uniform within persona band. Real salary distributions within a job category cluster near a median but have a long upper tail.

**Daily transaction count**: Poisson process with persona-specific rate λ (e.g. λ=0.9/day for salaried, λ=6.5/day for shopkeeper). Poisson is appropriate because UPI transactions are roughly independent arrivals.

**Transaction amounts**: Log-uniform within persona/category band. Prevents amounts clustering at the midpoint and preserves realistic variability.

**ATM withdrawal probability**: Bernoulli trial once per week (p=0.60), scaled by persona (students 0.25× base, businesses 1.3× base).

**EMI amount**: Sized as 8–22% of monthly income (not a flat range), so a ₹22K salary account can't be assigned a ₹30K EMI.

### Time Distributions

**Time of day**: Sampled from a 24-bucket empirical weight array derived from Indian UPI usage patterns — peaks at 12–14h and 18–21h, near-zero 01–05h.

**Salary credit day**: Exponential-right-skewed in [1, 5] — most salaries land on the 1st or 2nd, tail to the 5th.

**Recurring bills**: Fixed-day-range jitter (rent: 1–3, utilities: 8–20, EMI: 5–10) to avoid all obligations landing identically each month.

**Fraud timing**: 35% probability of fraud hops being forced into 00:00–05:00 (odd-hour bias — a real investigator heuristic, kept probabilistic rather than deterministic so it's a soft signal, not a trivial label).

### Behavioral Simulation

Every legitimate account runs a complete 90-day event simulation. A **spending floor** is applied per account: discretionary UPI spend and ATM withdrawals are scaled down (never fully suppressed) when the running balance approaches the persona's floor (e.g. −₹3K for family, −₹1K for student), so accounts can't be driven arbitrarily into overdraft by random spending alone.

**~15% of discretionary transfers** resolve to another account within the synthetic population (linked debit+credit pair sharing a UTR reference), creating real internal graph structure for the graph-analytics layer (Phase 9). The remaining 85% resolve to named external counterparties that only exist in narration text, mirroring how most real bank statement lines reference parties not in any seized account list.

**Business and shopkeeper inflows** (customer collections) are modeled separately from their personal discretionary spend, with ~25% of those collections generating linked transfers from salaried/family account counterparts, creating a realistic merchant payment sub-graph.

---

## Fraud Typologies

The engine guarantees one ring of **each** typology when `N_FRAUD_RINGS ≥ 4` (via a pre-shuffle-then-fill-extras approach), so the demo always shows all four patterns regardless of random sampling.

### 1. Layering Chain (`layering_chain`)

```
Victim proceeds → Node₀ → Node₁ → ... → Nodeₙ → EXIT
```

Sequential hop-to-hop movement with 2–8% skim at each hop (launderer's commission) and delays of 0.5–18 hours between hops. Final node exits via ATM/remittance/crypto. **Key signal**: rapid sequential transfers with decaying amounts, all nodes showing near-zero legitimate activity, odd-hour timestamps.

### 2. Fan-Out Mule Network (`fan_out_mule_network`)

```
Victim proceeds → HUB → Mule₁ → EXIT
                      → Mule₂ → EXIT
                      → ...
                      → Muleₙ → EXIT
```

Large lump sum received by a hub account, split within minutes across 4–12 mule legs with randomly weighted shares. Each mule rapidly withdraws 75–97% of what it received. **Key signal**: single large credit followed by many rapid small debits to different accounts; mules each show one large credit, one large ATM withdrawal, nothing else.

### 3. Fan-In Collector (`fan_in_collector`)

```
Victim₁ → Mule₁ ↘
Victim₂ → Mule₂ → COLLECTOR → EXIT
...
Victimₙ → Muleₙ ↗
```

Many small proceeds inflows each routed through an individual mule, then all converging on a single collector account within a tight time window (4–72 hours). Collector exits the pooled sum. **Key signal**: many accounts crediting the same beneficiary in a burst; collector account has no prior activity.

### 4. Smurfing / Structuring (`smurfing_structuring`)

```
Large sum broken into N sub-threshold transfers:
Source₁ → BENEFICIARY (₹X < ₹50K threshold)
Source₂ → BENEFICIARY (₹Y < ₹50K threshold)
... spread over 3–14 days ...
Sourceₙ → BENEFICIARY
                → EXIT
```

Deliberately keeps each individual transfer below the ₹50,000 scrutiny threshold. Spread across multiple days and multiple source accounts. **Key signal**: same beneficiary receiving many credits each just under a round threshold, from accounts with no other relationship to each other.

---

## Statement Format Diversity

Each bank uses a different column layout, date format, and field naming convention, deliberately reproducing the chaos of real multi-bank seizures:

| Bank | Date Format | "Debit" Column | "Credit" Column | "Balance" Column |
|---|---|---|---|---|
| SBI | `dd/mm/yy` | Debit | Credit | Balance |
| HDFC | `dd/mm/yyyy` | Withdrawal Amt. | Deposit Amt. | Closing Balance |
| ICICI | `dd-Mon-yyyy` | Withdrawal Amount (INR) | Deposit Amount (INR) | Balance (INR) |
| Axis | `yyyy-mm-dd` | Debit | Credit | Balance |
| Canara | `dd-mm-yyyy` | Withdrawals | Deposits | Balance |
| PNB | `dd.mm.yyyy` | Debit | Credit | Balance |

File formats cycle across: CSV (with bank header block above the table), Excel (bank header in rows 1–3, data from row 5), and scanned PNG (rendered table with rotation ±1.2°, Gaussian blur, pixel noise, and low-quality JPEG re-encoding to simulate a phone camera scan).

---

## Module Reference

| File | Responsibility |
|---|---|
| `config.py` | All global parameters and distributions |
| `entities.py` | Account generation (legitimate + fraud-role populations) |
| `behavior_engine.py` | Day-to-day legitimate transaction simulation |
| `fraud_engine.py` | Fraud ring planning and typology injection |
| `ledger.py` | Chronological sort, running balances, transaction ID assignment |
| `statement_formatter.py` | Per-bank format export + scanned PNG rendering |
| `ground_truth.py` | Label-only exports (account/transaction/ring level) |
| `generate.py` | CLI orchestrator |
