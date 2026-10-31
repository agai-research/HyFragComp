# trap

- mode: **full**, rho = 1.00, sub-goals 4, to recompose 0, repair rounds 0
- LLM calls: 5; latency 20329 ms (LLM part analytic)
- verification: C1=True, C2=True, C3=True, C4=True, C5=True

## Decisions

- g0 `MAR.longstay/elig/first`: **reuse** -> MAR.longstay.elig_f
- g1 `MAR.longstay/docs/`: **reuse** -> MAR.longstay.docs
- g2 `MAR.longstay/appt/`: **switch** -> MAR.longstay.appt_p, MAR.longstay.fee
- g3 `MAR.longstay/sub/`: **reuse** -> MAR.longstay.sub

## Explanation

assess whether I am eligible to apply for the long-stay resi: reused from record c-58db00e5 (tier C_e), match 0.99 = 0.49 + 0.30 + 0.20, valid at t=4.5 within window [0, 10000000]
validate the supporting documents for the long-stay residenc: reused from record c-58db00e5 (tier C_e), match 0.97 = 0.47 + 0.30 + 0.20, valid at t=4.5 within window [0, 10000000]
book the earliest priority appointment for the long-stay res: switched variant of from record c-58db00e5 (tier C_e), match 0.82 = 0.33 + 0.29 + 0.20, valid at t=4.5 within window [0, 10000000], ['MAR.longstay.appt_s', 'MAR.longstay.fee'] replaced by ['MAR.longstay.appt_p', 'MAR.longstay.fee'] because the guard no longer holds
submit my long-stay residence permit file in Morocco: reused from record c-58db00e5 (tier C_e), match 0.99 = 0.50 + 0.29 + 0.20, valid at t=4.5 within window [0, 10000000]

## Figures

- trap_workflow.png
- trap_workflow.svg
- trap_goals.png
- trap_goals.svg