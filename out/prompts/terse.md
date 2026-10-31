# terse

- mode: **partial**, rho = 0.75, sub-goals 4, to recompose 1, repair rounds 0
- LLM calls: 6; latency 11378 ms (LLM part analytic)
- verification: C1=True, C2=True, C3=True, C4=True, C5=True

## Decisions

- g0 `MAR.longstay/elig/first`: **new** -> MAR.longstay.elig_f
- g1 `MAR.longstay/docs/`: **reuse** -> MAR.longstay.docs
- g2 `MAR.longstay/appt/`: **switch** -> MAR.longstay.appt_s, MAR.longstay.fee
- g3 `MAR.longstay/sub/`: **reuse** -> MAR.longstay.sub

## Explanation

long-stay residence permit Morocco: verify eligibility: composed anew with ['MAR.longstay.elig_f'] (previous reuse refused: V3)
verify documents for the long-stay residence permit in Moroc: reused from record c-274325cc (tier C_e), match 1.00 = 0.50 + 0.30 + 0.20, valid at t=4.0 within window [0, 10000000]
book appointment for the long-stay residence permit in Moroc: switched variant of from record c-274325cc (tier C_e), match 0.91 = 0.42 + 0.29 + 0.20, valid at t=4.0 within window [0, 10000000], ['MAR.longstay.appt_p', 'MAR.longstay.fee'] replaced by ['MAR.longstay.appt_s', 'MAR.longstay.fee'] because the guard no longer holds
submit application for the long-stay residence permit in Mor: reused from record c-274325cc (tier C_e), match 0.76 = 0.47 + 0.29 + 0.00, valid at t=4.0 within window [0, 10000000]
rejected ['MAR.longstay.elig_r'] by chi_fun

## Figures

- terse_workflow.png
- terse_workflow.svg
- terse_goals.png
- terse_goals.svg