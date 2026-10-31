# scenario

- mode: **scratch**, rho = 0.00, sub-goals 5, to recompose 5, repair rounds 0
- LLM calls: 11; latency 20165 ms (LLM part analytic)
- verification: C1=True, C2=True, C3=True, C4=True, C5=True

## Decisions

- g0 `MAR.longstay/elig/renew`: **new** -> MAR.longstay.elig_r
- g1 `MAR.longstay/docs/`: **new** -> MAR.longstay.docs
- g2 `MAR.longstay/extra/`: **new** -> MAR.longstay.extra_a
- g3 `MAR.longstay/appt/`: **new** -> MAR.longstay.appt_p, MAR.longstay.fee
- g4 `MAR.longstay/sub/`: **new** -> MAR.longstay.sub

## Explanation

verify my renewal eligibility for the long-stay residence pe: composed anew with ['MAR.longstay.elig_r'] (previous reuse refused: no-match)
check my documents for the long-stay residence permit in Mor: composed anew with ['MAR.longstay.docs'] (previous reuse refused: no-match)
validate my proof of funds from the new employment contract : composed anew with ['MAR.longstay.extra_a'] (previous reuse refused: no-match)
book the earliest priority appointment for the long-stay res: composed anew with ['MAR.longstay.appt_p', 'MAR.longstay.fee'] (previous reuse refused: no-match)
notify me by <CHANNEL> for the long-stay residence permit in: composed anew with ['MAR.longstay.sub'] (previous reuse refused: no-match)
rejected ['MAR.longstay.elig_f'] by chi_fun
rejected ['MAR.longstay.extra_b'] by chi_rol
rejected ['MAR.longstay.appt_s'] by guard
rejected ['MAR.longstay.appt_p'] by c4

## Figures

- scenario_workflow.png
- scenario_workflow.svg
- scenario_goals.png
- scenario_goals.svg