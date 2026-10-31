# light_unseen

- mode: **scratch**, rho = 0.00, sub-goals 3, to recompose 3, repair rounds 0
- LLM calls: 9; latency 13696 ms (LLM part analytic)
- verification: C1=True, C2=True, C3=True, C4=True, C5=True

## Decisions

- g0 `MAR.longstay/docs/`: **new** -> MAR.longstay.docs
- g1 `MAR.longstay/appt/`: **new** -> MAR.longstay.appt_s, MAR.longstay.fee
- g2 `MAR.longstay/sub/`: **new** -> MAR.longstay.sub

## Explanation

prepare the documentation file for the long-stay residence p: composed anew with ['MAR.longstay.docs'] (previous reuse refused: no-match)
reserve my appointment for the long-stay residence permit in: composed anew with ['MAR.longstay.appt_s', 'MAR.longstay.fee'] (previous reuse refused: no-match)
hand in my long-stay residence permit application in Morocco: composed anew with ['MAR.longstay.sub'] (previous reuse refused: no-match)
rejected ['MAR.longstay.appt_p'] by guard
rejected ['MAR.longstay.appt_s'] by c4

## Figures

- light_unseen_workflow.png
- light_unseen_workflow.svg
- light_unseen_goals.png
- light_unseen_goals.svg