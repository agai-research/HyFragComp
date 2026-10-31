"""Domain lexicon: administrative domains, step phrase templates, filler text, personal-data pools."""

# slug: (display, kind, extra document, alternative extra document)
DOMAINS = {
    "studperm": ("student residence permit", "permit", "enrollment letter", "sponsor guarantee"),
    "workperm": ("work residence permit", "permit", "employment contract", "employer declaration"),
    "famperm": ("family reunification permit", "permit", "proof of kinship", "housing certificate"),
    "invperm": ("investor residence permit", "permit", "investment proof", "bank guarantee"),
    "resperm": ("researcher residence permit", "permit", "hosting agreement", "funding letter"),
    "longstay": ("long-stay residence permit", "permit", "proof of funds", "insurance certificate"),
    "retperm": ("retirement residence permit", "permit", "pension certificate", "health coverage letter"),
    "seasperm": ("seasonal worker permit", "permit", "seasonal contract", "farm declaration"),
    "nomad": ("remote worker residence permit", "permit", "remote income statement", "client letter"),
    "startvisa": ("startup founder visa", "permit", "business plan", "incubator letter"),
    "llc": ("limited liability company registration", "company", "articles of association", "capital certificate"),
    "sole": ("sole proprietorship registration", "company", "activity declaration", "premises lease"),
    "jsc": ("joint-stock company registration", "company", "shareholder register", "board resolution"),
    "branch": ("foreign company branch registration", "company", "parent company certificate", "legal representative mandate"),
    "coop": ("cooperative registration", "company", "members list", "bylaws approval"),
    "ngo": ("non-profit association registration", "company", "founders list", "statute draft"),
    "partner": ("partnership registration", "company", "partnership agreement", "partner identity file"),
    "digstart": ("digital startup registration", "company", "founder identity record", "platform terms"),
    "buslic": ("business operating licence", "licence", "premises inspection report", "zoning certificate"),
    "taxid": ("tax identification registration", "registration", "activity statement", "address proof"),
    "socaff": ("social security affiliation", "registration", "payroll declaration", "employee list"),
    "vat": ("VAT registration", "registration", "turnover estimate", "accounting certificate"),
    "implic": ("import licence", "licence", "customs classification", "supplier invoice"),
    "explic": ("export licence", "licence", "buyer contract", "product certificate"),
    "customs": ("customs operator registration", "registration", "warehouse declaration", "guarantee bond"),
    "foodsaf": ("food safety permit", "permit", "hygiene inspection report", "supplier traceability file"),
    "envclr": ("environmental clearance", "permit", "impact assessment", "waste plan"),
    "bldperm": ("building permit", "permit", "architectural drawings", "structural study"),
    "landreg": ("land title registration", "registration", "survey plan", "ownership deed"),
    "trademark": ("trademark registration", "registration", "mark specimen", "classification sheet"),
    "patent": ("patent registration", "registration", "technical description", "inventor declaration"),
    "drivlic": ("driving licence", "licence", "medical certificate", "driving school record"),
    "vehreg": ("vehicle registration", "registration", "purchase invoice", "insurance policy"),
    "idcard": ("national identity card", "document", "birth extract", "residence proof"),
    "passport": ("passport issuance", "document", "photo specification", "previous passport"),
    "birth": ("birth certificate", "document", "hospital notice", "parent identity file"),
    "marriage": ("marriage registration", "document", "witness statements", "celibacy certificate"),
    "school": ("school enrollment", "registration", "previous school record", "vaccination card"),
    "health": ("health insurance enrollment", "registration", "income statement", "family composition"),
    "pension": ("pension claim", "benefit", "contribution history", "retirement notice"),
    "unemp": ("unemployment benefit claim", "benefit", "termination letter", "job search record"),
    "disab": ("disability allowance claim", "benefit", "medical assessment", "care plan"),
    "fishlic": ("fishing licence", "licence", "boat registration", "catch declaration"),
    "taxilic": ("taxi operator licence", "licence", "vehicle inspection", "criminal record extract"),
    "pharm": ("pharmacy licence", "licence", "pharmacist diploma", "premises plan"),
    "medlic": ("medical practice licence", "licence", "diploma recognition", "malpractice insurance"),
    "teachacc": ("teacher accreditation", "licence", "teaching diploma", "reference letters"),
    "notary": ("notarial deed registration", "registration", "deed draft", "parties identity file"),
    "courtfile": ("civil court filing", "registration", "claim statement", "evidence bundle"),
    "grant": ("small business grant application", "benefit", "budget plan", "tax clearance"),
    "subsidy": ("agricultural subsidy application", "benefit", "land declaration", "crop plan"),
    "utility": ("utility connection request", "registration", "property title", "technical survey"),
    "telecom": ("telecom operator authorisation", "licence", "network plan", "spectrum request"),
    "bank": ("bank account opening for companies", "registration", "beneficial owner declaration", "company extract"),
    "insolv": ("insolvency filing", "registration", "creditor list", "asset inventory"),
    "contract": ("public procurement registration", "registration", "supplier certificate", "past performance record"),
    "hotel": ("tourism accommodation licence", "licence", "fire safety certificate", "classification report"),
    "event": ("public event permit", "permit", "security plan", "venue agreement"),
    "mining": ("small mining concession", "permit", "geological survey", "rehabilitation bond"),
    "forest": ("forestry harvesting permit", "permit", "management plan", "transport declaration"),
    "water": ("water extraction permit", "permit", "hydrological study", "usage declaration"),
    "energy": ("solar installation permit", "permit", "grid study", "installer certificate"),
    "adopt": ("child adoption registration", "document", "social report", "consent record"),
    "citizen": ("naturalisation application", "document", "language certificate", "residence history"),
    "veteran": ("veteran benefit claim", "benefit", "service record", "medical file"),
}

STEP_ORDER = ["elig", "docs", "extra", "appt", "sub"]

# request-side phrasings; {dom} {eco} {extra} {channel} are slots
PHRASES = {
    "elig_first": ["verify my eligibility for a first {dom} in {eco}",
                   "check that I qualify for a new {dom} in {eco}",
                   "assess whether I am eligible to apply for the {dom} in {eco} for the first time",
                   "confirm my eligibility for the {dom} in {eco}"],
    "elig_renew": ["verify my renewal eligibility for the {dom} in {eco}",
                   "check that I can renew my {dom} in {eco}",
                   "assess whether my {dom} in {eco} can be renewed",
                   "confirm I am eligible to renew the {dom} in {eco}"],
    "docs": ["collect and verify my documents for the {dom} in {eco}",
             "upload and validate the supporting documents for the {dom} in {eco}",
             "gather the required paperwork for the {dom} in {eco} and have it verified",
             "prepare the documentation file for the {dom} in {eco}"],
    "extra": ["validate my {extra} for the {dom} in {eco}",
              "check the {extra} required for the {dom} in {eco}",
              "have my {extra} reviewed for the {dom} in {eco}",
              "confirm the {extra} for the {dom} in {eco}"],
    "appt": ["pay the fee and book an appointment for the {dom} in {eco}",
             "settle the fees and schedule an appointment for the {dom} in {eco}",
             "pay the application fee and arrange my appointment for the {dom} in {eco}",
             "book my appointment for the {dom} in {eco} and pay the fee"],
    "appt_prio": ["pay the fee and book the earliest priority appointment for the {dom} in {eco}",
                  "settle the fees and obtain an expedited appointment for the {dom} in {eco}",
                  "pay the fee and book a priority slot for the {dom} in {eco}",
                  "book the fastest available appointment for the {dom} in {eco} and pay the fee"],
    "sub": ["submit the {dom} application in {eco} and notify me by {channel}",
            "send in my {dom} application in {eco} and confirm by {channel}",
            "file the {dom} application in {eco} and send me a {channel} notification",
            "submit my {dom} file in {eco} and let me know by {channel}"],
}

# repository-side fragment descriptions
FRAG_TEXT = {
    "ent": "Identity check and dossier opening at the {eco} administration",
    "elig_f": "Eligibility assessment for the first application of a {dom} in {eco}",
    "elig_r": "Renewal eligibility assessment for an existing {dom} in {eco}",
    "docs": "Document collection and verification for a {dom} in {eco}",
    "fee": "Fee payment and receipt for a {dom} in {eco}",
    "extra_a": "Validation of the {extra} for a {dom} in {eco}",
    "extra_b": "Employer agent review of the {extra_alt} for a {dom} in {eco}",
    "appt_s": "Standard appointment booking for a {dom} in {eco}",
    "appt_p": "Priority expedited appointment booking for a {dom} in {eco}",
    "sub": "Submission of the {dom} application and applicant notification in {eco}",
}

CHANNELS = ["SMS", "email", "postal letter", "mobile app"]

FILLER = [
    "I arrived in the country last spring and my family is still waiting for news.",
    "The rent increased again this year and the neighbourhood has become noisy.",
    "My employer has been very patient about the delay and the rest of the team is supportive.",
    "I work in a small office near the harbour and take the tram every morning.",
    "My sister recommended that I ask about all of this before the summer holidays.",
    "Last week the weather was terrible and the trains were running late.",
    "I have been learning the local language in the evening and it is going well.",
    "The kids are settled in their new school and they like their teacher.",
    "I tried the website yesterday but the page kept freezing on my old laptop.",
    "A colleague told me the process used to take much longer a few years ago.",
    "I would appreciate clear instructions because English is not my first language.",
    "The office closes early on Fridays so I need to plan carefully.",
    "I am travelling for work next month and will be reachable only by phone.",
    "My previous experience with administration in another country was frustrating.",
    "The city centre is crowded during the festival season and parking is difficult.",
    "I also need to renew my gym membership and fix the heating at home.",
    "Thank you in advance for your help with this matter.",
    "I keep all my paperwork in a blue folder on my desk.",
    "My cousin runs a small bakery and asked me for advice about this.",
    "I read the guidelines twice but some sentences remain unclear to me.",
]

# fillers with trigger words that make naive decomposition over-segment
TRAP_FILLER = [
    "I also have an appointment with my dentist next week and need to pay the invoice.",
    "My neighbour asked me to submit his parking complaint but that is unrelated.",
    "I need to collect my parcel and verify the delivery documents at the post office.",
    "Please do not book anything for my brother, he already has his own appointment.",
    "A friend wondered whether she is eligible for something similar and wants to verify that.",
    "My cousin might check his eligibility next year but that is a separate matter.",
    "I should also confirm whether my gym membership qualifies for the corporate discount.",
    "Someone at work suggested I submit a complaint about the parking fees, which I will ignore.",
]

FIRST = ["Amine", "Sara", "Karim", "Leila", "Omar", "Nadia", "Youssef", "Ines", "Mehdi", "Salma",
         "Hamza", "Rania", "Walid", "Meriem", "Anis", "Dorra", "Fares", "Lina", "Sami", "Yasmine"]
LAST = ["Ben Salah", "Trabelsi", "Gharbi", "Mansour", "Jebali", "Haddad", "Karoui", "Dridi", "Slimani", "Chebbi"]
CITIES = ["Tunis", "Sfax", "Sousse", "Lyon", "Turin", "Porto", "Gdansk", "Leeds", "Ghent", "Seville"]


def domain_text(slug):
    return DOMAINS[slug][0]


# word-level substitutions used to build phrasings the sentence encoder never saw
SYNONYMS = {
    "verify": "ascertain", "check": "inspect", "confirm": "make sure of", "assess": "evaluate", "collect": "assemble",
    "gather": "pull together", "upload": "provide", "prepare": "put together", "validate": "authenticate", "pay": "clear",
    "settle": "discharge", "book": "reserve", "schedule": "set up", "arrange": "organise", "submit": "lodge", "send in": "hand in",
    "file": "register", "notify": "inform", "documents": "papers", "paperwork": "formalities", "appointment": "meeting",
    "fee": "charge", "fees": "charges", "eligibility": "entitlement", "eligible": "entitled", "qualify": "meet the conditions",
}

# words that mark a clause as concerning somebody else or an unrelated errand (the planner does not turn it into a goal)
OFFTOPIC = r"\b(neighbou?r\w*|brother|sister|cousin|friend|colleague|someone|somebody|wife|husband|dentist|gym|parcel|parking|post office|unrelated|ignore|separate matter|corporate discount|next year|do not|don't)\b"
