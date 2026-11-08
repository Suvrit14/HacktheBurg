// Demo GOV.UK Design System service: "Apply for a disabled parking badge (demo)".
//
// Follows the GOV.UK "one thing per page" pattern: every question has its own
// URL, a Back link, server-side validation with an error summary, and the
// answer is saved before redirecting to the next page (POST, then GET).
// Answers live in an in-memory session; nothing leaves this computer.

const path = require("node:path")
const crypto = require("node:crypto")
const express = require("express")
const session = require("express-session")
const nunjucks = require("nunjucks")

const PORT = process.env.PORT || 3000
const SERVICE_NAME = "Apply for a disabled parking badge (demo)"
const GOVUK_DIST = path.join(__dirname, "node_modules/govuk-frontend/dist")

const app = express()

nunjucks.configure([path.join(__dirname, "views"), GOVUK_DIST], {
  autoescape: true,
  express: app
})
app.set("view engine", "njk")

// Own icons first (they replace GOV.UK's, same filenames), then GOV.UK assets.
app.use("/assets", express.static(path.join(__dirname, "public/assets")))
app.use("/assets", express.static(path.join(GOVUK_DIST, "govuk/assets")))
app.use("/govuk", express.static(path.join(GOVUK_DIST, "govuk")))
app.use(express.static(path.join(__dirname, "public"), { index: false }))
app.use(express.urlencoded({ extended: true }))
app.use(
  session({
    name: "badge_session",
    secret: crypto.randomBytes(32).toString("hex"),
    resave: false,
    saveUninitialized: false,
    cookie: { httpOnly: true, sameSite: "lax" }
  })
)
app.use((req, res, next) => {
  res.locals.serviceName = SERVICE_NAME
  req.session.answers = req.session.answers || {}
  next()
})

// ---------------------------------------------------------------- answers

const WHO_FOR = { myself: "Myself", "someone-else": "Someone else" }
const BENEFITS = {
  pip: "Personal Independence Payment (PIP)",
  dla: "Disability Living Allowance (DLA) higher rate mobility component",
  wpms: "War Pensioners' Mobility Supplement",
  afcs: "Armed Forces Compensation Scheme (tariff levels 1 to 8)",
  none: "None of these"
}
const WALKING = {
  "less-than-50": "Less than 50 metres",
  "50-to-100": "Between 50 and 100 metres",
  "more-than-100": "More than 100 metres",
  "cannot-walk": "I cannot walk at all"
}
const CONTACT = { email: "Email", phone: "Phone", post: "Post" }
const CONDITION_MAX = 500

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

const clean = (value) => (typeof value === "string" ? value.trim() : "")
const asList = (value) => (value === undefined ? [] : [].concat(value))

// ------------------------------------------------------------- validation
// Each validator returns { errors: { inputId: message }, answer }.
// Messages follow the "Error messages" guidance on each component page.

function validateWhoFor(body) {
  const whoFor = clean(body.whoFor)
  if (!WHO_FOR[whoFor]) return { errors: { "who-for": "Select who you are applying for" } }
  return { answer: { whoFor } }
}

function validateBenefits(body) {
  const benefits = asList(body.benefits).filter((b) => BENEFITS[b])
  if (benefits.length === 0 || (benefits.includes("none") && benefits.length > 1)) {
    return {
      errors: { benefits: "Select which benefits you get, or select 'None of these'" },
      values: { benefits: benefits.filter((b) => b !== "none") }
    }
  }
  return { answer: { benefits } }
}

function validateWalking(body) {
  const walking = clean(body.walking)
  if (!WALKING[walking]) return { errors: { walking: "Select how far you can walk without stopping" } }
  return { answer: { walking } }
}

function validateName(body) {
  const fullName = clean(body.fullName)
  if (!fullName) return { errors: { "full-name": "Enter your full name" } }
  if (fullName.length > 100) return { errors: { "full-name": "Full name must be 100 characters or less" } }
  return { answer: { fullName } }
}

function parseMonth(text) {
  if (/^\d{1,2}$/.test(text)) return Number(text)
  const index = MONTHS.indexOf(text.slice(0, 3).toLowerCase())
  return index === -1 ? NaN : index + 1
}

function validateDateOfBirth(body) {
  const day = clean(body["dob-day"])
  const month = clean(body["dob-month"])
  const year = clean(body["dob-year"])
  const values = { day, month, year }
  const parts = { day, month, year }
  const missing = Object.keys(parts).filter((k) => !parts[k])

  if (missing.length === 3) {
    return { errors: { "dob-day": "Enter your date of birth" }, invalid: ["day", "month", "year"], values }
  }
  if (missing.length > 0) {
    return {
      errors: { [`dob-${missing[0]}`]: `Date of birth must include a ${missing.join(" and ")}` },
      invalid: missing,
      values
    }
  }
  if (!/^\d{4}$/.test(year)) {
    return { errors: { "dob-year": "Year must include 4 numbers" }, invalid: ["year"], values }
  }
  const d = Number(day)
  const m = parseMonth(month)
  const y = Number(year)
  const date = new Date(Date.UTC(y, m - 1, d))
  const real =
    /^\d{1,2}$/.test(day) && date.getUTCFullYear() === y && date.getUTCMonth() === m - 1 && date.getUTCDate() === d
  if (!real) {
    return { errors: { "dob-day": "Date of birth must be a real date" }, invalid: ["day", "month", "year"], values }
  }
  if (date > new Date()) {
    return { errors: { "dob-day": "Date of birth must be in the past" }, invalid: ["day", "month", "year"], values }
  }
  if (y < 1900) {
    return {
      errors: { "dob-day": "Date of birth must be the same as or after 1 January 1900" },
      invalid: ["day", "month", "year"],
      values
    }
  }
  return { answer: { day: String(d), month: String(m), year: String(y) } }
}

const POSTCODE = /^[A-Z]{1,2}[0-9][A-Z0-9]?[0-9][A-Z]{2}$/

function validateAddress(body) {
  const values = {
    line1: clean(body.addressLine1),
    line2: clean(body.addressLine2),
    town: clean(body.addressTown),
    county: clean(body.addressCounty),
    postcode: clean(body.addressPostcode)
  }
  const errors = {}
  if (!values.line1) errors["address-line-1"] = "Enter address line 1, typically the building and street"
  if (!values.town) errors["address-town"] = "Enter town or city"
  const compact = values.postcode.replace(/\s+/g, "").toUpperCase()
  if (!compact) errors["address-postcode"] = "Enter postcode"
  else if (!POSTCODE.test(compact)) errors["address-postcode"] = "Enter a real postcode"
  if (Object.keys(errors).length) return { errors, values }
  const postcode = `${compact.slice(0, -3)} ${compact.slice(-3)}`
  return { answer: { ...values, postcode } }
}

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
const PHONE = /^(\+44|0)\d{9,10}$/

function validateContact(body) {
  const contact = clean(body.contact)
  const email = clean(body.contactEmail)
  const phone = clean(body.contactPhone)
  const values = { contact, email, phone }
  if (!CONTACT[contact]) return { errors: { contact: "Select how you would like us to contact you" }, values }
  if (contact === "email") {
    if (!email) return { errors: { "contact-email": "Enter an email address" }, values }
    if (!EMAIL.test(email)) {
      return { errors: { "contact-email": "Enter an email address in the correct format, like name@example.com" }, values }
    }
    return { answer: { contact, email } }
  }
  if (contact === "phone") {
    if (!phone) return { errors: { "contact-phone": "Enter a phone number" }, values }
    if (!PHONE.test(phone.replace(/[\s()-]/g, ""))) {
      return {
        errors: { "contact-phone": "Enter a phone number, like 01632 960 001, 07700 900 982 or +44 808 157 0192" },
        values
      }
    }
    return { answer: { contact, phone } }
  }
  return { answer: { contact } }
}

function validateCondition(body) {
  const condition = clean(body.condition)
  if (!condition) return { errors: { condition: "Describe how your condition affects your walking" }, values: { condition } }
  if (condition.length > CONDITION_MAX) {
    return { errors: { condition: `Description must be ${CONDITION_MAX} characters or less` }, values: { condition } }
  }
  return { answer: { condition } }
}

// ------------------------------------------------------------------ steps

const STEPS = [
  { slug: "who-for", title: "Who are you applying for?", validate: validateWhoFor },
  { slug: "benefits", title: "Do you get any of these benefits?", validate: validateBenefits },
  { slug: "walking", title: "How far can you walk without stopping?", validate: validateWalking },
  { slug: "name", title: "What is your full name?", validate: validateName },
  { slug: "date-of-birth", title: "What is your date of birth?", validate: validateDateOfBirth },
  { slug: "address", title: "What is your address?", validate: validateAddress },
  { slug: "contact", title: "How should we contact you?", validate: validateContact },
  { slug: "condition", title: "Describe how your condition affects your walking", validate: validateCondition }
]

const isChange = (req) => req.query.change === "true"

function backLinkFor(index, req) {
  if (isChange(req)) return "/check-answers"
  return index === 0 ? "/" : `/${STEPS[index - 1].slug}`
}

function renderStep(res, step, index, req, extra = {}) {
  const errors = extra.errors || {}
  res.render(`${step.slug}.njk`, {
    pageTitle: step.title,
    backLink: backLinkFor(index, req),
    errors,
    errorList: Object.entries(errors).map(([id, text]) => ({ text, href: `#${id}` })),
    invalid: extra.invalid || [],
    values: extra.values || req.session.answers[step.slug] || {},
    options: { WHO_FOR, BENEFITS, WALKING, CONTACT, CONDITION_MAX }
  })
}

STEPS.forEach((step, index) => {
  app.get(`/${step.slug}`, (req, res) => renderStep(res, step, index, req))

  app.post(`/${step.slug}`, (req, res) => {
    const result = step.validate(req.body)
    if (result.errors) {
      return renderStep(res, step, index, req, {
        errors: result.errors,
        invalid: result.invalid,
        values: result.values || {}
      })
    }
    req.session.answers[step.slug] = result.answer
    const next = isChange(req) || index === STEPS.length - 1 ? "/check-answers" : `/${STEPS[index + 1].slug}`
    res.redirect(next)
  })
})

// --------------------------------------------------------- check answers

const MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"]

function summaryRows(a) {
  const dob = a["date-of-birth"]
  const addr = a.address
  const contact = a.contact
  const contactText =
    contact.contact === "email" ? `Email: ${contact.email}`
      : contact.contact === "phone" ? `Phone: ${contact.phone}`
        : "Post"
  const row = (key, value, slug, hidden) => ({ key, value, href: `/${slug}?change=true`, hidden })
  return [
    row("Applying for", WHO_FOR[a["who-for"].whoFor], "who-for", "who you are applying for"),
    row("Benefits", a.benefits.benefits.map((b) => BENEFITS[b]), "benefits", "benefits"),
    row("Walking distance", WALKING[a.walking.walking], "walking", "how far you can walk"),
    row("Full name", a.name.fullName, "name", "full name"),
    row("Date of birth", `${dob.day} ${MONTH_NAMES[dob.month - 1]} ${dob.year}`, "date-of-birth", "date of birth"),
    row("Address", [addr.line1, addr.line2, addr.town, addr.county, addr.postcode].filter(Boolean), "address", "address"),
    row("Contact", contactText, "contact", "how we contact you"),
    row("How your condition affects your walking", a.condition.condition, "condition", "description of your condition")
  ]
}

app.get("/check-answers", (req, res) => {
  const missing = STEPS.find((s) => !req.session.answers[s.slug])
  if (missing) return res.redirect(`/${missing.slug}`)
  res.render("check-answers.njk", {
    pageTitle: "Check your answers before sending your application",
    backLink: `/${STEPS[STEPS.length - 1].slug}`,
    rows: summaryRows(req.session.answers)
  })
})

app.post("/check-answers", (req, res) => {
  const missing = STEPS.find((s) => !req.session.answers[s.slug])
  if (missing) return res.redirect(`/${missing.slug}`)
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
  const pick = () => chars[crypto.randomInt(chars.length)]
  const reference = `PB${Array.from({ length: 6 }, pick).join("")}`
  const contact = req.session.answers.contact
  // Submitted: clear the answers so the application cannot be changed or sent twice.
  req.session.answers = {}
  req.session.submitted = { reference, contact: contact.contact }
  res.redirect("/confirmation")
})

app.get("/confirmation", (req, res) => {
  if (!req.session.submitted) return res.redirect("/")
  res.render("confirmation.njk", {
    pageTitle: "Application complete",
    reference: req.session.submitted.reference,
    contact: req.session.submitted.contact
  })
})

// ------------------------------------------------------------ other pages

app.get("/", (req, res) => res.render("start.njk", { pageTitle: SERVICE_NAME }))
app.get("/accessibility", (req, res) => res.render("accessibility.njk", { pageTitle: "Accessibility statement" }))
app.get("/cookies", (req, res) => res.render("cookies.njk", { pageTitle: "Cookies" }))
app.get("/privacy", (req, res) => res.render("privacy.njk", { pageTitle: "Privacy notice" }))
app.get("/start-again", (req, res) => req.session.destroy(() => res.redirect("/")))

app.use((req, res) => res.status(404).render("not-found.njk", { pageTitle: "Page not found" }))

app.listen(PORT, () => {
  console.log(`${SERVICE_NAME} running at http://localhost:${PORT}`)
})
