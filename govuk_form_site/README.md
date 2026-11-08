# GOV.UK form demo site

A demo service, **"Apply for a disabled parking badge (demo)"**. It is built with the
official [GOV.UK Design System](https://design-system.service.gov.uk/) components (the
`govuk-frontend` 6.6.0 Nunjucks macros), so it behaves like a real GOV.UK form. We
use it to test the MyoPilot gestures, and later the LLM and voice agent.

It is **not** a GOV.UK service, so it follows
[Using GOV.UK Frontend without GOV.UK branding](https://frontend.design-system.service.gov.uk/using-govuk-frontend-without-govuk-branding/):
- It uses the Generic header and Service navigation. There is no crown and no GOV.UK logotype.
- It uses Helvetica Neue / Arial instead of GDS Transport.
- It has its own brand colour (`#4c2c92`), its own icons and its own footer.
- A "Demo" phase banner is shown on every page.

## Run it

You need Node 20 or later.

```bash
cd govuk_form_site
npm install
npm start
```

Then open http://localhost:3000. To use another port, start it with `PORT=4000 npm start`.

## Pages (one question per page)

| URL | Question | GOV.UK component / pattern |
|---|---|---|
| `/` | Start page | Start button, inset text, warning text, details |
| `/who-for` | Who are you applying for? | Radios |
| `/benefits` | Do you get any of these benefits? | Checkboxes with an exclusive "None of these" |
| `/walking` | How far can you walk without stopping? | Radios (4 options) |
| `/name` | What is your full name? | Text input (Names pattern) |
| `/date-of-birth` | What is your date of birth? | Date input (Dates pattern) |
| `/address` | What is your address? | Addresses pattern |
| `/contact` | How should we contact you? | Radios with conditional reveal |
| `/condition` | Describe how your condition affects your walking | Character count (500) |
| `/check-answers` | Check your answers, then Accept and send | Summary list with Change links |
| `/confirmation` | Application complete | Panel with a reference number |

The footer links to `/accessibility`, `/cookies` and `/privacy`. Going to `/start-again` clears your answers.

How the pages behave:
- **Errors:** if an answer is missing or wrong, the page shows an error summary at the top, the page title starts with "Error: ", and an error message appears next to the field. The message wording comes from the GOV.UK guidance for each component.
- **Saving:** every answer is saved, and then the browser is sent to the next page (POST, then GET). This means the browser's Back button always works.
- **Change links:** a Change link on the check answers page takes you to that question, and saving brings you back to check answers.
- **Storage:** answers are kept only in memory. Nothing is sent anywhere.

## Testing with the gestures

```bash
uv run gesture_control/test_ges.py --open http://localhost:3000
uv run gesture_control/test_ges.py --tab      # first tab: "Skip to main content"
uv run gesture_control/test_ges.py --click
```

The site is designed so that the five gestures can do everything:
- There are no dropdowns, file uploads or timeouts.
- Every link, button, radio and checkbox can be reached with `tab` and pressed with `click`.
- Text fields only get focus from a gesture. Typing comes later, from voice or the LLM.
