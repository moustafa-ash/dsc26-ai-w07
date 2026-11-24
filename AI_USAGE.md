# AI usage

## Tools I used, and for what

Codex drafted the PII guards, PSI functions and FastAPI service. I used Python, pytest and FastAPI's test client to check original-text offsets, the prediction paths and the CSV drift results.

## The PII cases my AI-written regex missed at first, and how I found them

I tested an ASCII-only, compact phone regex against a synthetic mixed Arabic/Persian-digit number with a dash and against a number with spaces and dashes inside the country prefix. It missed both. The final guard translates digits without changing offsets and handles allowed separators. I also verified that the validation response for an oversized note does not echo the rejected value.

## One sentence on what my service does when the model is down, written for a pharmacist

If the model is down, you still get a shortage risk estimate based on stock cover, supplier fill rate and lead time, marked as a fallback.
