# Amendment 3: separated canary calibration labels

Recorded before the next calibration attempt. The previous 20 and 20 labelled calibration was
rejected because its separability confidence interval did not establish the production bar. The
production gate remains unchanged.

Run one final replacement draft calibration for generation
`gen_ed341f43ef39462f9664d706443f0d53`, using 20 answerable labels focused on the canary fixture
and 20 unrelated unanswerable labels below. Use idempotency key
`structural-edge-canary-calibration-v3-20260910`. If certification still fails, stop the canary
without changing trust settings or publishing an uncertified artifact.

```json
[
  {"query":"Alice owns the booking.","answerable":true},
  {"query":"Who owns the booking?","answerable":true},
  {"query":"What booking does Alice own?","answerable":true},
  {"query":"Name the person who owns the booking.","answerable":true},
  {"query":"Identify Alice's booking responsibility.","answerable":true},
  {"query":"Which person is the owner of the booking?","answerable":true},
  {"query":"Who is responsible for this booking?","answerable":true},
  {"query":"What reservation belongs to Alice?","answerable":true},
  {"query":"Who has ownership of the booking?","answerable":true},
  {"query":"Which person owns the reservation?","answerable":true},
  {"query":"What does the attached itinerary contain?","answerable":true},
  {"query":"Where are the booking details?","answerable":true},
  {"query":"What details are in the itinerary?","answerable":true},
  {"query":"What is contained in the booking attachment?","answerable":true},
  {"query":"Which attachment contains itinerary details?","answerable":true},
  {"query":"What does the itinerary say about the booking?","answerable":true},
  {"query":"Which file contains the booking information?","answerable":true},
  {"query":"Where can the booking itinerary be found?","answerable":true},
  {"query":"What information is in the attached booking document?","answerable":true},
  {"query":"Which document has the reservation details?","answerable":true},
  {"query":"What is the chemical formula for water?","answerable":false},
  {"query":"Who composed the ninth symphony?","answerable":false},
  {"query":"What is the tallest mountain?","answerable":false},
  {"query":"How does photosynthesis work?","answerable":false},
  {"query":"What is the capital of Japan?","answerable":false},
  {"query":"Who painted the Mona Lisa?","answerable":false},
  {"query":"What causes a solar eclipse?","answerable":false},
  {"query":"How many planets are in the solar system?","answerable":false},
  {"query":"What is the boiling point of water?","answerable":false},
  {"query":"Who wrote Pride and Prejudice?","answerable":false},
  {"query":"What is the speed of light?","answerable":false},
  {"query":"Which element has atomic number six?","answerable":false},
  {"query":"What is the largest ocean?","answerable":false},
  {"query":"Who discovered penicillin?","answerable":false},
  {"query":"What is the square root of sixteen?","answerable":false},
  {"query":"How far is the Moon from Earth?","answerable":false},
  {"query":"What is the currency of Brazil?","answerable":false},
  {"query":"Who was the first person on the Moon?","answerable":false},
  {"query":"What is the structure of an atom?","answerable":false},
  {"query":"Which continent contains Egypt?","answerable":false}
]
```
