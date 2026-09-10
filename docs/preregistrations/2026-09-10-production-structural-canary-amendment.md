# Amendment: production structural edge canary calibration

Recorded before the first canary calibration or query measurement.

The original canary upload built and validated a generation but did not make it live because the
three document fixture cannot produce the production certification floor. This amendment permits
the minimum calibration apparatus needed to exercise the serving path.

Only tenant `structural-edge-canary-20260910` may be affected. Run one draft calibration for the
pinned generation using the following fixed labelled query set, then publish that calibration only
to the canary tenant. The labels are apparatus inputs, not quality outcomes.

```json
[
  {"query":"Who owns the booking?","answerable":true},
  {"query":"What does Alice own?","answerable":true},
  {"query":"Which person owns the reservation?","answerable":true},
  {"query":"Who is responsible for the booking?","answerable":true},
  {"query":"What is in the attached itinerary?","answerable":true},
  {"query":"What details are in the itinerary?","answerable":true},
  {"query":"What does the attachment contain?","answerable":true},
  {"query":"Which file has the booking details?","answerable":true},
  {"query":"Who owns the garden appointment?","answerable":false},
  {"query":"Which person scheduled the garden appointment?","answerable":false},
  {"query":"What is the flight number?","answerable":false},
  {"query":"Where is the hotel reservation?","answerable":false},
  {"query":"Who paid the invoice?","answerable":false},
  {"query":"What time is the meeting?","answerable":false},
  {"query":"Which city was visited?","answerable":false},
  {"query":"What is the dinner menu?","answerable":false},
  {"query":"Who sent the package?","answerable":false},
  {"query":"What was the weather?","answerable":false},
  {"query":"Which train was booked?","answerable":false},
  {"query":"When is the conference?","answerable":false}
]
```

The original pass criteria remain unchanged except for cleanup. After source cleanup, verify zero
indexed sources and no active generation for the canary tenant. Calibration and audit history may
remain as immutable canary history and must be reported rather than deleted.
