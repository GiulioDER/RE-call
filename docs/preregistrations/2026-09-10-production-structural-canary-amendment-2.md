# Amendment 2: production canary calibration sample floor

Recorded after production rejected the first calibration attempt for having only 8 answerable and
12 unanswerable labels. No canary query measurement or live generation resulted from that attempt.

The production calibration floor requires at least 20 labels in each class. The canary may run one
replacement draft calibration with the following fixed 40 labels, using a new idempotency key. The
calibration remains limited to tenant `structural-edge-canary-20260910` and its publication, if
certified, remains limited to that tenant.

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
  {"query":"Who has the booking?","answerable":true},
  {"query":"Name the owner of the booking.","answerable":true},
  {"query":"Who owns this reservation?","answerable":true},
  {"query":"What does the itinerary contain?","answerable":true},
  {"query":"Where are the booking details?","answerable":true},
  {"query":"Which document mentions the itinerary?","answerable":true},
  {"query":"Who is named as the booking owner?","answerable":true},
  {"query":"What is attached to the booking?","answerable":true},
  {"query":"Which note describes the reservation?","answerable":true},
  {"query":"Who is connected to the itinerary?","answerable":true},
  {"query":"What does the booking attachment say?","answerable":true},
  {"query":"Identify the booking owner.","answerable":true},
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
  {"query":"When is the conference?","answerable":false},
  {"query":"What is the passport number?","answerable":false},
  {"query":"Who cancelled the reservation?","answerable":false},
  {"query":"Which restaurant was selected?","answerable":false},
  {"query":"What is the appointment address?","answerable":false},
  {"query":"Who owns the car?","answerable":false},
  {"query":"What was the invoice total?","answerable":false},
  {"query":"Which airport was used?","answerable":false},
  {"query":"What date is the concert?","answerable":false}
]
```
