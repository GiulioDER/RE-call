# RE-call brand guidelines

The RE-call desktop UI, README graphics, GitHub Pages, and social preview use the same visual language. This file is the versioned source of truth for that language.

## Palette

The palette is intentionally restrained. The warm gold is the product signal. Green and red are reserved for runtime and validation state. The background is a warm near black rather than pure black so the logo and controls retain depth.

| Token | Hex | Use |
| --- | --- | --- |
| `canvas` | `#0E100F` | Application and page background |
| `surface` | `#141714` | Panels, cards, table heads |
| `surface-raised` | `#171A17` | Raised controls and secondary panels |
| `ink` | `#F4F1E8` | Primary text and headings |
| `ink-muted` | `#B6B7AC` | Supporting text and metadata |
| `gold` | `#D7A52A` | Brand signal, active state, primary action |
| `gold-bright` | `#F0BE4A` | Hover, focus, and high emphasis |
| `gold-soft` | `rgba(215, 165, 42, 0.18)` | Quiet active surfaces |
| `green` | `#63D39E` | Connected, certified, or successful state |
| `red` | `#EF6262` | Error, disconnect, or destructive state |
| `line` | `#465047` | Borders and table rules |

## Logo

Use `docs/assets/re_call_logo.png` as the canonical transparent logo. The same asset is mirrored at `site/assets/re_call_logo.png` for GitHub Pages. Keep the logo on the `canvas` or `surface` colors, preserve its aspect ratio, and leave clear space around it. Do not recreate the wordmark with a different font or place it on a white rectangle.

The README banner and GitHub social preview are both 1280 × 640. The logo is the visual anchor; copy stays short and readable at thumbnail size.

## Type and composition

Use Geist for display and interface text, with Geist Mono for compact metadata and labels. Headings use tight tracking and sentence case unless the interface calls for an all-caps section label. Prefer a single gold rule or active state to multiple competing gradients. Layouts use generous outer margins, clear alignment, and high contrast before decoration.

## Accessibility and state

Primary text must remain high contrast against `canvas` and `surface`. Gold is a signal, not body copy. Green and red must always be paired with text or an icon, never used as the only indication of state. Focus uses `gold-bright` with a visible outline.

## Source files

The generated graphics are written by `docs/make_brand_assets.py`. Run it after changing the logo or palette to rebuild `docs/banner.png` and `docs/social_card.png`.
