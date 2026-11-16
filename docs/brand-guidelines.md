# RE-call brand guidelines

The RE-call desktop UI, README graphics and GitHub social preview use the same visual language. This file is the versioned source of truth for that language.

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

The standard RE-call logo is the gold circuit brain with the integrated RE-call wordmark. Use the transparent PNG at `docs/assets/re_call_logo.png` as the source for all sizes. Keep the full silhouette and leave clear space around it. Use the `canvas` or `surface` color behind the mark in RE-call interfaces. For profile pictures, keep the logo centered on a square so the full silhouette fits a circular crop.

The README banner and GitHub social preview are both 1280 × 640. The logo is the visual anchor; copy stays short and readable at thumbnail size.

The banner uses the homepage palette of near black (`#100D0F`), warm rose, ivory text, and muted gold circuitry. Use [banner.gif](banner.gif) where animation is supported. Use [banner.png](banner.png), the same first card, for forums, reduced motion, and social uploads. [social_card.png](social_card.png) is an identical compatibility copy for existing links. These files are the only approved banner artwork.

## Type and composition

Use Geist for display and interface text, with Geist Mono for compact metadata and labels. Headings use tight tracking and sentence case unless the interface calls for an all-caps section label. Prefer a single gold rule or active state to multiple competing gradients. Layouts use generous outer margins, clear alignment, and high contrast before decoration.

## Accessibility and state

Primary text must remain high contrast against `canvas` and `surface`. Gold is a signal, not body copy. Green and red must always be paired with text or an icon, never used as the only indication of state. Focus uses `gold-bright` with a visible outline.

## Source files

Run `python docs/make_brand_assets.py` after changing the canonical logo. It updates the legacy site logo, desktop watermark, Claude plugin icon, README banner, and GitHub social preview.

Run `python docs/make_readme_banner.py` to rebuild the GIF, static PNG, and identical social preview after changing the banner copy or design.
