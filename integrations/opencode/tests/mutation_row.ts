/** One mutation row, as tests/mutations.ts replays it (its header says how), declared once (owner,
 * 2026-11-11, audit DEF-21). It lives here, not in tests/mutations.ts, because that file imports every
 * row file and an import back would be a cycle. `assertion` is required (DEF-50 backfill, 2026-11-12),
 * except on a live row: its line can be captured only by a live run, and until then it stays UNKNOWN. */
export type Row = {
  id: string;
  file: string;
  find: string;
  replace: string;
  test: string;
  name: string;
} & ({ live?: undefined; assertion: string } | { live: string; assertion?: string });
