// Layout classes shared by tabbed panels (run page, live runs), so every tab fills its
// panel the same way on `fit` screens and the page itself never scrolls.

/**
 * A region that scrolls inside its panel on `fit` screens. The 4px inset keeps focus rings
 * of rows inside it from being clipped by the overflow edge, and `relative` makes it the
 * containing block of rows' absolutely positioned sr-only text, which would otherwise escape
 * the clip and scroll the whole tab panel.
 */
export const PANEL_SCROLL =
  "relative rounded-xl outline-none focus-visible:ring-[3px] focus-visible:ring-brand/40 fit:-m-1 fit:min-h-0 fit:flex-1 fit:overflow-y-auto fit:p-1";

/** A glass panel that fills its grid cell on `fit` screens and lays its body out as a column. */
export const FILL_PANEL = "min-w-0 fit:flex fit:min-h-0 fit:flex-col";

/** Main panel beside an evidence column; on `fit` screens the row fills the height it is given. */
export const TAB_GRID =
  "grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)] fit:grid-rows-[minmax(0,1fr)]";
