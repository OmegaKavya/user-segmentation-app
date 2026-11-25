/**
 * CSV parsing and column preparation for uploaded datasets.
 *
 * The upload used to split every line on every comma, so a quoted number
 * such as "1,200" became two cells and every column after it shifted by one.
 * This follows RFC 4180: fields may be quoted, quoted fields may contain
 * commas, newlines and doubled quotes ("").
 */

export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const src = text.replace(/^﻿/, "");

  for (let i = 0; i < src.length; i += 1) {
    const ch = src[i];
    if (quoted) {
      if (ch === '"') {
        if (src[i + 1] === '"') { field += '"'; i += 1; }
        else quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"' && field === "") {
      quoted = true;
    } else if (ch === ",") {
      row.push(field); field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && src[i + 1] === "\n") i += 1;
      row.push(field); field = "";
      rows.push(row); row = [];
    } else {
      field += ch;
    }
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.some((cell) => cell.trim() !== ""));
}

/** A cell as a number, allowing thousands separators, or NaN. */
export function toNumber(cell) {
  if (cell === undefined) return NaN;
  const trimmed = String(cell).trim();
  if (trimmed === "") return NaN;
  return Number(trimmed.replace(/(\d),(?=\d{3}\b)/g, "$1"));
}

// Columns that are numeric but are labels, not measurements. Clustering on a
// row id or a postcode finds structure in how the file was numbered.
const IDENTIFIER = /(^|[_\s-])(id|uuid|index|zip|zipcode|postcode|pincode|phone)$/i;

/**
 * The numeric columns of a parsed CSV, each marked with whether it should be
 * included by default.
 */
export function numericColumns(rows) {
  const [header, ...body] = rows;
  const sample = body.slice(0, 40);
  return header
    .map((name, index) => ({ name: name.trim() || `column ${index + 1}`, index }))
    .filter(({ index }) => sample.length > 0
      && sample.every((r) => !Number.isNaN(toNumber(r[index]))))
    .map((col) => ({ ...col, include: !IDENTIFIER.test(col.name) }));
}

/** The rows complete in every chosen column, as numbers. */
export function matrix(rows, columns) {
  return rows.slice(1)
    .map((r) => columns.map((c) => toNumber(r[c.index])))
    .filter((values) => values.every((v) => Number.isFinite(v)));
}
