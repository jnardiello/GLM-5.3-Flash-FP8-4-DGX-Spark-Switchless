export function slugify(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-');   // bugs: keeps leading/trailing dashes, drops unicode letters like "ł"
}
