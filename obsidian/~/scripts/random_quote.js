async function random_quote(quotes_note) {
  const file = await app.vault.getAbstractFileByPath(quotes_note);
  const content = await app.vault.read(file);

  // Get all list items that start with "- "
  const lines = content.split("\n").filter(l => l.trim().startsWith("- "));
  if (lines.length === 0) return "_(No quotes found)_";

  // Pick a random one
  const random = lines[Math.floor(Math.random() * lines.length)]
    .replace(/^- /, "")
    .trim();

  // Try to split into quote and author parts
  const match = random.match(/^["“](.+?)["”]\s*[–-]\s*(.+)$/);
  if (match) {
    const quote = match[1].trim();
    const author = match[2].trim();
    return `> "${quote}" — **${author}**`;
  } else {
    // Fallback if format doesn't match
    return `> ${random}`;
  }
}

module.exports = random_quote;
