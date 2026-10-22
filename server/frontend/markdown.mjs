function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function safeHref(value) {
  const href = String(value).trim();
  if (/^(?:https?:\/\/|\/|#|\.\.?\/)/i.test(href)) return escapeHtml(href);
  if (documentationLink(href)) {
    return escapeHtml(href);
  }
  return "#";
}

export function documentationLink(href, currentDocument = "", currentLocale = "en") {
  const value = String(href || "").trim();
  const hash = value.indexOf("#");
  const path = hash < 0 ? value : value.slice(0, hash);
  let anchor;
  try {
    anchor = hash < 0 ? "" : decodeURIComponent(value.slice(hash + 1));
  } catch {
    return null;
  }
  if (!path && hash === 0 && ["user-guide", "api", "maintainer"].includes(currentDocument)) {
    return { document: currentDocument, locale: currentDocument === "maintainer" ? "en" : currentLocale, anchor };
  }
  const local = /^(?:\.\/)?(user-guide|api)(?:\.(zh-CN|ja|de|es|fr|ru|pt|ko))?\.md$/.exec(path);
  if (local) return { document: local[1], locale: local[2] === "zh-CN" ? "zh" : local[2] || "en", anchor };
  if (/^(?:\.\/)?maintainer-guide\.md$/.test(path)) return { document: "maintainer", locale: "en", anchor };
  const bundled = /^\/guide-content\/(en|zh|ja|de|es|fr|ru|pt|ko)\/(user-guide|api|maintainer)\.md$/.exec(path);
  return bundled ? { document: bundled[2], locale: bundled[2] === "maintainer" ? "en" : bundled[1], anchor } : null;
}

function headingAnchor(value) {
  return String(value)
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]*>/g, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\p{M}\s_-]/gu, "")
    .replace(/\s/g, "-");
}

function safeImageHref(value) {
  const href = String(value).trim();
  if (href.startsWith("docs/images/")) return escapeHtml(`/guide-images/${href.slice("docs/images/".length)}`);
  return safeHref(href);
}

function inline(value) {
  let output = escapeHtml(value);
  output = output.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, label, href) => `<img src="${safeImageHref(href)}" alt="${label}" loading="lazy">`);
  output = output.replace(/`([^`]+)`/g, "<code>$1</code>");
  output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  output = output.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  output = output.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => `<a href="${safeHref(href)}" target="_blank" rel="noopener">${label}</a>`);
  return output;
}

function isTableSeparator(value) {
  return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(value);
}

function tableCells(value) {
  return value.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

export function renderMarkdown(markdown) {
  const lines = String(markdown || "").replaceAll("\r\n", "\n").split("\n");
  const output = [];
  const anchors = new Set();
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    if (line.startsWith("```")) {
      const language = line.slice(3).trim().replace(/[^A-Za-z0-9_-]/g, "");
      const code = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith("```")) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      output.push(`<pre><code${language ? ` class="language-${language}"` : ""}>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const base = headingAnchor(heading[2]);
      let anchor = base;
      let suffix = 0;
      while (anchors.has(anchor)) anchor = `${base}-${++suffix}`;
      anchors.add(anchor);
      output.push(`<h${level} id="${escapeHtml(anchor)}">${inline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (index + 1 < lines.length && line.includes("|") && isTableSeparator(lines[index + 1])) {
      const headers = tableCells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      output.push(`<div class="markdown-table-wrap"><table><thead><tr>${headers.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }

    const unordered = /^\s*[-*+]\s+(.+)$/.exec(line);
    const ordered = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (unordered || ordered) {
      const tag = unordered ? "ul" : "ol";
      const items = [];
      while (index < lines.length) {
        const match = tag === "ul" ? /^\s*[-*+]\s+(.+)$/.exec(lines[index]) : /^\s*\d+[.)]\s+(.+)$/.exec(lines[index]);
        if (!match) break;
        items.push(`<li>${inline(match[1])}</li>`);
        index += 1;
      }
      output.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    if (line.startsWith("> ")) {
      const quotes = [];
      while (index < lines.length && lines[index].startsWith("> ")) {
        quotes.push(lines[index].slice(2));
        index += 1;
      }
      output.push(`<blockquote>${inline(quotes.join(" "))}</blockquote>`);
      continue;
    }

    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !/^(?:#{1,6}\s|```|\s*[-*+]\s+|\s*\d+[.)]\s+|> )/.test(lines[index])) {
      if (index + 1 < lines.length && isTableSeparator(lines[index + 1])) break;
      paragraph.push(lines[index].trim());
      index += 1;
    }
    output.push(`<p>${inline(paragraph.join(" "))}</p>`);
  }

  return output.join("\n");
}
