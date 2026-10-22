import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { documentationLink, renderMarkdown } from "../../frontend/markdown.mjs";

function headingIds(rendered) {
  return [...rendered.matchAll(/<h[1-6] id="([^"]*)">/g)].map((match) => match[1]);
}

test("documentation links resolve local and bundled guides in every supported language", () => {
  for (const document of ["user-guide", "api"]) {
    for (const suffix of ["", ".zh-CN", ".ja", ".de", ".es", ".fr", ".ru", ".pt", ".ko"]) {
      for (const prefix of ["", "./"]) {
        assert.deepEqual(documentationLink(`${prefix}${document}${suffix}.md`), { document, locale: suffix === ".zh-CN" ? "zh" : suffix.slice(1) || "en", anchor: "" });
      }
    }
  }
  for (const document of ["user-guide", "api", "maintainer"]) {
    for (const language of ["en", "zh", "ja", "de", "es", "fr", "ru", "pt", "ko"]) {
      assert.deepEqual(documentationLink(`/guide-content/${language}/${document}.md`), { document, locale: document === "maintainer" ? "en" : language, anchor: "" });
    }
  }
  assert.deepEqual(documentationLink("maintainer-guide.md"), { document: "maintainer", locale: "en", anchor: "" });
  assert.deepEqual(documentationLink("./maintainer-guide.md"), { document: "maintainer", locale: "en", anchor: "" });
});

test("same-document links preserve English and Unicode anchors and decode encoded fragments", () => {
  const anchors = ["optional-liveness-addon", "可选活体检测-addon", "résultats-de-détection"];
  for (const document of ["user-guide", "api", "maintainer"]) {
    for (const anchor of anchors) {
      assert.deepEqual(documentationLink(`#${anchor}`, document), { document, locale: "en", anchor });
      assert.deepEqual(documentationLink(`#${encodeURIComponent(anchor)}`, document), { document, locale: "en", anchor });
    }
    assert.deepEqual(documentationLink("#", document), { document, locale: "en", anchor: "" });
  }
  assert.equal(documentationLink("#optional-liveness-addon"), null);
  assert.equal(documentationLink("#optional-liveness-addon", "unknown"), null);
  assert.equal(documentationLink("#%E6%", "api"), null);
});

test("cross-document links select the destination and retain the requested anchor", () => {
  assert.deepEqual(documentationLink(" api.md#optional-liveness-addon ", "user-guide"), {
    document: "api", locale: "en", anchor: "optional-liveness-addon",
  });
  assert.deepEqual(documentationLink("./user-guide.zh-CN.md#可选活体检测-addon", "api"), {
    document: "user-guide", locale: "zh", anchor: "可选活体检测-addon",
  });
  assert.deepEqual(documentationLink(`/guide-content/zh/api.md#${encodeURIComponent("可选活体检测-addon")}`, "maintainer"), {
    document: "api", locale: "zh", anchor: "可选活体检测-addon",
  });
  assert.deepEqual(documentationLink("maintainer-guide.md#release-checks", "api"), {
    document: "maintainer", locale: "en", anchor: "release-checks",
  });
  assert.equal(documentationLink("api.md#%E6%"), null);
});

test("an explicit document path selects its language while fragment links keep the current language", () => {
  for (const language of ["en", "zh", "ja", "de", "es", "fr", "ru", "pt", "ko"]) {
    assert.deepEqual(documentationLink("#活体", "user-guide", language), {
      document: "user-guide", locale: language, anchor: "活体",
    });
    assert.deepEqual(documentationLink("user-guide.md#web-download-permissions", "user-guide", language), {
      document: "user-guide", locale: "en", anchor: "web-download-permissions",
    });
    assert.deepEqual(documentationLink("api.ja.md#結果", "user-guide", language), {
      document: "api", locale: "ja", anchor: "結果",
    });
    assert.deepEqual(documentationLink("#release-checks", "maintainer", language), {
      document: "maintainer", locale: "en", anchor: "release-checks",
    });
  }
});

test("unrecognized paths and external links are not intercepted as bundled documentation", () => {
  for (const href of [
    "", null, undefined,
    "README.md", "other.md", "api.it.md", "../api.md", "api.md?download=1#liveness",
    "/guide-content/it/api.md", "/guide-content/en/unknown.md", "/v1/system",
    "https://example.com/api.md#liveness", "http://example.com/user-guide.md",
    "//example.com/api.md", "mailto:help@example.com", "javascript:alert(1)", "data:text/html,test",
  ]) {
    assert.equal(documentationLink(href, "api"), null, `unexpected documentation target: ${href}`);
  }
});

test("heading anchors retain Unicode words and strip Markdown formatting", () => {
  const markdown = [
    "# Optional liveness addon",
    "## 可选活体检测 addon",
    "### Résultats de détection",
    "#### **Liveness** `is_live` [status](api.md)",
    "##### BMP / JPEG & PNG (输入)",
  ].join("\n");
  assert.deepEqual(headingIds(renderMarkdown(markdown)), [
    "optional-liveness-addon", "可选活体检测-addon", "résultats-de-détection",
    "liveness-is_live-status", "bmp--jpeg--png-输入",
  ]);
});

test("duplicate heading IDs stay unique even when headings contain numbered suffixes", () => {
  const markdown = "# Liveness\n## Liveness\n## Liveness-1\n## Liveness\n## 活体\n## 活体";
  const expected = ["liveness", "liveness-1", "liveness-1-1", "liveness-2", "活体", "活体-1"];
  assert.deepEqual(headingIds(renderMarkdown(markdown)), expected);
  assert.deepEqual(headingIds(renderMarkdown(markdown)), expected, "anchor state must reset for each document");
});

test("rendered links keep documentation fragments while unsafe protocols and raw HTML stay inert", () => {
  const encoded = encodeURIComponent("可选活体检测-addon");
  const rendered = renderMarkdown([
    `# 活体 <script>alert("x")</script> & 结果`,
    "",
    "[Section](#optional-liveness-addon)",
    "",
    `[中文 API](api.zh-CN.md#${encoded})`,
    "",
    "[External](https://example.com/guide#liveness)",
    "",
    "[Unsupported](other.md)",
    "",
    "[Unsafe](javascript:alert(1))",
    "",
    "```html",
    '<img src="x" onerror="alert(1)">',
    "```",
  ].join("\n"));

  assert.ok(rendered.includes('href="#optional-liveness-addon"'));
  assert.ok(rendered.includes(`href="api.zh-CN.md#${encoded}"`));
  assert.ok(rendered.includes('href="https://example.com/guide#liveness" target="_blank" rel="noopener"'));
  assert.match(rendered, /href="#"[^>]*>Unsupported<\/a>/);
  assert.match(rendered, /href="#"[^>]*>Unsafe<\/a>/);
  assert.match(rendered, /&lt;script&gt;alert\(&quot;x&quot;\)&lt;\/script&gt; &amp; 结果/);
  assert.match(rendered, /&lt;img src=&quot;x&quot; onerror=&quot;alert\(1\)&quot;&gt;/);
  assert.doesNotMatch(rendered, /<(?:script|img)\b/);
  assert.doesNotMatch(rendered, /href="javascript:/);
});

test("English and Chinese liveness guide links resolve to headings rendered from the actual bundled documents", async () => {
  for (const document of ["user-guide", "api"]) {
    for (const [suffix, anchor] of [["", "optional-liveness-addon"], [".zh-CN", "可选活体检测-addon"]]) {
      const filename = `${document}${suffix}.md`;
      const markdown = await readFile(new URL(`../../docs/${filename}`, import.meta.url), "utf8");
      const target = documentationLink(`${filename}#${encodeURIComponent(anchor)}`);
      assert.equal(target.document, document);
      assert.ok(headingIds(renderMarkdown(markdown)).includes(target.anchor), `missing liveness destination in ${filename}`);
    }
  }
});

test("all embedded guide links resolve to headings in the document and language they request", async () => {
  const suffixes = { en: "", zh: ".zh-CN", ja: ".ja", de: ".de", es: ".es", fr: ".fr", ru: ".ru", pt: ".pt", ko: ".ko" };
  const documents = new Map();
  for (const [language, suffix] of Object.entries(suffixes)) {
    for (const document of ["user-guide", "api"]) {
      const markdown = await readFile(new URL(`../../docs/${document}${suffix}.md`, import.meta.url), "utf8");
      documents.set(`${document}:${language}`, renderMarkdown(markdown));
    }
  }
  documents.set("maintainer:en", renderMarkdown(await readFile(new URL("../../docs/maintainer-guide.md", import.meta.url), "utf8")));
  for (const [source, rendered] of documents) {
    const [document, language] = source.split(":");
    for (const link of rendered.matchAll(/<a href="([^"]*)"/g)) {
      const target = documentationLink(link[1], document, language);
      if (!target) continue;
      const destination = `${target.document}:${target.locale}`;
      assert.ok(documents.has(destination), `${source} links to unavailable ${destination}`);
      if (target.anchor) assert.ok(headingIds(documents.get(destination)).includes(target.anchor),
        `${source}: ${link[1]} has no heading ${target.anchor} in ${destination}`);
    }
  }
  for (const [language, suffix] of Object.entries(suffixes)) {
    const filename = `README${suffix}.md`;
    const markdown = await readFile(new URL(`../../${filename}`, import.meta.url), "utf8");
    for (const link of markdown.matchAll(/\]\((docs\/(?:user-guide|api|maintainer-guide)[^)]*\.md(?:#[^)]*)?)\)/g)) {
      const target = documentationLink(link[1].slice("docs/".length), "", language);
      assert.ok(target, `${filename} has an unsupported guide link: ${link[1]}`);
      const destination = `${target.document}:${target.locale}`;
      assert.ok(documents.has(destination), `${filename} links to unavailable ${destination}`);
      if (target.anchor) assert.ok(headingIds(documents.get(destination)).includes(target.anchor),
        `${filename}: ${link[1]} has no heading ${target.anchor} in ${destination}`);
    }
  }
});
