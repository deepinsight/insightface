import assert from "node:assert/strict";
import test from "node:test";
import { locale, setLocale, t, translateTree } from "../../frontend/i18n.mjs";

test("changing UI language preserves authored documentation, its TOC, and nested code text", () => {
  // Minimal DOM traversal to exercise the actual translation boundary, including
  // descendants rather than only a direct parent with translate="no".
  class FakeElement {
    nodeType = 1;
    children = [];
    attributes = new Map();
    constructor(tagName, parent = null) {
      this.tagName = tagName.toUpperCase();
      this.parentElement = parent;
      if (parent) parent.children.push(this);
    }
    hasAttribute(name) { return this.attributes.has(name); }
    getAttribute(name) { return this.attributes.get(name); }
    setAttribute(name, value) { this.attributes.set(name, value); }
    closest() {
      for (let node = this; node; node = node.parentElement) {
        if (node.getAttribute("translate") === "no" || ["SCRIPT", "STYLE", "CODE", "PRE", "TEXTAREA"].includes(node.tagName)) return node;
      }
      return null;
    }
  }
  function text(parent, value) {
    const node = { nodeType: 3, parentElement: parent, nodeValue: value };
    parent.children.push(node);
    return node;
  }
  class FakeDocument {
    nodeType = 9;
    documentElement = new FakeElement("html");
    querySelector() { return null; }
    createTreeWalker(root) {
      const nodes = [];
      function visit(node) {
        for (const child of node.children ?? [node.documentElement].filter(Boolean)) {
          nodes.push(child);
          visit(child);
        }
      }
      visit(root);
      return { nextNode: () => nodes.shift() ?? null };
    }
  }
  const doc = new FakeDocument();
  const button = new FakeElement("button", doc.documentElement);
  button.setAttribute("title", "Model");
  const uiText = text(button, "Model");
  const chapter = new FakeElement("section", doc.documentElement);
  chapter.setAttribute("translate", "no");
  const heading = new FakeElement("h2", chapter);
  heading.setAttribute("title", "Model");
  const authoredText = text(new FakeElement("strong", heading), "Model");
  const toc = new FakeElement("nav", doc.documentElement);
  toc.setAttribute("translate", "no");
  const tocText = text(new FakeElement("button", toc), "Liveness");
  const codeText = text(new FakeElement("span", new FakeElement("code", doc.documentElement)), "Model");
  const events = [];
  const globals = {
    Element: FakeElement, Document: FakeDocument, Node: { TEXT_NODE: 3 }, NodeFilter: { SHOW_ELEMENT: 1, SHOW_TEXT: 4 },
    document: doc,
    window: { localStorage: { setItem() { throw new Error("Storage unavailable"); } }, dispatchEvent(event) { events.push(event); } },
  };
  const original = Object.fromEntries(Object.keys(globals).map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  try {
    for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
    for (const language of ["zh", "pt", "ja"]) {
      setLocale(language);
      assert.equal(locale(), language);
      assert.equal(uiText.nodeValue, t("Model", {}, language));
      assert.equal(button.getAttribute("title"), t("Model", {}, language));
      assert.equal(authoredText.nodeValue, "Model");
      assert.equal(heading.getAttribute("title"), "Model");
      assert.equal(tocText.nodeValue, "Liveness");
      assert.equal(codeText.nodeValue, "Model");
      translateTree(authoredText);
      assert.equal(authoredText.nodeValue, "Model");
      assert.equal(events.at(-1).detail.locale, language);
    }
  } finally {
    setLocale("en", { persist: false, announce: false });
    for (const [key, descriptor] of Object.entries(original)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});
