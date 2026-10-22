// Shared by new-person enrollment and adding samples to an existing person.
export function renderRejectionList(target, rejected, files, { element, t, livenessText }) {
  const items = rejected ?? [];
  target.replaceChildren();
  target.hidden = items.length === 0;
  if (!items.length) return;
  target.append(element("strong", { text: t(items.length === 1 ? "{count} image was rejected" : "{count} images were rejected", { count: items.length }) }));
  items.forEach((item, index) => {
    const row = element("div", { className: "rejection-item" });
    const file = files[Number(item.index)];
    const details = element("div", { className: "rejection-details" });
    details.append(element("strong", { text: item.reason || item.code || "rejected" }));
    if (item.liveness) {
      details.append(element("span", { className: "rejection-liveness", text: livenessText(item) }));
    }
    row.append(
      element("span", { text: item.filename || item.file_name || file?.name || t("Image {index}", { index: index + 1 }) }),
      details,
    );
    target.append(row);
  });
}
