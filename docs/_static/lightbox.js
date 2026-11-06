// Figures and notebook images open over the page instead of as a separate page:
// a click (or Enter) opens the full image, a click anywhere, the close button or Escape closes it.
document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.createElement("div");
  overlay.className = "lightbox";
  overlay.hidden = true;
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  const closeButton = document.createElement("button");
  closeButton.className = "lightbox-close";
  closeButton.type = "button";
  closeButton.setAttribute("aria-label", document.documentElement.lang === "ru" ? "Закрыть" : "Close");
  closeButton.textContent = "×";
  const image = document.createElement("img");
  const caption = document.createElement("div");
  caption.className = "lightbox-caption";
  overlay.append(closeButton, image, caption);
  document.body.append(overlay);

  let opener = null;
  const close = () => {
    overlay.hidden = true;
    image.removeAttribute("src");
    document.body.classList.remove("lightbox-open");
    if (opener) opener.focus();
  };
  const open = (source, alt, text, from) => {
    opener = from;
    image.src = source;
    image.alt = alt || "";
    caption.textContent = text || "";
    overlay.hidden = false;
    document.body.classList.add("lightbox-open");
    closeButton.focus();
  };
  overlay.addEventListener("click", close);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !overlay.hidden) close();
  });

  document.querySelectorAll("article img").forEach((img) => {
    const link = img.closest("a");
    // images that link somewhere else (badges, logos) keep their link
    if (link && !link.classList.contains("image-reference")) return;
    if (img.closest(".lightbox")) return;
    const target = link || img;
    const source = link ? link.href : img.currentSrc || img.src;
    const figure = img.closest("figure");
    // the caption text without the paragraph-link sign
    const captionNode = figure && (figure.querySelector(".caption-text") || figure.querySelector("figcaption"));
    const text = captionNode ? captionNode.textContent.trim() : "";
    img.classList.add("zoomable");
    if (!link) img.tabIndex = 0;
    const handler = (event) => {
      if (event.type === "keydown" && event.key !== "Enter") return;
      event.preventDefault();
      open(source, img.alt, text, target);
    };
    target.addEventListener("click", handler);
    target.addEventListener("keydown", handler);
  });
});
