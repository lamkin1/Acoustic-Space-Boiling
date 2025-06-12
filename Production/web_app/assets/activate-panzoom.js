document.addEventListener("DOMContentLoaded", function () {
    const obj = document.querySelector("object[type='image/svg+xml']");
    if (!obj) return;

    obj.addEventListener("load", function () {
        const svg = obj.contentDocument.querySelector("svg");
        if (!svg) return;

        // Initialize pan/zoom
        svgPanZoom(svg, {
            zoomEnabled: true,
            controlIconsEnabled: true,
            fit: true,
            center: true,
            minZoom: 0.5,
            maxZoom: 10,
        });
    });
});
