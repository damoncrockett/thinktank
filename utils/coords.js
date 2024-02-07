export function normalizeCoordinates(pcaCoordinates) {
    // Find min and max for both axes
    const xValues = pcaCoordinates.map(coord => coord[0]);
    const yValues = pcaCoordinates.map(coord => coord[1]);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // Normalize coordinates to [0, 1]
    return pcaCoordinates.map(([x, y]) => [
        (x - xMin) / (xMax - xMin),
        (y - yMin) / (yMax - yMin)
    ]);
}

export function toScreenCoordinates(normalizedCoordinates, screenWidth, screenHeight, padding = 40) { // Increased padding value
    // Define a drawable area reduced by padding
    const drawableWidth = screenWidth - 2 * padding;
    const drawableHeight = screenHeight - 2 * padding;

    // Apply further scaling to ensure no element is right on the edge
    const scaleMargin = padding; // Further reduce drawable area on each side

    return normalizedCoordinates.map(([x, y]) => {
        // Scale x and y to the drawable area with margins considered
        const left = (x * (drawableWidth - 2 * scaleMargin)) + padding + scaleMargin;
        const top = (y * (drawableHeight - 2 * scaleMargin)) + padding + scaleMargin;

        return { left, top };
    });
}