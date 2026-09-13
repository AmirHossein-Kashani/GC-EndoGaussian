# Reference frames covering the Figure 2 target area (cutting_tissues_twice)

Target area: the Figure 2 drag-handle region, a disc of radius 105 px centered at pixel (351, 222)
in the 640x512 frames of `data/endonerf/cutting/images/`. All numbers below are computed from the
raw data alone (images + tool masks in `data/endonerf/cutting/masks/`), independent of the model:
a pixel counts as observed in a frame when the tool mask marks it as tissue.

Summary over all 156 frames: the average frame shows 66.7 percent of the target area as tissue;
8 frames show more than 80 percent of it at once; and the model's never-observed test (Figure 2,
fourth panel, black) covers 23.4 percent of the revealed area, matching the band that no frame
below reaches.

## Frames with the widest direct view of the target area

The instrument is clear of the region early in the sequence; these frames each show more than
80 percent of the disc, including 31 to 43 percent of the seldom-seen band (the corridor the
instrument usually covers). Frame 11 is the single best reference.

| Frame | File | Disc visible | Seldom-seen band visible |
|---|---|---|---|
| 7  | data/endonerf/cutting/images/000007.png | 81.7% | 34.1% |
| 8  | data/endonerf/cutting/images/000008.png | 82.6% | 37.4% |
| 9  | data/endonerf/cutting/images/000009.png | 82.7% | 38.0% |
| 10 | data/endonerf/cutting/images/000010.png | 83.2% | 39.6% |
| 11 | data/endonerf/cutting/images/000011.png | 84.2% | 43.1% |
| 12 | data/endonerf/cutting/images/000012.png | 83.7% | 41.6% |
| 13 | data/endonerf/cutting/images/000013.png | 82.6% | 37.4% |
| 14 | data/endonerf/cutting/images/000014.png | 80.8% | 31.2% |

## Frames evidencing the flap-edge displacement (content behind the ridge)

During the cutting phase the flap edge is displaced the most relative to frame 0 (mean absolute
intensity change over the mutually visible part of the disc). These frames expose the band along
the ridge that frame 0 hides; they are the observation source for revealed content immediately
behind the flap edge.

| Frame | File | Mean change vs frame 0 (0-255) |
|---|---|---|
| 70 | data/endonerf/cutting/images/000070.png | 20.9 |
| 71 | data/endonerf/cutting/images/000071.png | 21.6 |
| 72 | data/endonerf/cutting/images/000072.png | 21.8 |
| 73 | data/endonerf/cutting/images/000073.png | 21.9 |
| 76 | data/endonerf/cutting/images/000076.png | 20.9 |
| 79 | data/endonerf/cutting/images/000079.png | 21.2 |

## Baseline

| Frame | File | Role |
|---|---|---|
| 0 | data/endonerf/cutting/images/000000.png | The edit frame of Figure 2 (before/after panels) |

## How to read this against Figure 2

Most of the target disc is plateau and instrument-corridor tissue that the frames above show
directly, many times over; the drag edit re-shows that observed content. The strip that none of
these frames reaches, along the ridge where the flap covers the same spot in every frame, is
exactly the black region of Figure 2's fourth panel. Evidence sheet with the per-pixel observation
map: regenerate via the inline script in the session or see the per-frame statistics computed by
the same method (tool-mask counting over the 105 px disc).
