import cv2
import mediapipe as mp
import numpy as np
import math
import os
import random
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

CAM_SOURCE = 0

CAM_WIDTH = 960
CAM_HEIGHT = 540

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

# ------------------------------------------------------------
# CAPTURE
# ------------------------------------------------------------

MIN_BOX_SIZE = 120

PINCH_HOLD_TIME = 1.0
COUNTDOWN_TIME = 3.0

# ------------------------------------------------------------
# HAND
# ------------------------------------------------------------

SMOOTHING = 0.35

# Hysteresis:
# Smaller = easier to pinch
# Larger = easier to release
PINCH_CLOSE = 0.060
PINCH_OPEN = 0.085

# ------------------------------------------------------------
# PUZZLE
# ------------------------------------------------------------

ROWS = 3
COLS = 3

SHUFFLE_MOVES = 80

# Size of dragged picture inside the target box
DRAG_TILE_SCALE = 0.92


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = python.BaseOptions

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAM_SOURCE,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAM_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAM_HEIGHT
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not cap.isOpened():
    print("ERROR: Camera cannot be opened.")
    exit()


# ============================================================
# VARIABLES
# ============================================================

timestamp_ms = 0

# ------------------------------------------------------------
# CAPTURE RECTANGLE
# ------------------------------------------------------------

point_a = None
point_b = None

smooth_a = None
smooth_b = None

# ------------------------------------------------------------
# COUNTDOWN
# ------------------------------------------------------------

pinch_start = None
countdown_active = False
countdown_start = 0

# ------------------------------------------------------------
# CAPTURED IMAGE
# ------------------------------------------------------------

captured_image = None
captured_tiles = None

# ------------------------------------------------------------
# PUZZLE
# ------------------------------------------------------------

puzzle_active = False

# Each position contains a tile number.
#
# 0 1 2
# 3 4 5
# 6 7 8
#
# Tile 8 = empty
#
puzzle_state = list(range(9))

solved_state = list(range(9))

puzzle_rect = None

# ------------------------------------------------------------
# HAND
# ------------------------------------------------------------

hand_point = None
hand_pinch = False

# Stable pinch state
pinch_state = False

# ------------------------------------------------------------
# DRAGGING
# ------------------------------------------------------------

dragging_tile = None

drag_source_position = None

drag_current_point = None

dragging_hand_index = None

# ------------------------------------------------------------
# SOLVED
# ------------------------------------------------------------

solved_message = False
solved_message_time = 0


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def smooth_point(old, new):

    if old is None:
        return new

    return (
        int(
            old[0] * (1.0 - SMOOTHING)
            + new[0] * SMOOTHING
        ),
        int(
            old[1] * (1.0 - SMOOTHING)
            + new[1] * SMOOTHING
        )
    )


def landmark_point(hand, number, width, height):

    x = int(hand[number].x * width)
    y = int(hand[number].y * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


def is_index_up(hand):

    return (
        hand[8].y < hand[6].y
        and
        hand[8].y < hand[5].y
    )


# ============================================================
# PINCH DETECTION
# ============================================================

def pinch_distance(hand):

    thumb = hand[4]
    index = hand[8]

    return math.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2
    )


def update_pinch_state(hand):

    global pinch_state

    d = pinch_distance(hand)

    # --------------------------------------------------------
    # HYSTERESIS
    # --------------------------------------------------------

    if not pinch_state:

        if d <= PINCH_CLOSE:
            pinch_state = True

    else:

        if d >= PINCH_OPEN:
            pinch_state = False

    return pinch_state


# ============================================================
# RECTANGLE
# ============================================================

def get_rectangle(p1, p2, width, height):

    x1 = min(p1[0], p2[0])
    y1 = min(p1[1], p2[1])

    x2 = max(p1[0], p2[0])
    y2 = max(p1[1], p2[1])

    x1 = max(
        0,
        min(width - 1, x1)
    )

    y1 = max(
        0,
        min(height - 1, y1)
    )

    x2 = max(
        0,
        min(width - 1, x2)
    )

    y2 = max(
        0,
        min(height - 1, y2)
    )

    return x1, y1, x2, y2


# ============================================================
# PUZZLE SHUFFLE
# ============================================================

def get_neighbors(position):

    neighbors = []

    row = position // 3
    col = position % 3

    if row > 0:
        neighbors.append(position - 3)

    if row < 2:
        neighbors.append(position + 3)

    if col > 0:
        neighbors.append(position - 1)

    if col < 2:
        neighbors.append(position + 1)

    return neighbors


def shuffle_puzzle():

    global puzzle_state

    puzzle_state = list(range(9))

    blank = 8

    previous = -1

    for _ in range(SHUFFLE_MOVES):

        choices = get_neighbors(blank)

        if (
            previous in choices
            and
            len(choices) > 1
        ):
            choices.remove(previous)

        selected = random.choice(choices)

        puzzle_state[blank], puzzle_state[selected] = (
            puzzle_state[selected],
            puzzle_state[blank]
        )

        previous = blank
        blank = selected

    if puzzle_state == solved_state:

        shuffle_puzzle()


# ============================================================
# PREPARE TILES
# ============================================================

def prepare_tiles(image):

    h, w = image.shape[:2]

    w = (w // 3) * 3
    h = (h // 3) * 3

    if w <= 0 or h <= 0:
        return []

    image = image[:h, :w]

    tile_w = w // 3
    tile_h = h // 3

    tiles = []

    for row in range(3):

        for col in range(3):

            x1 = col * tile_w
            y1 = row * tile_h

            x2 = x1 + tile_w
            y2 = y1 + tile_h

            tile = image[
                y1:y2,
                x1:x2
            ].copy()

            tiles.append(tile)

    return tiles


# ============================================================
# PUZZLE TILE RECTANGLE
# ============================================================

def get_cell_rect(position, rect):

    if rect is None:
        return None

    x1, y1, x2, y2 = rect

    puzzle_w = x2 - x1
    puzzle_h = y2 - y1

    tile_w = puzzle_w // 3
    tile_h = puzzle_h // 3

    row = position // 3
    col = position % 3

    px1 = x1 + col * tile_w
    py1 = y1 + row * tile_h

    if col == 2:
        px2 = x2
    else:
        px2 = px1 + tile_w

    if row == 2:
        py2 = y2
    else:
        py2 = py1 + tile_h

    return px1, py1, px2, py2


# ============================================================
# GET TILE POSITION
# ============================================================

def get_tile_position(point, rect):

    if point is None or rect is None:
        return None

    x1, y1, x2, y2 = rect

    px, py = point

    if px < x1 or px >= x2:
        return None

    if py < y1 or py >= y2:
        return None

    puzzle_w = x2 - x1
    puzzle_h = y2 - y1

    tile_w = puzzle_w / 3.0
    tile_h = puzzle_h / 3.0

    col = int(
        (px - x1) / tile_w
    )

    row = int(
        (py - y1) / tile_h
    )

    col = max(
        0,
        min(2, col)
    )

    row = max(
        0,
        min(2, row)
    )

    return row * 3 + col


# ============================================================
# CAPTURE IMAGE
# ============================================================

def capture_image(frame):

    global captured_image
    global captured_tiles
    global puzzle_active
    global puzzle_rect
    global solved_message

    if point_a is None or point_b is None:
        return False

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = get_rectangle(
        point_a,
        point_b,
        w,
        h
    )

    if (
        x2 - x1 < MIN_BOX_SIZE
        or
        y2 - y1 < MIN_BOX_SIZE
    ):
        return False

    crop = frame[
        y1:y2,
        x1:x2
    ].copy()

    if crop.size == 0:
        return False

    captured_image = crop

    captured_tiles = prepare_tiles(
        captured_image
    )

    if len(captured_tiles) != 9:
        return False

    puzzle_rect = (
        x1,
        y1,
        x2,
        y2
    )

    shuffle_puzzle()

    puzzle_active = True

    solved_message = False

    return True


# ============================================================
# DRAW CAPTURE RECTANGLE
# ============================================================

def draw_capture_rectangle(frame):

    if (
        smooth_a is None
        or
        smooth_b is None
    ):
        return

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = get_rectangle(
        smooth_a,
        smooth_b,
        w,
        h
    )

    if (
        x2 - x1 < MIN_BOX_SIZE
        or
        y2 - y1 < MIN_BOX_SIZE
    ):
        return

    # Main border
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    # Corner design
    corner = 25

    color = (0, 255, 255)

    # Top-left
    cv2.line(
        frame,
        (x1, y1),
        (x1 + corner, y1),
        color,
        5
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + corner),
        color,
        5
    )

    # Top-right
    cv2.line(
        frame,
        (x2, y1),
        (x2 - corner, y1),
        color,
        5
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + corner),
        color,
        5
    )

    # Bottom-left
    cv2.line(
        frame,
        (x1, y2),
        (x1 + corner, y2),
        color,
        5
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - corner),
        color,
        5
    )

    # Bottom-right
    cv2.line(
        frame,
        (x2, y2),
        (x2 - corner, y2),
        color,
        5
    )

    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - corner),
        color,
        5
    )

    # Finger points
    cv2.circle(
        frame,
        smooth_a,
        9,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        frame,
        smooth_b,
        9,
        (0, 255, 255),
        -1
    )


# ============================================================
# DRAW PUZZLE
# ============================================================

def draw_puzzle(frame):

    if (
        puzzle_rect is None
        or
        captured_tiles is None
    ):
        return frame

    x1, y1, x2, y2 = puzzle_rect

    puzzle_w = x2 - x1
    puzzle_h = y2 - y1

    if (
        puzzle_w <= 0
        or
        puzzle_h <= 0
    ):
        return frame

    # --------------------------------------------------------
    # DRAW TILES DIRECTLY
    #
    # NO DARK BACKGROUND
    # NO OPAQUE OVERLAY
    # LIVE CAMERA REMAINS VISIBLE
    # --------------------------------------------------------

    for position in range(9):

        # Don't draw the dragged tile in its original box.
        if (
            dragging_tile is not None
            and
            position == drag_source_position
        ):
            continue

        tile_number = puzzle_state[position]

        # Empty slot
        if tile_number == 8:
            continue

        cell = get_cell_rect(
            position,
            puzzle_rect
        )

        if cell is None:
            continue

        px1, py1, px2, py2 = cell

        cell_w = px2 - px1
        cell_h = py2 - py1

        tile = captured_tiles[tile_number]

        resized = cv2.resize(
            tile,
            (
                cell_w,
                cell_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

        frame[
            py1:py2,
            px1:px2
        ] = resized

    # --------------------------------------------------------
    # EMPTY CELL
    #
    # NO FILLED RECTANGLE.
    # ONLY BORDER.
    # --------------------------------------------------------

    blank = puzzle_state.index(8)

    blank_rect = get_cell_rect(
        blank,
        puzzle_rect
    )

    if blank_rect is not None:

        bx1, by1, bx2, by2 = blank_rect

        cv2.rectangle(
            frame,
            (bx1 + 3, by1 + 3),
            (bx2 - 3, by2 - 3),
            (180, 180, 180),
            2
        )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    for i in range(4):

        gx = int(
            x1 +
            i * (x2 - x1) / 3
        )

        cv2.line(
            frame,
            (gx, y1),
            (gx, y2),
            (255, 255, 255),
            2
        )

    for i in range(4):

        gy = int(
            y1 +
            i * (y2 - y1) / 3
        )

        cv2.line(
            frame,
            (x1, gy),
            (x2, gy),
            (255, 255, 255),
            2
        )

    # Outer border
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        3
    )

    return frame


# ============================================================
# DRAW DRAGGED TILE
# ============================================================

def draw_dragged_tile(frame):

    if (
        dragging_tile is None
        or
        drag_current_point is None
        or
        captured_tiles is None
    ):
        return

    if dragging_tile < 0 or dragging_tile >= 9:
        return

    tile = captured_tiles[
        dragging_tile
    ]

    if tile is None:
        return

    # --------------------------------------------------------
    # Find current box size
    # --------------------------------------------------------

    if puzzle_rect is None:
        return

    x1, y1, x2, y2 = puzzle_rect

    cell_w = (x2 - x1) // 3
    cell_h = (y2 - y1) // 3

    draw_w = int(
        cell_w * DRAG_TILE_SCALE
    )

    draw_h = int(
        cell_h * DRAG_TILE_SCALE
    )

    if draw_w <= 0 or draw_h <= 0:
        return

    resized = cv2.resize(
        tile,
        (
            draw_w,
            draw_h
        ),
        interpolation=cv2.INTER_LINEAR
    )

    cx, cy = drag_current_point

    px1 = cx - draw_w // 2
    py1 = cy - draw_h // 2

    px2 = px1 + draw_w
    py2 = py1 + draw_h

    # --------------------------------------------------------
    # Clip tile to screen
    # --------------------------------------------------------

    src_x1 = 0
    src_y1 = 0
    src_x2 = draw_w
    src_y2 = draw_h

    if px1 < 0:

        src_x1 = -px1
        px1 = 0

    if py1 < 0:

        src_y1 = -py1
        py1 = 0

    if px2 > frame.shape[1]:

        src_x2 -= (
            px2 - frame.shape[1]
        )

        px2 = frame.shape[1]

    if py2 > frame.shape[0]:

        src_y2 -= (
            py2 - frame.shape[0]
        )

        py2 = frame.shape[0]

    if (
        px1 >= px2
        or
        py1 >= py2
        or
        src_x1 >= src_x2
        or
        src_y1 >= src_y2
    ):
        return

    frame[
        py1:py2,
        px1:px2
    ] = resized[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    # --------------------------------------------------------
    # Dragged tile border
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (px1, py1),
        (px2 - 1, py2 - 1),
        (0, 255, 255),
        3
    )


# ============================================================
# DRAW HAND
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


def draw_hand(frame, hand):

    h, w = frame.shape[:2]

    points = []

    for landmark in hand:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        points.append(
            (x, y)
        )

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    for a, b in HAND_CONNECTIONS:

        if (
            a >= len(points)
            or
            b >= len(points)
        ):
            continue

        cv2.line(
            frame,
            points[a],
            points[b],
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Landmarks
    # --------------------------------------------------------

    for i, p in enumerate(points):

        radius = 4

        if i == 8:
            radius = 9

        if i == 4:
            radius = 7

        cv2.circle(
            frame,
            p,
            radius,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Pinch visual
    # --------------------------------------------------------

    thumb = points[4]
    index = points[8]

    if pinch_state:

        cv2.line(
            frame,
            thumb,
            index,
            (0, 255, 255),
            3,
            cv2.LINE_AA
        )

        midpoint = (
            (thumb[0] + index[0]) // 2,
            (thumb[1] + index[1]) // 2
        )

        cv2.circle(
            frame,
            midpoint,
            12,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


# ============================================================
# DRAW CURSOR
# ============================================================

def draw_cursor(frame):

    if hand_point is None:
        return

    if dragging_tile is not None:

        cv2.circle(
            frame,
            hand_point,
            20,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            hand_point,
            7,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

    elif hand_pinch:

        cv2.circle(
            frame,
            hand_point,
            16,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        cv2.circle(
            frame,
            hand_point,
            7,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# DROP TILE
# ============================================================

def drop_tile(destination_position):

    global puzzle_state
    global dragging_tile
    global drag_source_position
    global drag_current_point

    if dragging_tile is None:
        return

    source = drag_source_position

    if source is None:
        dragging_tile = None
        return

    # --------------------------------------------------------
    # If dropped outside puzzle:
    # return to original position
    # --------------------------------------------------------

    if destination_position is None:

        dragging_tile = None
        drag_source_position = None
        drag_current_point = None

        return

    # --------------------------------------------------------
    # Dropped on same box
    # --------------------------------------------------------

    if destination_position == source:

        dragging_tile = None
        drag_source_position = None
        drag_current_point = None

        return

    # --------------------------------------------------------
    # MOVE TO ANY BOX
    #
    # This is intentional:
    #
    # PINCH tile
    # MOVE tile
    # UNPINCH over another box
    #
    # The tile stays there.
    # --------------------------------------------------------

    destination_tile = puzzle_state[
        destination_position
    ]

    puzzle_state[
        destination_position
    ] = dragging_tile

    puzzle_state[
        source
    ] = destination_tile

    dragging_tile = None
    drag_source_position = None
    drag_current_point = None


# ============================================================
# SOLVED CHECK
# ============================================================

def check_solved():

    return (
        puzzle_state ==
        solved_state
    )


# ============================================================
# RESET PUZZLE
# ============================================================

def reset_puzzle():

    global dragging_tile
    global drag_source_position
    global drag_current_point
    global pinch_state
    global solved_message

    shuffle_puzzle()

    dragging_tile = None
    drag_source_position = None
    drag_current_point = None

    pinch_state = False

    solved_message = False


# ============================================================
# HEADER
# ============================================================

def draw_puzzle_header(frame):

    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        "PUZZLE CAM",
        (15, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "PINCH = GRAB    UNPINCH = DROP",
        (245, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA
    )

    if dragging_tile is not None:

        cv2.putText(
            frame,
            "GRABBING",
            (w - 145, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


# ============================================================
# SOLVED DISPLAY
# ============================================================

def draw_solved(frame):

    h, w = frame.shape[:2]

    # Don't cover the whole camera.
    # Just display text.

    text = "PUZZLE SOLVED!"

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 1.3
    thickness = 4

    size = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )[0]

    tx = (
        w - size[0]
    ) // 2

    ty = h - 45

    cv2.putText(
        frame,
        text,
        (tx, ty),
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Press N for a new photo",
        (
            tx - 5,
            ty + 30
        ),
        font,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print("           HAND PUZZLE CAMERA")
print("==============================================")
print()
print("CAPTURE:")
print("1. Raise both index fingers.")
print("2. Position the two fingers to create a box.")
print("3. Pinch both index fingers to both thumbs.")
print("4. Hold for 1 second.")
print("5. 3 -> 2 -> 1 -> CAPTURE")
print()
print("PUZZLE:")
print("1. Point your index finger at a picture.")
print("2. Pinch index + thumb.")
print("3. Keep pinching.")
print("4. Move the picture to another box.")
print("5. UNPINCH to drop it.")
print("6. The picture stays in that box.")
print()
print("CONTROLS:")
print("S = Shuffle")
print("R = Restart")
print("N = New Capture")
print("C = Manual Capture")
print("Q / ESC = Quit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    h, w = frame.shape[:2]

    timestamp_ms += 33

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    hands = result.hand_landmarks

    # ========================================================
    # CAPTURE MODE
    # ========================================================

    if not puzzle_active:

        index_points = []
        pinch_states = []

        valid_hands = []

        for hand in hands:

            if is_index_up(hand):

                valid_hands.append(
                    hand
                )

                p = landmark_point(
                    hand,
                    8,
                    w,
                    h
                )

                index_points.append(p)

                pinch_states.append(
                    pinch_distance(hand)
                    <= PINCH_CLOSE
                )

        # ----------------------------------------------------
        # TWO INDEX FINGERS
        # ----------------------------------------------------

        if len(index_points) >= 2:

            point_a = index_points[0]
            point_b = index_points[1]

            smooth_a = smooth_point(
                smooth_a,
                point_a
            )

            smooth_b = smooth_point(
                smooth_b,
                point_b
            )

            draw_capture_rectangle(
                frame
            )

        else:

            smooth_a = None
            smooth_b = None

        # ----------------------------------------------------
        # BOTH PINCH
        # ----------------------------------------------------

        both_pinching = (
            len(pinch_states) >= 2
            and
            pinch_states[0]
            and
            pinch_states[1]
        )

        if both_pinching and not countdown_active:

            if pinch_start is None:

                pinch_start = time.time()

            held = (
                time.time()
                -
                pinch_start
            )

            progress = min(
                held / PINCH_HOLD_TIME,
                1.0
            )

            # Progress bar
            bar_w = 300
            bar_h = 12

            bx = (
                w - bar_w
            ) // 2

            by = h - 65

            cv2.rectangle(
                frame,
                (bx, by),
                (
                    bx + bar_w,
                    by + bar_h
                ),
                (50, 50, 50),
                -1
            )

            cv2.rectangle(
                frame,
                (bx, by),
                (
                    bx +
                    int(
                        bar_w *
                        progress
                    ),
                    by + bar_h
                ),
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                "HOLD TO CAPTURE",
                (
                    bx + 65,
                    by - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if held >= PINCH_HOLD_TIME:

                countdown_active = True

                countdown_start = time.time()

                pinch_start = None

        else:

            if not countdown_active:

                pinch_start = None

        # ----------------------------------------------------
        # COUNTDOWN
        # ----------------------------------------------------

        if countdown_active:

            elapsed = (
                time.time()
                -
                countdown_start
            )

            remaining = (
                COUNTDOWN_TIME
                -
                elapsed
            )

            if remaining > 0:

                number = int(
                    math.ceil(
                        remaining
                    )
                )

                text = str(number)

                font = cv2.FONT_HERSHEY_SIMPLEX

                scale = 4.5
                thickness = 10

                size = cv2.getTextSize(
                    text,
                    font,
                    scale,
                    thickness
                )[0]

                tx = (
                    w -
                    size[0]
                ) // 2

                ty = (
                    h +
                    size[1]
                ) // 2

                # Shadow
                cv2.putText(
                    frame,
                    text,
                    (tx, ty),
                    font,
                    scale,
                    (0, 0, 0),
                    25,
                    cv2.LINE_AA
                )

                # Number
                cv2.putText(
                    frame,
                    text,
                    (tx, ty),
                    font,
                    scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

            else:

                if capture_image(frame):

                    print(
                        "Photo captured!"
                    )

                countdown_active = False

        # ----------------------------------------------------
        # CAPTURE MODE INSTRUCTIONS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "2 FINGERS = SELECT AREA",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # PUZZLE MODE
    # ========================================================

    else:

        hand_point = None
        hand_pinch = False

        active_hand = None

        # ----------------------------------------------------
        # FIND HAND
        # ----------------------------------------------------

        if len(hands) > 0:

            # Use the first detected hand
            active_hand = hands[0]

            raw_point = landmark_point(
                active_hand,
                8,
                w,
                h
            )

            # Smooth hand point
            hand_point = smooth_point(
                hand_point,
                raw_point
            )

            hand_pinch = update_pinch_state(
                active_hand
            )

        else:

            hand_point = None

            # IMPORTANT:
            # Don't instantly release when tracking
            # briefly disappears.
            #
            # The tile stays grabbed until the hand
            # comes back and actually unpinches.

        # ----------------------------------------------------
        # DRAGGING
        # ----------------------------------------------------

        if (
            dragging_tile is not None
            and
            hand_point is not None
        ):

            drag_current_point = hand_point

            # ------------------------------------------------
            # RELEASE
            # ------------------------------------------------

            if not hand_pinch:

                destination = get_tile_position(
                    hand_point,
                    puzzle_rect
                )

                drop_tile(
                    destination
                )

        # ----------------------------------------------------
        # NEW PINCH / GRAB
        # ----------------------------------------------------

        elif (
            dragging_tile is None
            and
            hand_point is not None
            and
            hand_pinch
        ):

            tile_position = get_tile_position(
                hand_point,
                puzzle_rect
            )

            if tile_position is not None:

                tile_number = puzzle_state[
                    tile_position
                ]

                # Can't grab empty space
                if tile_number != 8:

                    dragging_tile = tile_number

                    drag_source_position = (
                        tile_position
                    )

                    drag_current_point = (
                        hand_point
                    )

        # ====================================================
        # DRAW PUZZLE
        # ====================================================

        frame = draw_puzzle(
            frame
        )

        # ====================================================
        # DRAW DRAGGED TILE
        # ====================================================

        draw_dragged_tile(
            frame
        )

        # ====================================================
        # DRAW REAL HAND
        #
        # THIS IS DRAWN LAST SO YOUR HAND REMAINS VISIBLE
        # OVER THE PUZZLE.
        # ====================================================

        for hand in hands:

            draw_hand(
                frame,
                hand
            )

        # ====================================================
        # CURSOR
        # ====================================================

        draw_cursor(
            frame
        )

        # ====================================================
        # HEADER
        # ====================================================

        draw_puzzle_header(
            frame
        )

        # ====================================================
        # SOLVED
        # ====================================================

        if check_solved():

            if not solved_message:

                solved_message = True

                solved_message_time = (
                    time.time()
                )

            draw_solved(
                frame
            )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Puzzle Cam",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if (
        key == ord("q")
        or
        key == 27
    ):
        break

    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    elif key == ord("s"):

        if puzzle_active:

            reset_puzzle()

            print(
                "Puzzle shuffled."
            )

    # --------------------------------------------------------
    # RESTART
    # --------------------------------------------------------

    elif key == ord("r"):

        if puzzle_active:

            reset_puzzle()

            print(
                "Puzzle restarted."
            )

    # --------------------------------------------------------
    # NEW CAPTURE
    # --------------------------------------------------------

    elif key == ord("n"):

        puzzle_active = False

        captured_image = None

        captured_tiles = None

        puzzle_rect = None

        solved_message = False

        dragging_tile = None

        drag_source_position = None

        drag_current_point = None

        pinch_start = None

        countdown_active = False

        pinch_state = False

        point_a = None
        point_b = None

        smooth_a = None
        smooth_b = None

        print(
            "Ready for a new capture."
        )

    # --------------------------------------------------------
    # MANUAL CAPTURE
    # --------------------------------------------------------

    elif key == ord("c"):

        if not puzzle_active:

            if capture_image(frame):

                print(
                    "Photo captured. "
                    "Solve the puzzle!"
                )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()