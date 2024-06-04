from htbuilder.units import unit

# This works in 3.7+:
# from htbuilder.units import rem
#
# ...so we use the 3.7 version of the code above here:
rem = unit.rem


# Colors from the Streamlit palette.
# These are red-70, orange-70, ..., violet-70, gray-70.
PALETTE = [
    "#ff4b4b",
    "#ffa421",
    "#ffe312",
    "#21c354",
    "#00d4b1",
    "#00c0f2",
    "#1c83e1",
    "#803df5",
    "#808495",
]

OPACITIES = [
    "33", "66",
]

PADDING=(rem(0.25), rem(0.5))
BORDER_RADIUS=rem(0.5)
LABEL_FONT_SIZE=rem(0.75)
LABEL_OPACITY=0.5
LABEL_SPACING=rem(0.5)
SHOW_LABEL_SEPARATOR=True
