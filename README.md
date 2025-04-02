# Hand Gesture Control Systems

This repository contains two powerful computer vision-based gesture control applications:

1. **AI Virtual Keyboard** - Type without touching your physical keyboard
2. **Advanced Virtual Mouse** - Control your cursor with hand gestures

Both applications use webcam input to track hand movements and translate them into keyboard or mouse actions, providing a touchless interface for computer interaction.

## Features

### AI Virtual Keyboard
- Virtual on-screen keyboard with visual feedback
- Type by making "clicking" gestures with your hand
- Text editing capabilities and cursor control
- Word suggestions and autocorrection
- Special keys (Space, Backspace, Shift, Enter, etc.)
- Page navigation controls
- Zoom functionality
- Audio feedback for keypresses

### Advanced Virtual Mouse
- Control cursor movement with hand gestures
- Perform left and right clicks
- Drag and drop functionality
- Scrolling (single and continuous modes)
- Zoom control
- Multiple gesture modes with visual indicators

## Requirements

```
opencv-python
numpy
mediapipe
autopy
pyautogui
pygame
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hand-gesture-control.git
cd hand-gesture-control
```

2. Install dependencies:
```bash
pip install opencv-python numpy mediapipe autopy pyautogui pygame
```

## Usage

### Running the AI Virtual Keyboard
```bash
python virtual_keyboard.py
```

### Running the Advanced Virtual Mouse
```bash
python virtual_mouse.py
```

## Gesture Controls

### AI Virtual Keyboard
- Move your hand to position the cursor
- Bring your index and middle fingers together to "click" keys
- Distance between fingers must be less than 40 pixels to register a click

### Advanced Virtual Mouse
- **Two fingers up** (index and middle): Move cursor
- **Middle finger down, index up**: Right-click
- **Index finger down, middle up**: Left-click
- **Closed fist**: Start drag operation (hold and move)
- **Opening hand after drag**: End drag operation
- **Three fingers up**: Single scroll mode (move hand up/down to scroll)
- **Four fingers up**: Continuous scroll mode with variable speed
- **Five fingers up**: Zoom mode (move hand up to zoom in, down to zoom out)

## Implementation Details

Both applications use MediaPipe's hand tracking solution to detect hand landmarks in real-time. The systems process these landmarks to determine finger positions and gestures, which are then mapped to keyboard or mouse actions.

Key components include:
- Hand detection and tracking
- Gesture recognition algorithms
- Coordinate mapping and transformation
- User interface elements
- Performance optimization for real-time operation

## Customization

You can modify these parameters in the code to customize your experience:

### AI Virtual Keyboard
- Keyboard layout and key size
- Click detection threshold
- Sound effects
- Word suggestion dictionary
- Auto-correction pairs

### Advanced Virtual Mouse
- Gesture cooldown times
- Scroll speed
- Movement smoothening factor
- Click detection sensitivity

## Troubleshooting

- **Poor detection**: Ensure you have good lighting conditions
- **Cursor jitter**: Increase the smoothening factor
- **Accidental clicks**: Adjust the click detection threshold
- **Performance issues**: Reduce camera resolution or close background applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for their excellent hand tracking solution
- OpenCV community for computer vision tools
- PyAutoGUI developers for system control capabilities
